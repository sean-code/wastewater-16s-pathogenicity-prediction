#!/usr/bin/env python3
"""
extract_16s.py -- pull each MAG's 16S rRNA gene from barrnap's GFF (reliable),
slicing the coordinates out of the genome ourselves. Builds the genome-resolved
16S rep-seq FASTA for `pathogen_predict.py 16s`.

Per genome: run barrnap (GFF on stdout), keep Name=16S_rRNA hits, extract each
hit's subsequence (reverse-complement on '-' strand), keep the LONGEST as the
genome's representative 16S (>= MIN_LEN bp). Writes:
  - per_mag/<acc>.16s.fna     one representative 16S per genome (with 16S)
  - all_16s.fna               combined rep-seq FASTA (header = <acc>__16S)
  - manifest_16s.tsv          accession, n_16S_hits, rep_len_bp, status

Requires barrnap. Run:
  MAGS=... OUT16S=... python extract_16s.py
  KEEP_ALL=1  -> keep every 16S copy per genome instead of just the longest
"""
import os, sys, subprocess, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

MAGS    = Path(os.environ.get("MAGS",  str(Path("~/datasets/04_ncbi_wastewater_metagenome_527639/mags_input").expanduser())))
OUT16S  = Path(os.environ.get("OUT16S", str(Path("~/datasets/05_16s_from_mags").expanduser())))
KINGDOM = os.environ.get("BARRNAP_KINGDOM", "bac")
WORKERS = int(os.environ.get("WORKERS", "8"))
MIN_LEN = int(os.environ.get("MIN_LEN", "400"))
KEEP_ALL = os.environ.get("KEEP_ALL") == "1"
PER_MAG = OUT16S / "per_mag"; ALL_FA = OUT16S / "all_16s.fna"; MANIFEST = OUT16S / "manifest_16s.tsv"
PER_MAG.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("16s"); log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")); log.addHandler(h)

_RC = str.maketrans("ACGTNacgtn", "TGCANtgcan")
def revcomp(s): return s.translate(_RC)[::-1]

def read_fasta(path):
    name, seq = None, []
    for line in open(path):
        if line.startswith(">"):
            if name is not None: yield name, "".join(seq)
            name, seq = line[1:].strip(), []
        else: seq.append(line.strip())
    if name is not None: yield name, "".join(seq)

def load_contigs(fna):
    d, name, seq = {}, None, []
    for line in open(fna):
        if line.startswith(">"):
            if name is not None: d[name] = "".join(seq)
            name, seq = line[1:].split()[0], []      # key = first token (matches GFF col1)
        else: seq.append(line.strip())
    if name is not None: d[name] = "".join(seq)
    return d

def parse_16s_hits(gff_text):
    """Return list of (contig, start, end, strand) for Name=16S_rRNA rRNA rows."""
    hits = []
    for line in gff_text.splitlines():
        if not line or line.startswith("#"): continue
        c = line.split("\t")
        if len(c) < 9 or c[2] != "rRNA": continue
        if "Name=16S_rRNA" not in c[8]: continue
        try: hits.append((c[0], int(c[3]), int(c[4]), c[6]))
        except ValueError: continue
    return hits

def extract_seqs(fna, hits):
    contigs = load_contigs(fna); out = []
    for cid, s, e, strand in hits:
        ctg = contigs.get(cid)
        if not ctg: continue
        sub = ctg[s - 1:e]                            # GFF is 1-based inclusive
        if strand == "-": sub = revcomp(sub)
        if len(sub) >= MIN_LEN: out.append(sub)
    out.sort(key=len, reverse=True)
    return out

def extract_one(fna):
    acc = fna.stem; dest = PER_MAG / f"{acc}.16s.fna"
    if dest.exists() and dest.stat().st_size > 0:
        recs = list(read_fasta(dest)); return (acc, len(recs), max((len(s) for _, s in recs), default=0), "skip")
    try:
        p = subprocess.run(["barrnap", "--kingdom", KINGDOM, "--quiet", str(fna)],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True, text=True)
    except subprocess.CalledProcessError as e:
        return (acc, 0, 0, f"barrnap_fail:{e.returncode}")
    hits = parse_16s_hits(p.stdout)
    if not hits: return (acc, 0, 0, "no_16S")
    seqs = extract_seqs(fna, hits)
    if not seqs: return (acc, len(hits), 0, "too_short")
    if not KEEP_ALL: seqs = seqs[:1]                 # longest = genome representative
    with open(dest, "w") as o:
        for i, sq in enumerate(seqs, 1):
            hdr = f"{acc}__16S" if (len(seqs) == 1) else f"{acc}__16S_{i}"
            o.write(f">{hdr}\n{sq}\n")
    return (acc, len(hits), len(seqs[0]), "ok")

def main():
    fnas = sorted(MAGS.glob("*.fna"))
    if not fnas: sys.exit(f"No .fna in {MAGS}")
    log.info("Extracting 16S from %d MAGs (min_len=%d, keep_all=%s) -> %s", len(fnas), MIN_LEN, KEEP_ALL, OUT16S)
    rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(extract_one, f): f for f in fnas}
        for i, fut in enumerate(as_completed(futs), 1):
            rows.append(fut.result())
            if i % 250 == 0 or i == len(fnas): log.info("  %d/%d", i, len(fnas))
    with open(ALL_FA, "w") as out:
        for acc, n, ln, st in rows:
            if st in ("ok", "skip"):
                p = PER_MAG / f"{acc}.16s.fna"
                if p.exists(): out.write(p.read_text())
    with open(MANIFEST, "w") as m:
        m.write("accession\tn_16S_hits\trep_len_bp\tstatus\n")
        for acc, n, ln, st in sorted(rows): m.write(f"{acc}\t{n}\t{ln}\t{st}\n")
    c = Counter(st.split(":")[0] for _, _, _, st in rows)
    with_16s = sum(1 for _, _, _, st in rows if st in ("ok", "skip"))
    log.info("DONE: %d MAGs | with 16S: %d (%.1f%%) | %s", len(fnas), with_16s, 100 * with_16s / len(fnas), dict(c))
    log.info("rep-seq FASTA -> %s   manifest -> %s", ALL_FA, MANIFEST)

if __name__ == "__main__":
    main()
