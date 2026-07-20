#!/usr/bin/env python3
"""
labeled_16s_build.py  --  build the Model B training set (labeled 16S genes).

Model B trains on the 16S gene extracted from the SAME labeled genomes used by
the whole-genome GOMICS classifier: pathogen vs non-pathogen. This script runs
barrnap on each source genome, keeps its longest >=MIN_LEN 16S, and writes a
single FASTA whose headers encode  id|label|genus  -- the exact format consumed
by train_16s_model.py and validate_16s_model.py.

Target composition (as in the paper): 826 pathogen + 606 non-pathogen genes.

Usage:
    python labeled_16s_build.py \
        --pathogen-genomes  /path/pathogen_genomes \
        --nonpathogen-genomes /path/non_pathogen_genomes \
        --out saved_model_16s/labeled_16s.fasta
"""
import argparse, re, subprocess, sys
from pathlib import Path

def genus_of(name):
    m = re.match(r"([A-Z][a-z]+)", str(name))
    return m.group(1) if m else "unknown"

def read_fasta(fna):
    seqs, name = {}, None
    for line in open(fna):
        if line.startswith(">"):
            name = line[1:].split()[0]; seqs[name] = []
        elif name is not None:
            seqs[name].append(line.strip())
    return {k: "".join(v) for k, v in seqs.items()}

def longest_16s(fna, min_len):
    try:
        gff = subprocess.run(f"barrnap --kingdom bac --quiet {fna}",
                             shell=True, text=True, capture_output=True).stdout
    except Exception:
        return None
    hits = []
    for ln in gff.splitlines():
        if ln.startswith("#") or "\t" not in ln:
            continue
        c = ln.split("\t")
        if len(c) >= 9 and c[2] == "rRNA" and "Name=16S_rRNA" in c[8]:
            hits.append((c[0], int(c[3]), int(c[4]), c[6]))
    if not hits:
        return None
    seqs = read_fasta(fna)
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    best = ""
    for cid, s, e, strand in hits:
        sub = seqs.get(cid, "")[s-1:e]
        if strand == "-":
            sub = sub.translate(comp)[::-1]
        if len(sub) > len(best):
            best = sub
    return best if len(best) >= min_len else None

def build(genome_dir, label, out_handle, min_len):
    n = 0
    genome_dir = Path(genome_dir)
    files = list(genome_dir.glob("*.fna")) + list(genome_dir.glob("*.fasta")) + \
            list(genome_dir.glob("*.fa"))
    for fna in sorted(files):
        seq = longest_16s(fna, min_len)
        if seq:
            out_handle.write(f">{fna.stem}|{label}|{genus_of(fna.stem)}\n{seq}\n")
            n += 1
    return n

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pathogen-genomes", required=True)
    ap.add_argument("--nonpathogen-genomes", required=True)
    ap.add_argument("--out", default="saved_model_16s/labeled_16s.fasta")
    ap.add_argument("--min-len", type=int, default=400)
    args = ap.parse_args()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        n_path = build(args.pathogen_genomes, "pathogen", fh, args.min_len)
        n_non = build(args.nonpathogen_genomes, "non_pathogen", fh, args.min_len)
    if n_path + n_non == 0:
        sys.exit("No 16S genes recovered -- check genome directories and that "
                 "barrnap is installed.")
    print(f"labeled 16S: {n_path} pathogen + {n_non} non-pathogen "
          f"= {n_path + n_non} genes -> {out}")

if __name__ == "__main__":
    main()
