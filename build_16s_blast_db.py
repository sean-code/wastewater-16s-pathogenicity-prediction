#!/usr/bin/env python3
"""
build_16s_blast_db.py  --  build a Model A BLAST reference from a 16S FASTA.

Use this instead of ref_16s_download.py when you have your own curated,
taxonomically labeled 16S reference FASTA (e.g. a filtered NCBI 16S RefSeq
Targeted Loci export). Runs makeblastdb and writes the DB under
    saved_model_16s/ref_16s/<name>

FASTA headers should carry the organism/taxonomy so Model A can assign a label,
e.g.  >NR_112116.2 Escherichia coli ...

Usage:
    python build_16s_blast_db.py --fasta 16S_reference.fasta \
        --out saved_model_16s/ref_16s/ncbi_16s_refseq
"""
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True, help="input 16S reference FASTA")
    ap.add_argument("--out", default="saved_model_16s/ref_16s/ncbi_16s_refseq",
                    help="output BLAST DB prefix")
    ap.add_argument("--parse-seqids", action="store_true", default=True)
    args = ap.parse_args()

    fasta = Path(args.fasta)
    if not fasta.exists():
        sys.exit(f"reference FASTA not found: {fasta}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    n = sum(1 for line in open(fasta) if line.startswith(">"))
    print(f"[ref] {n} sequences in {fasta}")

    cmd = f"makeblastdb -in {fasta} -dbtype nucl -out {out}"
    if args.parse_seqids:
        cmd += " -parse_seqids"
    print("[$]", cmd)
    subprocess.run(cmd, shell=True, check=True, text=True)
    print("Model A BLAST DB written ->", out)

if __name__ == "__main__":
    main()
