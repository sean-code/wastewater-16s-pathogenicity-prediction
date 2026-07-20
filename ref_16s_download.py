#!/usr/bin/env python3
"""
ref_16s_download.py  --  build the Model A reference (16S nearest-neighbour DB).

Downloads NCBI's prebuilt 16S rRNA BLAST database (RefSeq Targeted Loci,
~26,877 type-strain 16S sequences) and places it under
    saved_model_16s/ref_16s/
so pathogen_predict.py / validate_16s_model.py can use it as the Model A
reference. Mirrors the MAG repo's reference-DB setup (fix_build_blast_db.py).

Usage:
    python ref_16s_download.py [--outdir saved_model_16s/ref_16s]
"""
import argparse, subprocess, sys
from pathlib import Path

NCBI_16S_URL = "https://ftp.ncbi.nlm.nih.gov/blast/db/16S_ribosomal_RNA.tar.gz"

def sh(cmd, cwd=None):
    print("[$]", cmd)
    subprocess.run(cmd, shell=True, check=True, text=True, cwd=cwd)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="saved_model_16s/ref_16s")
    ap.add_argument("--url", default=NCBI_16S_URL)
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    if (out / "16S_ribosomal_RNA.nsq").exists():
        print("Model A reference already present in", out); return

    try:
        sh(f"wget -q -N {args.url}", cwd=out)
        sh("tar -xzf 16S_ribosomal_RNA.tar.gz", cwd=out)
    except subprocess.CalledProcessError as e:
        sys.exit(f"Download/extract failed ({e}). If offline, fetch {args.url} "
                 f"manually into {out} and extract, or use build_16s_blast_db.py "
                 f"on a local reference FASTA.")

    try:
        info = subprocess.run(
            f"blastdbcmd -db {out/'16S_ribosomal_RNA'} -info",
            shell=True, text=True, capture_output=True).stdout
        print(info.splitlines()[0] if info else
              "DB built; run 'blastdbcmd -info' to confirm ~26,877 sequences.")
    except Exception:
        pass
    print("Model A reference ready ->", out)

if __name__ == "__main__":
    main()
