#!/usr/bin/env python3
"""
kmer_feature_importance_16s.py  --  which canonical k-mers drive Model B.

The 16S analog of the MAG repo's feature_importance_genes_go.py (Step 1): loads
the trained Model B Random Forest and reports the top canonical k-mers by
importance, as a CSV + bar plot. (The gene->GO steps do not apply to a single
gene; the genomic interpretation of flagged organisms is provided by the
genome-resolved bridge -- see the notebook, Objective 9.)

Usage:
    python kmer_feature_importance_16s.py \
        --model saved_model_16s/kmer16s_model.pkl \
        --index saved_model_16s/kmer_index.json \
        --top 25 --outdir 16s_pipeline_outputs
"""
import argparse, json, pickle
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="saved_model_16s/kmer16s_model.pkl")
    ap.add_argument("--index", default="saved_model_16s/kmer_index.json")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--outdir", default="16s_pipeline_outputs")
    args = ap.parse_args()

    import pandas as pd
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    rf = pickle.load(open(args.model, "rb"))
    idx = json.load(open(args.index))                 # {index: kmer}
    imp = rf.feature_importances_
    kmers = [idx.get(str(i), str(i)) for i in range(len(imp))]

    df = (pd.DataFrame({"kmer": kmers, "importance": imp})
            .sort_values("importance", ascending=False)
            .head(args.top).reset_index(drop=True))
    csv = out / "top_kmers_model_b.csv"
    df.to_csv(csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.barh(df["kmer"][::-1], df["importance"][::-1], color="#2f6ea5")
    plt.xlabel("Random-forest importance")
    plt.title(f"Top {args.top} canonical k-mers driving Model B")
    plt.tight_layout()
    png = out / "top_kmers_model_b.png"
    plt.savefig(png, dpi=200)
    print(df.head(10).to_string(index=False))
    print(f"\n[done] {csv}\n       {png}")

if __name__ == "__main__":
    main()
