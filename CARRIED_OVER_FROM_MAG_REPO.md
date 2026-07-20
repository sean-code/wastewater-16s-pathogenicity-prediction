# Files to carry over from the whole-genome MAG repo

This repository is the **16S track** and reuses three scripts + the genome-list
inputs from the whole-genome project
[`bicbioeng/wastewater-mag-pathogenicity-prediction`](https://github.com/bicbioeng/wastewater-mag-pathogenicity-prediction).
They are NOT duplicated here to keep a single source of truth. Copy these into the
repo root (or install the MAG repo alongside and point paths at it) before running
Objectives 5 and 9:

| File | Used by | Role |
|------|---------|------|
| `pathogen_predict.py` | Objective 5 | Unified inference tool — the `16s` subcommand (Model A + B + fusion) |
| `feature_importance_genes_go.py` | Objective 9 | Pangenome feature → gene → GO on flagged genomes |
| `go_pathway_prediction.py` | Objective 9 | GO enrichment → pathway summary + plots |
| `pathogen_complete_genomes_fixed.json` | Objective 2 | Pathogen genome list / metadata |
| `non_pathogen_complete_genomes_fixed.json` | Objective 2 | Non-pathogen genome list / metadata |
| `complete_pathogen_genome_only.json` | Objective 2 | Pathogen metadata for BLAST labelling |

Everything else in this repo is self-contained and specific to the 16S track:
`extract_16s.py`, `ref_16s_download.py`, `build_16s_blast_db.py`,
`labeled_16s_build.py`, `train_16s_model.py`, `validate_16s_model.py`,
`kmer_feature_importance_16s.py`, and the notebook.
