# ASO CSF Biomarker Live Coding Exercise

This repo contains a lightweight environment for a 45-minute live coding session focused on multi-omics biomarker analysis in a simulated ASO clinical trial.

---

### Scenario (for candidates)

You will work with synthetic CSF omics data from a small ASO trial targeting a CNS gene (Gene X).
There are 20 subjects (ASO vs placebo), each with baseline and week 4 CSF proteomics and transcriptomics.

**Your goals in the session:**
- Sanity-check and join the data
- Compute within-subject baseline → week 4 changes
- Compare ASO vs placebo
- Propose candidate CSF biomarkers of target engagement

---

### Repository Structure

```
livecoding-master/
├── README.md
├── livecoding.ipynb                  # Main starter notebook for the session
├── multiomics_live_coding_template.ipynb  # Template notebook
│
├── BDS_2025/                         # Primary exercise environment
│   ├── README.md
│   ├── livecoding.ipynb              # Session notebook
│   └── data/
│       ├── README_data.txt
│       ├── meta.csv                  # Subject metadata (20 subjects, treatment, timepoint, batch, site)
│       ├── proteome.csv              # CSF proteomics (~200 proteins, 40 samples)
│       └── transcriptome.csv         # CSF transcriptomics (40 samples)
│
├── mock/                             # Data generation scripts and originals
│   ├── gen_files.py                  # Script used to generate synthetic data
│   ├── helpers.py
│   └── livecoding_orig.ipynb
│
├── GENE001_rna_long.csv              # Processed RNA long-format output
├── GENE001_rna_deltas.csv            # RNA within-subject delta values
├── GENE001_rna_by_group.png          # RNA expression by treatment group (plot)
├── GENE001_rna_spaghetti.png         # RNA spaghetti plot (individual trajectories)
├── pca_proteomics_batch.png          # PCA of proteomics colored by batch
├── pca_proteomics_treatment.png      # PCA of proteomics colored by treatment
├── pca_rna_batch.png                 # PCA of RNA colored by batch
└── pca_rna_treatment.png             # PCA of RNA colored by treatment
```

---

### Data Description

| File | Description |
|------|-------------|
| `meta.csv` | 40 rows (20 subjects × 2 timepoints). Columns: `sample_id`, `subject_id`, `treatment`, `timepoint`, `batch`, `site`, `age`, `sex`, `dose` |
| `proteome.csv` | Log-normalized CSF protein abundances (~200 proteins: PROT001–PROT200+) |
| `transcriptome.csv` | Log-normalized CSF transcript levels (GENE001 is the primary target) |

---

### Getting Started

#### GitHub Codespaces (recommended)
Open a Codespace for this repo, then open `BDS_2025/livecoding.ipynb`.

#### Local setup
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter
jupyter notebook BDS_2025/livecoding.ipynb
```

---

### Notes

- All data is **synthetic** — designed to mimic realistic omics signal with a clear PROT001/GENE001 target engagement effect.
- Batch effects are present intentionally; correction is optional but encouraged.
- The session is time-boxed: prioritize clear reasoning over exhaustive analysis.
