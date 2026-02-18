### simulated reads
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind, pearsonr

# ==========================================
# 1. DATA SIMULATION (Skip this in real interview)
# ==========================================
np.random.seed(42)
n_subs = 20
subjects = [f'SUBJ_{i:02d}' for i in range(n_subs)]
# Metadata: ~12 ASO, ~8 Placebo, 2 Batches
treatments = ['ASO'] * 12 + ['Placebo'] * 8
batches = ['Batch1'] * 10 + ['Batch2'] * 10
np.random.shuffle(treatments) 

# Create Long-Format Metadata (Baseline & Week4 for each subject)
records = []
for s, t, b in zip(subjects, treatments, batches):
    for timepoint in ['Baseline', 'Week4']:
        records.append({
            'subject_id': s, 'treatment': t, 'batch': b, 'timepoint': timepoint
        })
df_meta = pd.DataFrame(records)

# Simulate Omics Data (300 Proteins)
# Add signal to PROT001 in ASO group at Week 4 (Target Engagement)
n_prot = 300
prot_data = np.random.normal(0, 1, (len(df_meta), n_prot))
df_omics = pd.DataFrame(prot_data, columns=[f'PROT{i:03d}' for i in range(1, n_prot+1)])

# Inject Signal: ASO lowers PROT001 at Week 4
aso_w4_mask = (df_meta['treatment'] == 'ASO') & (df_meta['timepoint'] == 'Week4')
df_omics.loc[aso_w4_mask, 'PROT001'] -= 2.0 

# Inject Batch Effect (Batch 2 is higher)
batch2_mask = df_meta['batch'] == 'Batch2'
df_omics.loc[batch2_mask] += 0.5 

# Combine
df = pd.concat([df_meta, df_omics], axis=1)

# ==========================================
# 2. PREPROCESSING & PCA (QC)
# ==========================================
print("--- Step 2: PCA for QC ---")
# Pivot to wide format for PCA (samples x features)
features = [c for c in df.columns if 'PROT' in c]
X = df[features]

pca = PCA(n_components=2)
pca_res = pca.fit_transform(X)
df['PC1'] = pca_res[:, 0]
df['PC2'] = pca_res[:, 1]

# Plot PCA: Check for Batch Effect
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='batch', style='treatment')
plt.title("PCA: Raw Data (Check for Batch Effects)")
plt.show()

# ==========================================
# 3. BATCH CORRECTION
# ==========================================
# Simple mean-centering per batch (Quick fix for interview)
# In production, use ComBat (pycombat)
print("--- Step 3: Batch Correction ---")
for col in features:
    batch_means = df.groupby('batch')[col].transform('mean')
    df[col] = df[col] - batch_means

# ==========================================
# 4. CALCULATE DELTAS (Week 4 - Baseline)
# ==========================================
print("--- Step 4: Calculating Deltas ---")
# Pivot so each row is a subject, cols are Baseline/Week4 values
df_long = df.melt(id_vars=['subject_id', 'timepoint', 'treatment'], value_vars=features)
# df_wide = df_long.pivot_table(index=['subject_id', 'treatment'], 
#                               columns='timepoint', 
#                               values='value').reset_index()

df_wide = (
    df_long
    .pivot_table(
        index=["subject_id", "treatment"],   # include treatment if it exists
        columns=["variable", "timepoint"],
        values="value",
        aggfunc="mean"                       # or another agg if duplicates exist
    )
    .sort_index(axis=1)
)


## delta
import pandas as pd
idx = pd.IndexSlice

delta = df_wide.loc[:, idx[:, "Week4"]] - df_wide.loc[:, idx[:, "Baseline"]]
delta.columns = [f"Delta_{v}" for v in delta.columns.get_level_values(0)]

delta_data = delta.reset_index()  # includes subject_id and treatment

print(type(df_wide.columns))          # should be MultiIndex
print(df_wide.columns.names)          # ['variable', 'timepoint']
print(df_wide.columns[:6])            # see a few (var, timepoint) pairs

print(df_long["timepoint"].unique())

print(type(df_wide.columns))
print(df_wide.columns.names)
print(df_wide.columns[:10])
print(df_wide.columns.levels)


# ==========================================
# 5. TARGET ENGAGEMENT (Boxplots)
# ==========================================

print("Delta_PROT001 in columns?", "Delta_PROT001" in delta_data.columns)
print("Treatments:", delta_data["treatment"].unique())
print(delta_data.columns.tolist())
delta_data[["treatment", "Delta_PROT001"]].isna().sum()
plot_df = delta_data.dropna(subset=["Delta_PROT001"])

#duplicates?
delta_data.columns[delta_data.columns.duplicated()].tolist()
delta_data = delta_data.loc[:, ~delta_data.columns.duplicated()].copy()

plot_df = delta_data.dropna(subset=["Delta_PROT001"]).copy()
order = sorted(plot_df["treatment"].unique())

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.boxplot(data=plot_df, x="treatment", y="Delta_PROT001", order=order)
sns.stripplot(data=plot_df, x="treatment", y="Delta_PROT001", order=order,
              color="black", alpha=0.5)
plt.axhline(0, linestyle="--", color="gray", linewidth=1)
plt.title("Target Engagement: Change in PROT001")
plt.ylabel("Delta (Week4 - Baseline)")
plt.show()

print("--- Step 5: Target Engagement ---")
plt.figure(figsize=(6, 4))
sns.boxplot(data=delta_data, x='treatment', y='Delta_PROT001')
sns.stripplot(data=delta_data, x='treatment', y='Delta_PROT001', color='black', alpha=0.5)
plt.title("Target Engagement: Change in PROT001")
plt.ylabel("Delta (Week4 - Baseline)")
plt.show()

# ==========================================
# 6. BIOMARKER DISCOVERY (Correlation)
# ==========================================
d
######### Step by step ############
# Source - https://stackoverflow.com/a
# Posted by Gandreoletti
# Retrieved 2026-01-11, License - CC BY-SA 3.0

import os
path="/Users/gaiaandreoletti/Downloads/"
os.chdir(path)

"""
End-to-end multi-omics (RNA-seq + proteomics) analysis script:
- Load metadata + expression matrices
- QC (missingness, sample counts, pairing)
- Transform (RNA log2CPM; proteomics log2 intensity)
- PCA plots (colored by treatment/timepoint)
- Target checks:
    * GENE001 expression across groups/timepoints
    * PROT001 expression across groups/timepoints
    * paired spaghetti plots by subject
    * delta = week4 - baseline, compare deltas between ASO vs placebo
- Optional: feature-wide differential-change scan using OLS on deltas

Assumptions (very common):
- metadata has columns: sample_id, subject_id, treatment, timepoint
  optional: batch, site, dose
- RNA table: rows=genes, cols=samples OR rows=samples, cols=genes
- Proteomics table: rows=proteins, cols=samples OR rows=samples, cols=proteins
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats


# -----------------------------
# 0) Config: EDIT THESE PATHS CHECK input files
# -----------------------------
DATA_DIR = "data"
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
RNA_PATH = os.path.join(DATA_DIR, "transcriptomics.csv")   # counts preferred
PROT_PATH = os.path.join(DATA_DIR, "proteomics.csv")       # intensities preferred

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GENE_OF_INTEREST = "GENE001"
PROT_OF_INTEREST = "PROT001"

# If your labels differ, script tries to normalize them.
BASELINE_LABEL = "baseline"
WEEK4_LABEL = "week4"

## load files
meta = pd.read_csv("data/metadata.csv")
rna  = pd.read_csv("data/transcriptomics.csv")
prot = pd.read_csv("data/proteomics.csv")

print("Files loaded successfully")

print("Metadata shape:", meta.shape)
print("\nMetadata columns:")
print(meta.columns.tolist())

print("\nFirst 5 rows:")
print(meta.head())


print("RNA raw shape:", rna.shape)
print("\nRNA columns (first 10):")
print(rna.columns[:10])
print("\nRNA rows (first 5):")
print(rna.iloc[:5, :5])

## check if GEN001 exists
print("GENE001 present:", "GENE001" in rna.iloc[:, 0].values)


print("Proteomics raw shape:", prot.shape)
print("\nProteomics columns (first 10):")
print(prot.columns[:10])
print("\nProteomics rows (first 5):")
print(prot.iloc[:5, :5])
print("PROT001 present:", "PROT001" in prot.iloc[:, 0].values)


##  Convert RNA & proteomics to “samples × features”
# RNA: genes x samples → samples x genes
rna_mat = (
    rna
    .set_index(rna.columns[0])   # gene column
    .T
)
rna_mat.index.name = "sample_id"

# Proteomics: proteins x samples → samples x proteins
prot_mat = (
    prot
    .set_index(prot.columns[0])  # protein column
    .T
)
prot_mat.index.name = "sample_id"

print("RNA matrix shape (samples x genes):", rna_mat.shape)
print("Proteomics matrix shape (samples x proteins):", prot_mat.shape)

# -----------------------------
# 1) I/O helpers
# -----------------------------
def read_table(path: str) -> pd.DataFrame:
    """Read csv/tsv by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported extension for {path}")


def load_matrix_flexible(path: str) -> pd.DataFrame:
    """
    Load expression matrix that might be:
      A) features x samples, with first column = feature IDs
      B) samples x features, with a sample_id column
      C) already has index set in file (rare)

    Returns a DataFrame shaped: samples x features (index=sample_id).
    """
    df = read_table(path)

    # Case B: if a column looks like sample_id, use it as index
    for c in df.columns:
        if c.lower() in ("sample_id", "sample", "sid"):
            df = df.set_index(c)
            df.index.name = "sample_id"
            return df.apply(pd.to_numeric, errors="coerce")

    # Case A: first column is feature IDs (strings), rest mostly numeric -> transpose
    first_col = df.columns[0]
    if df[first_col].dtype == object:
        # assess numeric-ness of remaining data
        stacked = pd.to_numeric(df[df.columns[1:]].stack(), errors="coerce")
        numeric_ratio = stacked.notna().mean()
        if numeric_ratio > 0.8:
            feat_by_sample = df.set_index(first_col)
            sample_by_feat = feat_by_sample.T
            sample_by_feat.index.name = "sample_id"
            return sample_by_feat.apply(pd.to_numeric, errors="coerce")

    # Otherwise assume df is already sample x feature with implicit sample index
    # (this is riskier; prefer explicit sample_id column)
    if df.index.name != "sample_id":
        df.index.name = "sample_id"
    return df.apply(pd.to_numeric, errors="coerce")


# -----------------------------
# 2) Metadata cleaning
# -----------------------------
def normalize_treatment(x: str) -> str:
    x0 = str(x).strip().lower()
    if re.search(r"plac", x0):
        return "placebo"
    return "ASO"


def normalize_timepoint(x: str) -> str:
    x0 = str(x).strip().lower()
    if x0 in ("baseline", "base", "week0", "wk0", "0", "t0"):
        return BASELINE_LABEL
    if x0 in ("week4", "wk4", "4", "t4"):
        return WEEK4_LABEL
    return str(x).strip()


def load_and_clean_metadata(path: str) -> pd.DataFrame:
    meta = read_table(path)

    # Rename here if your columns are different
    required = {"sample_id", "subject_id", "treatment", "timepoint"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(
            f"Metadata missing columns: {missing}. "
            f"Please rename columns to include: {sorted(required)}"
        )

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["subject_id"] = meta["subject_id"].astype(str)

    meta["treatment"] = meta["treatment"].map(normalize_treatment)
    meta["timepoint"] = meta["timepoint"].map(normalize_timepoint)

    # Make ordered categoricals (useful for plotting)
    meta["treatment"] = pd.Categorical(meta["treatment"], categories=["placebo", "ASO"], ordered=True)
    meta["timepoint"] = pd.Categorical(meta["timepoint"], categories=[BASELINE_LABEL, WEEK4_LABEL], ordered=True)

    # Optional covariates if present
    for c in ("batch", "site", "dose"):
        if c in meta.columns:
            # keep as-is; will be used later if needed
            pass

    return meta


# -----------------------------
# 3) QC helpers
# -----------------------------
def qc_sample_counts(meta: pd.DataFrame, label: str) -> None:
    print(f"\n=== QC: sample counts ({label}) ===")
    print(pd.crosstab(meta["treatment"], meta["timepoint"], dropna=False))


def qc_pairing(meta: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Check each subject has baseline + week4 samples.
    Returns a table with baseline/week4 counts.
    """
    tab = (
        meta.groupby(["subject_id", "timepoint"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )
    # Make sure columns exist even if absent
    if BASELINE_LABEL not in tab.columns:
        tab[BASELINE_LABEL] = 0
    if WEEK4_LABEL not in tab.columns:
        tab[WEEK4_LABEL] = 0

    tab["has_both"] = (tab[BASELINE_LABEL] > 0) & (tab[WEEK4_LABEL] > 0)

    print(f"\n=== QC: pairing ({label}) ===")
    print(tab.sort_values("has_both", ascending=True).head(20))
    print(f"Paired subjects: {tab['has_both'].sum()} / {len(tab)}")
    return tab


def align_meta_and_matrix(meta: pd.DataFrame, mat: pd.DataFrame, label: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Keep only overlapping sample IDs; align ordering to matrix columns (index).
    mat must be samples x features, index=sample_id.
    """
    if mat.index.name != "sample_id":
        mat = mat.copy()
        mat.index.name = "sample_id"

    common = meta["sample_id"].isin(mat.index)
    meta2 = meta.loc[common].copy()
    mat2 = mat.loc[meta2["sample_id"].values].copy()  # match order
    print(f"\nAligned {label}: {len(meta2)} samples")
    return meta2, mat2


def feature_missingness(mat: pd.DataFrame) -> pd.Series:
    """Fraction missing per feature (column)."""
    return mat.isna().mean(axis=0)


# -----------------------------
# 4) Transforms
# -----------------------------
def log2_cpm(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """
    Convert raw counts (samples x genes) to log2(CPM + 1).
    Works for QC/PCA/visualization; not a substitute for DESeq2-style modeling.
    """
    counts = counts.copy()
    counts = counts.clip(lower=0)
    libsize = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(libsize, axis=0) * 1e6
    return np.log2(cpm + pseudocount)


def log2_intensity(intensities: pd.DataFrame) -> pd.DataFrame:
    """Log2 transform proteomics intensities (samples x proteins)."""
    x = intensities.copy()
    # if zeros exist, log2 will become -inf; treat zeros as missing
    x = x.replace(0, np.nan)
    return np.log2(x)


# -----------------------------
# 5) PCA plotting
# -----------------------------
def run_pca_plot(
    mat: pd.DataFrame,
    meta: pd.DataFrame,
    title: str,
    outfile: str,
    color_by: str = "treatment",
    marker_by: str = "timepoint",
    n_components: int = 2,
) -> None:
    """
    PCA on samples.
    mat: samples x features
    - drops features with any NaNs (strict) OR you can do imputation beforehand
    - standardizes features
    """
    # Drop features with missing values for PCA simplicity
    mat2 = mat.loc[:, mat.notna().all(axis=0)].copy()
    if mat2.shape[1] < 2:
        print(f"[WARN] PCA skipped for {title}: too few complete features after dropping NaNs.")
        return

    X = mat2.values
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    pca = PCA(n_components=n_components, random_state=0)
    pcs = pca.fit_transform(Xs)

    df = meta.copy()
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    # Simple categorical mapping to markers
    markers = {BASELINE_LABEL: "o", WEEK4_LABEL: "s"}
    unique_markers = list(df[marker_by].dropna().unique())

    plt.figure(figsize=(7, 5))
    for tp in unique_markers:
        sub = df[df[marker_by] == tp]
        plt.scatter(sub["PC1"], sub["PC2"], marker=markers.get(tp, "o"), alpha=0.85, label=str(tp))

    # Color by treatment using separate passes (keeps matplotlib-only)
    # We'll re-plot with edgecolors by treatment to avoid custom palettes.
    # (If you prefer full color mapping, you can set a cmap, but no need.)
    for tr in df[color_by].dropna().unique():
        sub = df[df[color_by] == tr]
        plt.scatter(sub["PC1"], sub["PC2"], facecolors="none", alpha=0.9, label=f"{tr} (outline)")

    var = pca.explained_variance_ratio_
    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved PCA: {outfile}")


# -----------------------------
# 6) Target extraction + plots
# -----------------------------
def extract_feature_long(
    mat: pd.DataFrame,
    meta: pd.DataFrame,
    feature_id: str,
    value_name: str,
) -> pd.DataFrame:
    """
    mat: samples x features
    returns long DF with sample_id, subject_id, treatment, timepoint, value
    """
    if feature_id not in mat.columns:
        raise KeyError(f"{value_name} feature not found: {feature_id}")

    df = meta[["sample_id", "subject_id", "treatment", "timepoint"]].copy()
    df["value"] = pd.to_numeric(mat[feature_id].values, errors="coerce")
    df["feature"] = feature_id
    df["layer"] = value_name
    return df


def plot_expression_by_group(df_long: pd.DataFrame, title: str, outfile: str) -> None:
    """
    Box + jitter per (treatment, timepoint). Matplotlib-only.
    """
    # Create group labels
    df = df_long.copy()
    df["group"] = df["treatment"].astype(str) + " / " + df["timepoint"].astype(str)

    groups = [g for g in df["group"].unique() if g != "nan / nan"]
    groups_sorted = sorted(groups, key=lambda x: (("placebo" not in x), ("baseline" not in x)))  # mild ordering

    data = [df.loc[df["group"] == g, "value"].dropna().values for g in groups_sorted]

    plt.figure(figsize=(9, 4))
    plt.boxplot(data, labels=groups_sorted, showfliers=False)
    # jitter points
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        x = np.random.normal(i, 0.04, size=len(vals))
        plt.scatter(x, vals, alpha=0.75)

    plt.title(title)
    plt.ylabel("Expression (transformed scale)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")


def plot_spaghetti(df_long: pd.DataFrame, title: str, outfile: str) -> None:
    """
    Paired lines baseline -> week4 per subject, faceted-ish by treatment using separate plots.
    """
    df = df_long.copy()
    # Keep only baseline & week4
    df = df[df["timepoint"].isin([BASELINE_LABEL, WEEK4_LABEL])].copy()

    # Two panels manually
    treatments = [t for t in ["placebo", "ASO"] if t in df["treatment"].astype(str).unique()]
    n = len(treatments)
    if n == 0:
        print(f"[WARN] No treatments found for spaghetti plot: {title}")
        return

    fig, axes = plt.subplots(1, n, figsize=(6*n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, tr in zip(axes, treatments):
        sub = df[df["treatment"].astype(str) == tr].copy()
        for sid, g in sub.groupby("subject_id"):
            # Ensure order baseline->week4
            g2 = g.sort_values("timepoint")
            ax.plot(g2["timepoint"].astype(str), g2["value"], alpha=0.5)
            ax.scatter(g2["timepoint"].astype(str), g2["value"], s=25, alpha=0.9)

        ax.set_title(f"{title} ({tr})")
        ax.set_xlabel("")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Expression (transformed scale)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")


def compute_subject_delta(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute delta = week4 - baseline per subject.
    Returns subject-level table with treatment and delta.
    """
    wide = (
        df_long.pivot_table(
            index=["subject_id", "treatment"],
            columns="timepoint",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    # Ensure both columns exist
    if BASELINE_LABEL not in wide.columns:
        wide[BASELINE_LABEL] = np.nan
    if WEEK4_LABEL not in wide.columns:
        wide[WEEK4_LABEL] = np.nan

    wide["delta"] = wide[WEEK4_LABEL] - wide[BASELINE_LABEL]
    return wide


def compare_deltas(delta_df: pd.DataFrame, label: str) -> None:
    """
    Print summary stats and run:
      - Welch t-test
      - Wilcoxon rank-sum (Mann–Whitney)
    """
    print(f"\n=== Delta comparison: {label} ===")
    summ = delta_df.groupby("treatment")["delta"].agg(["count", "mean", "std"])
    print(summ)

    a = delta_df.loc[delta_df["treatment"].astype(str) == "ASO", "delta"].dropna().values
    p = delta_df.loc[delta_df["treatment"].astype(str) == "placebo", "delta"].dropna().values

    if len(a) >= 2 and len(p) >= 2:
        t = stats.ttest_ind(a, p, equal_var=False, nan_policy="omit")
        print(f"Welch t-test p={t.pvalue:.4g}")
        w = stats.mannwhitneyu(a, p, alternative="two-sided")
        print(f"Mann–Whitney p={w.pvalue:.4g}")
    else:
        print("[WARN] Not enough data for tests (need >=2 per group).")


# -----------------------------
# 7) Optional: feature-wide scan (OLS on deltas)
# -----------------------------
def compute_delta_matrix(meta: pd.DataFrame, mat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create subject-level delta matrix:
      delta(subject, feature) = week4 - baseline

    Returns:
      cov: subject-level covariates (subject_id, treatment, optional batch/site/dose)
      delta_mat: subject x feature
    """
    # Join meta to mat
    df = meta[["sample_id", "subject_id", "treatment", "timepoint"] + [c for c in ["batch", "site", "dose"] if c in meta.columns]].copy()
    df = df.set_index("sample_id").join(mat, how="inner")

    # Split baseline/week4
    base = df[df["timepoint"] == BASELINE_LABEL].copy()
    wk4 = df[df["timepoint"] == WEEK4_LABEL].copy()

    # One row per subject (if duplicates, take first)
    base = base.groupby("subject_id").first()
    wk4 = wk4.groupby("subject_id").first()

    common = base.index.intersection(wk4.index)
    base = base.loc[common]
    wk4 = wk4.loc[common]

    # Covariates (take from wk4)
    cov_cols = ["treatment"] + [c for c in ["batch", "site", "dose"] if c in df.columns]
    cov = wk4[cov_cols].copy()
    cov.index.name = "subject_id"

    # Feature columns = everything not covariates/timepoint/subject_id
    drop_cols = set(cov_cols + ["timepoint"])
    feat_cols = [c for c in wk4.columns if c not in drop_cols]

    delta_mat = wk4[feat_cols] - base[feat_cols]
    return cov.reset_index(), delta_mat.reset_index()


def ols_scan_deltas(cov: pd.DataFrame, delta_mat: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Per-feature OLS:
      delta_feature ~ treatment(ASO vs placebo) + optional batch/site/dose
    Uses BH-FDR correction.

    Returns tidy results.
    """
    df = cov.merge(delta_mat, on="subject_id", how="inner").copy()

    # Treatment indicator
    df["treat_is_ASO"] = (df["treatment"].astype(str) == "ASO").astype(int)

    # Design matrix with optional covariates
    X_parts = [df["treat_is_ASO"]]

    # batch/site as dummies
    for c in ["batch", "site"]:
        if c in df.columns:
            d = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True)
            X_parts.append(d)

    # dose numeric
    if "dose" in df.columns:
        X_parts.append(pd.to_numeric(df["dose"], errors="coerce").fillna(0.0))

    X = pd.concat(X_parts, axis=1)
    X = sm.add_constant(X, has_constant="add")

    feature_cols = [c for c in df.columns if c not in {"subject_id", "treatment", "treat_is_ASO", "batch", "site", "dose"}]

    rows = []
    for feat in feature_cols:
        y = pd.to_numeric(df[feat], errors="coerce")
        mask = y.notna()
        if mask.sum() < 6:
            continue
        fit = sm.OLS(y[mask], X.loc[mask]).fit()
        rows.append({
            "feature": feat,
            "coef_ASO_vs_placebo_on_delta": float(fit.params.get("treat_is_ASO", np.nan)),
            "pvalue": float(fit.pvalues.get("treat_is_ASO", np.nan)),
            "n": int(mask.sum()),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError(f"No features modeled for {label}.")

    res["qvalue_fdr"] = multipletests(res["pvalue"].values, method="fdr_bh")[1]
    res = res.sort_values(["qvalue_fdr", "pvalue"]).reset_index(drop=True)
    return res


# -----------------------------
# 8) Main
# -----------------------------
def main() -> None:
    # ---- Load
    meta = load_and_clean_metadata(META_PATH)
    rna_raw = load_matrix_flexible(RNA_PATH)    # samples x genes
    prot_raw = load_matrix_flexible(PROT_PATH) # samples x proteins

    # ---- Align
    meta_rna, rna_raw = align_meta_and_matrix(meta, rna_raw, label="RNA")
    meta_prot, prot_raw = align_meta_and_matrix(meta, prot_raw, label="Proteomics")

    # ---- QC: counts & pairing
    qc_sample_counts(meta_rna, "RNA")
    qc_pairing(meta_rna, "RNA")

    qc_sample_counts(meta_prot, "Proteomics")
    qc_pairing(meta_prot, "Proteomics")

    # ---- Transform for QC/PCA/plots
    # RNA: log2CPM (works best if RNA matrix is counts)
    rna_logcpm = log2_cpm(rna_raw)

    # Proteomics: log2 intensity
    prot_log2 = log2_intensity(prot_raw)

    # Basic proteomics filtering: keep proteins with >=70% observed values
    miss = feature_missingness(prot_log2)
    keep = miss <= 0.30
    prot_log2_f = prot_log2.loc[:, keep].copy()
    print(f"\nProteomics: kept {prot_log2_f.shape[1]} / {prot_log2.shape[1]} proteins (<=30% missing)")

    # ---- PCA
    run_pca_plot(
        mat=rna_logcpm,
        meta=meta_rna,
        title="RNA PCA (log2CPM)",
        outfile=os.path.join(OUT_DIR, "pca_rna_log2cpm.png"),
    )

    run_pca_plot(
        mat=prot_log2_f,
        meta=meta_prot,
        title="Proteomics PCA (log2 intensity; complete-case features)",
        outfile=os.path.join(OUT_DIR, "pca_proteomics_log2.png"),
    )

    # ---- Target checks
    # RNA target (GENE001)
    if GENE_OF_INTEREST in rna_logcpm.columns:
        gene_df = extract_feature_long(rna_logcpm, meta_rna, GENE_OF_INTEREST, "RNA(log2CPM)")
        plot_expression_by_group(
            gene_df,
            title=f"{GENE_OF_INTEREST} RNA expression (log2CPM)",
            outfile=os.path.join(OUT_DIR, f"{GENE_OF_INTEREST}_rna_by_group.png"),
        )
        plot_spaghetti(
            gene_df,
            title=f"{GENE_OF_INTEREST} RNA spaghetti",
            outfile=os.path.join(OUT_DIR, f"{GENE_OF_INTEREST}_rna_spaghetti.png"),
        )
        gene_delta = compute_subject_delta(gene_df)
        gene_delta.to_csv(os.path.join(OUT_DIR, f"{GENE_OF_INTEREST}_rna_deltas.csv"), index=False)
        compare_deltas(gene_delta, f"{GENE_OF_INTEREST} (RNA log2CPM): Δweek4-baseline")
    else:
        print(f"\n[WARN] {GENE_OF_INTEREST} not found in RNA columns.")
        print("Tip: check available IDs via something like:")
        print("  print([c for c in rna_logcpm.columns if 'GEN' in c][:50])")

    # Proteomics target (PROT001)
    if PROT_OF_INTEREST in prot_log2.columns:
        prot_df = extract_feature_long(prot_log2, meta_prot, PROT_OF_INTEREST, "Protein(log2)")
        plot_expression_by_group(
            prot_df,
            title=f"{PROT_OF_INTEREST} protein expression (log2 intensity)",
            outfile=os.path.join(OUT_DIR, f"{PROT_OF_INTEREST}_protein_by_group.png"),
        )
        plot_spaghetti(
            prot_df,
            title=f"{PROT_OF_INTEREST} protein spaghetti",
            outfile=os.path.join(OUT_DIR, f"{PROT_OF_INTEREST}_protein_spaghetti.png"),
        )
        prot_delta = compute_subject_delta(prot_df)
        prot_delta.to_csv(os.path.join(OUT_DIR, f"{PROT_OF_INTEREST}_protein_deltas.csv"), index=False)
        compare_deltas(prot_delta, f"{PROT_OF_INTEREST} (Protein log2): Δweek4-baseline")
    else:
        print(f"\n[WARN] {PROT_OF_INTEREST} not found in proteomics columns.")
        print("Tip: check available IDs via something like:")
        print("  print([c for c in prot_log2.columns if 'PROT' in c][:50])")

    # ---- Optional: feature-wide scan on deltas (quick screen)
    # RNA wide scan (on log2CPM deltas)
    try:
        cov_rna, delta_rna = compute_delta_matrix(meta_rna, rna_logcpm)
        rna_scan = ols_scan_deltas(cov_rna, delta_rna, label="RNA")
        rna_scan.to_csv(os.path.join(OUT_DIR, "rna_delta_ols_scan.csv"), index=False)
        print("\nRNA delta scan: top 10")
        print(rna_scan.head(10).to_string(index=False))
        if "feature" in rna_scan.columns:
            hit = rna_scan[rna_scan["feature"] == GENE_OF_INTEREST]
            if not hit.empty:
                print(f"\nRNA delta scan entry for {GENE_OF_INTEREST}:")
                print(hit.to_string(index=False))
    except Exception as e:
        print(f"\n[WARN] RNA delta scan skipped/failed: {e}")

    # Proteomics wide scan (on log2 intensity deltas)
    try:
        cov_p, delta_p = compute_delta_matrix(meta_prot, prot_log2_f)
        prot_scan = ols_scan_deltas(cov_p, delta_p, label="Proteomics")
        prot_scan.to_csv(os.path.join(OUT_DIR, "proteomics_delta_ols_scan.csv"), index=False)
        print("\nProteomics delta scan: top 10")
        print(prot_scan.head(10).to_string(index=False))
        hit = prot_scan[prot_scan["feature"] == PROT_OF_INTEREST]
        if not hit.empty:
            print(f"\nProteomics delta scan entry for {PROT_OF_INTEREST}:")
            print(hit.to_string(index=False))
    except Exception as e:
        print(f"\n[WARN] Proteomics delta scan skipped/failed: {e}")

    print(f"\nDone. Outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
