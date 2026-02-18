## Jupyter notebook, replace the plt.savefig(...); plt.close() parts with also plt.show()
# Source - https://stackoverflow.com/a
# Posted by Gandreoletti
# Retrieved 2026-01-11, License - CC BY-SA 3.0

import os
path="/Users/gaiaandreoletti/Downloads/"
os.chdir(path)
import os; print(os.listdir('data/'))

from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats


# -----------------------------
# 0) Config
# -----------------------------
DATA_DIR = "data"
META_PATH = os.path.join(DATA_DIR, "metadata.csv")
RNA_PATH  = os.path.join(DATA_DIR, "transcriptomics.csv")
PROT_PATH = os.path.join(DATA_DIR, "proteomics.csv")

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

GENE_OF_INTEREST = "GENE001"
PROT_OF_INTEREST = "PROT001"

BASELINE_LABEL = "baseline"
WEEK4_LABEL = "week4"

np.random.seed(0)  # for jitter reproducibility

def save_text(s: str, path: str):
    with open(path, "w") as f:
        f.write(s)

print(">>> Step 0: Config")
print("META_PATH:", META_PATH)
print("RNA_PATH :", RNA_PATH)
print("PROT_PATH:", PROT_PATH)
print("OUT_DIR  :", OUT_DIR)
print()

## Files sanity check / load files
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



# -----------------------------
# 1) I/O helpers samples × features with index = sample_id
#Case B: matrix has a sample_id column → set it as index
#
#Case A: first column is feature IDs (gene/protein), rest numeric → set it as index and transpose
#
#Otherwise assumes it’s already sample×feature and converts to numeric
#
# -----------------------------
def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported extension for {path}")


def load_matrix_flexible(path: str) -> pd.DataFrame:
    """
    Returns samples x features with index=sample_id.
    """
    df = read_table(path)

    # Case B: sample_id column exists
    for c in df.columns:
        if c.lower() in ("sample_id", "sample", "sid"):
            df = df.set_index(c)
            df.index.name = "sample_id"
            return df.apply(pd.to_numeric, errors="coerce")

    # Case A: first col is feature IDs; transpose
    first_col = df.columns[0]
    if df[first_col].dtype == object:
        stacked = pd.to_numeric(df[df.columns[1:]].stack(), errors="coerce")
        numeric_ratio = stacked.notna().mean()
        if numeric_ratio > 0.8:
            feat_by_sample = df.set_index(first_col)
            sample_by_feat = feat_by_sample.T
            sample_by_feat.index.name = "sample_id"
            return sample_by_feat.apply(pd.to_numeric, errors="coerce")

    # fallback: assume already sample x feature
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
    required = {"sample_id", "subject_id", "treatment", "timepoint"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")

    meta = meta.copy()
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["subject_id"] = meta["subject_id"].astype(str)
    meta["treatment"] = meta["treatment"].map(normalize_treatment)
    meta["timepoint"] = meta["timepoint"].map(normalize_timepoint)

    meta["treatment"] = pd.Categorical(meta["treatment"], categories=["placebo", "ASO"], ordered=True)
    meta["timepoint"] = pd.Categorical(meta["timepoint"], categories=[BASELINE_LABEL, WEEK4_LABEL], ordered=True)
    return meta


# -----------------------------
# 3) QC helpers
# -----------------------------
def qc_sample_counts(meta: pd.DataFrame, label: str) -> pd.DataFrame:
    tab = pd.crosstab(meta["treatment"], meta["timepoint"], dropna=False)
    print(f"\n>>> QC counts ({label})")
    print(tab)
    tab.to_csv(os.path.join(OUT_DIR, f"qc_counts_{label.lower()}.csv"))
    return tab

def qc_pairing(meta: pd.DataFrame, label: str) -> pd.DataFrame:
    tab = (
        meta.groupby(["subject_id", "timepoint"])
        .size()
        .unstack(fill_value=0)
        .rename_axis(None, axis=1)
        .reset_index()
    )
    if BASELINE_LABEL not in tab.columns:
        tab[BASELINE_LABEL] = 0
    if WEEK4_LABEL not in tab.columns:
        tab[WEEK4_LABEL] = 0

    tab["has_both"] = (tab[BASELINE_LABEL] > 0) & (tab[WEEK4_LABEL] > 0)

    print(f"\n>>> QC pairing ({label})")
    print(tab.head(12))
    print(f"Paired subjects: {tab['has_both'].sum()} / {len(tab)}")

    tab.to_csv(os.path.join(OUT_DIR, f"qc_pairing_{label.lower()}.csv"), index=False)
    return tab

# Keeps only samples present in BOTH metadata and matrix, and prints:lost and missing samples
def align_meta_and_matrix(meta: pd.DataFrame, mat: pd.DataFrame, label: str):
    common = meta["sample_id"].isin(mat.index)
    meta2 = meta.loc[common].copy()
    mat2 = mat.loc[meta2["sample_id"].values].copy()

    print(f"\n>>> Alignment ({label})")
    print("meta samples:", meta.shape[0], "->", meta2.shape[0])
    print("matrix shape:", mat.shape, "->", mat2.shape)

    missing_in_mat = set(meta["sample_id"]) - set(mat.index)
    missing_in_meta = set(mat.index) - set(meta["sample_id"])
    print("Missing in matrix (from meta):", list(sorted(missing_in_mat))[:10], "..." if len(missing_in_mat) > 10 else "")
    print("Missing in meta (from matrix):", list(sorted(missing_in_meta))[:10], "..." if len(missing_in_meta) > 10 else "")

    return meta2, mat2

def feature_missingness(mat: pd.DataFrame) -> pd.Series:
    return mat.isna().mean(axis=0)


# -----------------------------
# 4) Transforms
# -----------------------------
#Converts raw counts to CPM per sample, then log2(CPM + 1).
def log2_cpm(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    counts = counts.copy().clip(lower=0)
    libsize = counts.sum(axis=1).replace(0, np.nan)
    cpm = counts.div(libsize, axis=0) * 1e6
    return np.log2(cpm + pseudocount)

def log2_intensity(intensities: pd.DataFrame) -> pd.DataFrame:
    x = intensities.copy().replace(0, np.nan)
    return np.log2(x)


# -----------------------------
# 5) PCA
# -----------------------------
def run_pca_plot(mat: pd.DataFrame, meta: pd.DataFrame, title: str, outfile: str):
    mat2 = mat.loc[:, mat.notna().all(axis=0)].copy()
    print(f"\n>>> PCA prep: {title}")
    print("Input shape:", mat.shape, "| complete-case features:", mat2.shape[1])

    if mat2.shape[1] < 2:
        print("[WARN] PCA skipped: too few complete features.")
        return

    X = StandardScaler().fit_transform(mat2.values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)

    df = meta.copy()
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    var = pca.explained_variance_ratio_
    markers = {BASELINE_LABEL: "o", WEEK4_LABEL: "s"}

    plt.figure(figsize=(7, 5))
    for tp in df["timepoint"].dropna().unique():
        sub = df[df["timepoint"] == tp]
        plt.scatter(sub["PC1"], sub["PC2"], marker=markers.get(tp, "o"), alpha=0.8, label=f"timepoint={tp}")

    for tr in df["treatment"].dropna().unique():
        sub = df[df["treatment"] == tr]
        plt.scatter(sub["PC1"], sub["PC2"], facecolors="none", alpha=0.9, label=f"treat={tr} (outline)")

    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)


##### batch effect ?

##### QC - Identify PCAs batch effect

def run_pca_plot_with_batch(
    mat,
    meta,
    title,
    outfile,
    color_by="batch",
    shape_by="timepoint"
):
    """
    PCA plot colored by batch (or any metadata column).
    mat: samples x features
    meta: metadata aligned to mat
    """

    # Keep only complete-case features for PCA
    mat2 = mat.loc[:, mat.notna().all(axis=0)]
    print(f"PCA ({title}): using {mat2.shape[1]} features")

    X = StandardScaler().fit_transform(mat2.values)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(X)

    df = meta.copy()
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    var = pca.explained_variance_ratio_

    plt.figure(figsize=(7, 5))
    for b in df[color_by].astype(str).unique():
        sub = df[df[color_by].astype(str) == b]
        plt.scatter(
            sub["PC1"],
            sub["PC2"],
            alpha=0.8,
            label=f"{color_by}={b}"
        )

    plt.title(title)
    plt.xlabel(f"PC1 ({var[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({var[1]*100:.1f}%)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()

#plot
# RNA
run_pca_plot_with_batch(
    mat=rna_logcpm,
    meta=meta_rna,
    title="RNA PCA colored by batch",
    outfile="outputs/pca_rna_batch.png",
    color_by="batch"
)

# Proteomics
run_pca_plot_with_batch(
    mat=prot_log2_f,
    meta=meta_prot,
    title="Proteomics PCA colored by batch",
    outfile="outputs/pca_proteomics_batch.png",
    color_by="batch"
)

### Quantify batch effect
import statsmodels.formula.api as smf

def variance_explained_by_batch(pc_values, meta, covariate):
    df = meta.copy()
    df["PC1"] = pc_values[:, 0]
    df["PC2"] = pc_values[:, 1]

    for pc in ["PC1", "PC2"]:
        model = smf.ols(f"{pc} ~ C({covariate})", data=df).fit()
        print(f"{pc}: R² explained by {covariate} = {model.rsquared:.3f}")

## If R² > ~0.2–0.3, batch is meaningfully influencing PCA.
X = StandardScaler().fit_transform(rna_logcpm.dropna(axis=1).values)
pcs = PCA(n_components=2).fit_transform(X)

variance_explained_by_batch(pcs, meta_rna, covariate="batch")

## correct for batch - not working
from pycombat import Combat

def combat_correct(mat, meta, batch_col, covariates=None):
    """
    mat: samples x features
    meta: metadata aligned
    batch_col: column name for batch
    covariates: list of columns to preserve (e.g. treatment, timepoint)
    """
    if covariates is None:
        covariates = []

    model = Combat(
        batch_col=batch_col,
        covariates=covariates
    )

    corrected = model.fit_transform(
        mat,
        meta
    )
    return pd.DataFrame(corrected, index=mat.index, columns=mat.columns)

# RNA
rna_combat = combat_correct(
    mat=rna_logcpm,
    meta=meta_rna,
    batch_col="batch",
    covariates=["treatment", "timepoint"]
)

# Proteomics
prot_combat = combat_correct(
    mat=prot_log2_f,
    meta=meta_prot,
    batch_col="batch",
    covariates=["treatment", "timepoint"]
)

## new wrapper works - correct for batch -
import numpy as np
import pandas as pd
from pycombat import Combat

def combat_correct_pycombat(mat: pd.DataFrame,
                           meta: pd.DataFrame,
                           batch_col: str,
                           covariates=None,
                           mode: str = "p") -> pd.DataFrame:
    """
    Robust wrapper for pycombat.pycombat.Combat variant.
    Tries common calling conventions and orientations.
    mat: samples x features (DataFrame)
    meta: aligned to mat rows
    """
    if covariates is None:
        covariates = []

    # batch vector
    batch = meta[batch_col].astype(str).values

    # covariate matrix (optional)
    cov = None
    if len(covariates) > 0:
        cov_df = pd.get_dummies(meta[covariates], drop_first=True)
        cov = cov_df.values

    X = mat.values  # samples x features

    model = Combat(mode=mode)

    # ---- Attempt 1: fit_transform(samples x features)
    if hasattr(model, "fit_transform"):
        try:
            out = model.fit_transform(X, batch, cov)
            return pd.DataFrame(out, index=mat.index, columns=mat.columns)
        except TypeError:
            pass
        except Exception:
            pass

        # ---- Attempt 2: fit_transform(features x samples)
        try:
            out = model.fit_transform(X.T, batch, cov)
            # out would be features x samples; transpose back
            out = np.asarray(out).T
            return pd.DataFrame(out, index=mat.index, columns=mat.columns)
        except Exception as e:
            last_err = e

    # ---- Attempt 3: combat(...) method
    if hasattr(model, "combat"):
        try:
            out = model.combat(X, batch, cov)
            return pd.DataFrame(out, index=mat.index, columns=mat.columns)
        except Exception:
            pass
        try:
            out = model.combat(X.T, batch, cov)
            out = np.asarray(out).T
            return pd.DataFrame(out, index=mat.index, columns=mat.columns)
        except Exception as e:
            last_err = e

    raise RuntimeError(
        "Could not run pycombat Combat with available methods/signatures. "
        "Run the method inspection snippet and paste the output."
    )

# rna_combat = combat_correct_pycombat(
#     mat=rna_logcpm,
#     meta=meta_rna,
#     batch_col="batch",
#     covariates=["treatment", "timepoint"],
#     mode="p"
# )

run_pca_plot_with_batch(
    mat=rna_logcpm,
    meta=meta_rna,
    title="RNA PCA BEFORE ComBat (color=batch)",
    outfile="outputs/pca_rna_batch_before.png",
    color_by="batch"
)

run_pca_plot_with_batch(
    mat=rna_combat,
    meta=meta_rna,
    title="RNA PCA AFTER ComBat (color=batch)",
    outfile="outputs/pca_rna_batch_after.png",
    color_by="batch"
)


run_pca_plot_with_batch(
    mat=prot_log2_f,
    meta=meta_prot,
    title="PROT PCA BEFORE ComBat (color=batch)",
    outfile="outputs/pca_prot_batch_before.png",
    color_by="batch"
)

run_pca_plot_with_batch(
    mat=prot_log2_f,
    meta=meta_prot,
    title="PROT PCA AFTER ComBat (color=batch)",
    outfile="outputs/pca_pro_batch_after.png",
    color_by="batch"
)


# -----------------------------
# 6) Target checks
# -----------------------------
def extract_feature_long(mat: pd.DataFrame, meta: pd.DataFrame, feature_id: str, layer: str) -> pd.DataFrame:
    if feature_id not in mat.columns:
        raise KeyError(f"{layer} feature not found: {feature_id}")

    df = meta[["sample_id", "subject_id", "treatment", "timepoint"]].copy()
    df["value"] = pd.to_numeric(mat[feature_id].values, errors="coerce")
    df["feature"] = feature_id
    df["layer"] = layer
    return df

def plot_expression_by_group(df_long: pd.DataFrame, title: str, outfile: str):
    df = df_long.copy()
    df["group"] = df["treatment"].astype(str) + " / " + df["timepoint"].astype(str)

    groups = [g for g in df["group"].unique() if "nan" not in g.lower()]
    groups_sorted = sorted(groups)

    data = [df.loc[df["group"] == g, "value"].dropna().values for g in groups_sorted]

    plt.figure(figsize=(9, 4))
    plt.boxplot(data, labels=groups_sorted, showfliers=False)
    for i, vals in enumerate(data, start=1):
        x = np.random.normal(i, 0.04, size=len(vals))
        plt.scatter(x, vals, alpha=0.75)

    plt.title(title)
    plt.ylabel("Expression (transformed)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)

def plot_spaghetti(df_long: pd.DataFrame, title: str, outfile: str):
    df = df_long.copy()
    df = df[df["timepoint"].isin([BASELINE_LABEL, WEEK4_LABEL])]

    treatments = [t for t in ["placebo", "ASO"] if t in df["treatment"].astype(str).unique()]
    fig, axes = plt.subplots(1, len(treatments), figsize=(6*len(treatments), 4), sharey=True)
    if len(treatments) == 1:
        axes = [axes]

    for ax, tr in zip(axes, treatments):
        sub = df[df["treatment"].astype(str) == tr]
        for sid, g in sub.groupby("subject_id"):
            g2 = g.sort_values("timepoint")
            ax.plot(g2["timepoint"].astype(str), g2["value"], alpha=0.5)
            ax.scatter(g2["timepoint"].astype(str), g2["value"], s=25, alpha=0.9)
        ax.set_title(f"{title} ({tr})")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Expression (transformed)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)

def compute_subject_delta(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = (
        df_long.pivot_table(
            index=["subject_id", "treatment"],
            columns="timepoint",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    if BASELINE_LABEL not in wide.columns:
        wide[BASELINE_LABEL] = np.nan
    if WEEK4_LABEL not in wide.columns:
        wide[WEEK4_LABEL] = np.nan
    wide["delta"] = wide[WEEK4_LABEL] - wide[BASELINE_LABEL]
    return wide

def compare_deltas(delta_df: pd.DataFrame, label: str):
    print(f"\n>>> Delta comparison: {label}")
    print(delta_df.groupby("treatment")["delta"].agg(["count","mean","std"]))

    a = delta_df.loc[delta_df["treatment"].astype(str) == "ASO", "delta"].dropna().values
    p = delta_df.loc[delta_df["treatment"].astype(str) == "placebo", "delta"].dropna().values
    if len(a) >= 2 and len(p) >= 2:
        t = stats.ttest_ind(a, p, equal_var=False)
        w = stats.mannwhitneyu(a, p, alternative="two-sided")
        print(f"Welch t-test p={t.pvalue:.4g}")
        print(f"Mann–Whitney p={w.pvalue:.4g}")
    else:
        print("[WARN] Not enough data for tests.")


# ============================================================
# RUN: step-by-step with outputs printed/saved as we go
# ============================================================

## print steps

print(">>> Step 1: Load data")
meta = load_and_clean_metadata(META_PATH)
rna_raw = load_matrix_flexible(RNA_PATH)
prot_raw = load_matrix_flexible(PROT_PATH)

print("\nMetadata shape:", meta.shape)
print(meta.head())

print("\nRNA raw matrix shape (samples x genes):", rna_raw.shape)
print("RNA sample IDs (first 5):", list(rna_raw.index[:5]))
print("RNA features (first 5):", list(rna_raw.columns[:5]))

print("\nProteomics raw matrix shape (samples x proteins):", prot_raw.shape)
print("Proteomics sample IDs (first 5):", list(prot_raw.index[:5]))
print("Proteomics features (first 5):", list(prot_raw.columns[:5]))

meta.to_csv(os.path.join(OUT_DIR, "loaded_metadata_clean.csv"), index=False)
print("Saved: loaded_metadata_clean.csv")


print("\n>>> Step 2: Align sample IDs")
meta_rna, rna_raw = align_meta_and_matrix(meta, rna_raw, "RNA")
meta_prot, prot_raw = align_meta_and_matrix(meta, prot_raw, "Proteomics")


print("\n>>> Step 3: QC tables")
qc_sample_counts(meta_rna, "RNA")
qc_pairing(meta_rna, "RNA")

qc_sample_counts(meta_prot, "Proteomics")
qc_pairing(meta_prot, "Proteomics")

print("\n>>> Step 4: Transform matrices")
rna_logcpm = log2_cpm(rna_raw)
prot_log2 = log2_intensity(prot_raw)

print("RNA log2CPM shape:", rna_logcpm.shape)
print("RNA log2CPM preview:\n", rna_logcpm.iloc[:3, :3])

print("Proteomics log2 shape:", prot_log2.shape)
print("Proteomics log2 preview:\n", prot_log2.iloc[:3, :3])

rna_logcpm.to_csv(os.path.join(OUT_DIR, "rna_log2cpm_matrix.csv"))
prot_log2.to_csv(os.path.join(OUT_DIR, "proteomics_log2_matrix.csv"))
print("Saved: rna_log2cpm_matrix.csv, proteomics_log2_matrix.csv")


print("\n>>> Step 5: Proteomics missingness filter")
miss = feature_missingness(prot_log2)
miss.to_csv(os.path.join(OUT_DIR, "proteomics_feature_missingness.csv"))
keep = miss <= 0.30
prot_log2_f = prot_log2.loc[:, keep].copy()
print(f"Proteomics kept {prot_log2_f.shape[1]} / {prot_log2.shape[1]} proteins (<=30% missing)")
prot_log2_f.to_csv(os.path.join(OUT_DIR, "proteomics_log2_filtered_matrix.csv"))
print("Saved: proteomics_feature_missingness.csv, proteomics_log2_filtered_matrix.csv")


print("\n>>> Step 6: PCA plots")
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


print("\n>>> Step 7: Target checks (RNA)")
if GENE_OF_INTEREST in rna_logcpm.columns:
    gene_df = extract_feature_long(rna_logcpm, meta_rna, GENE_OF_INTEREST, "RNA(log2CPM)")
    print(gene_df.head(10))

    gene_df.to_csv(os.path.join(OUT_DIR, f"{GENE_OF_INTEREST}_rna_long.csv"), index=False)
    print(f"Saved: {GENE_OF_INTEREST}_rna_long.csv")

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
    print("\nGENE001 deltas preview:")
    print(gene_delta.head(10))

    gene_delta.to_csv(os.path.join(OUT_DIR, f"{GENE_OF_INTEREST}_rna_deltas.csv"), index=False)
    print(f"Saved: {GENE_OF_INTEREST}_rna_deltas.csv")

    compare_deltas(gene_delta, f"{GENE_OF_INTEREST} (RNA log2CPM) Δweek4-baseline")
else:
    print(f"[WARN] {GENE_OF_INTEREST} not found in RNA matrix columns.")


print("\n>>> Step 8: Target checks (Proteomics)")
if PROT_OF_INTEREST in prot_log2.columns:
    prot_df = extract_feature_long(prot_log2, meta_prot, PROT_OF_INTEREST, "Protein(log2)")
    print(prot_df.head(10))

    prot_df.to_csv(os.path.join(OUT_DIR, f"{PROT_OF_INTEREST}_protein_long.csv"), index=False)
    print(f"Saved: {PROT_OF_INTEREST}_protein_long.csv")

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
    print("\nPROT001 deltas preview:")
    print(prot_delta.head(10))

    prot_delta.to_csv(os.path.join(OUT_DIR, f"{PROT_OF_INTEREST}_protein_deltas.csv"), index=False)
    print(f"Saved: {PROT_OF_INTEREST}_protein_deltas.csv")

    compare_deltas(prot_delta, f"{PROT_OF_INTEREST} (Protein log2) Δweek4-baseline")
else:
    print(f"[WARN] {PROT_OF_INTEREST} not found in proteomics matrix columns.")


print("\nAll done. Check outputs/ for plots and CSVs.")

###### Correlation filtering script (Δ-target vs Δ-all)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BASELINE_LABEL = "baseline"
WEEK4_LABEL = "week4"

GENE_TARGET = "GENE001"
PROT_TARGET = "PROT001"


# -----------------------------
# 1) Compute subject-level deltas for all features
# -----------------------------
def compute_delta_wide(meta: pd.DataFrame, mat: pd.DataFrame,
                       feature_kind: str = "feature") -> pd.DataFrame:
    """
    meta: must include sample_id, subject_id, timepoint (baseline/week4)
    mat: samples x features, index must be sample_id
    Returns: subject x features delta table (Δ = week4 - baseline)
    """
    # Join meta to matrix
    df = meta[["sample_id", "subject_id", "timepoint"]].copy()
    df = df.set_index("sample_id").join(mat, how="inner")

    # Split by timepoint
    base = df[df["timepoint"] == BASELINE_LABEL].copy()
    wk4  = df[df["timepoint"] == WEEK4_LABEL].copy()

    # One row per subject per timepoint (if duplicates, average)
    base = base.groupby("subject_id").mean(numeric_only=True)
    wk4  = wk4.groupby("subject_id").mean(numeric_only=True)

    common = base.index.intersection(wk4.index)
    base = base.loc[common]
    wk4  = wk4.loc[common]

    delta = wk4 - base
    delta.index.name = "subject_id"

    print(f"[Delta] {feature_kind}: subjects={delta.shape[0]} features={delta.shape[1]}")
    return delta


# -----------------------------
# 2) Correlate target delta with all other feature deltas
# -----------------------------
def correlate_target(delta: pd.DataFrame, target: str,
                     method: str = "spearman",
                     min_n: int = 8) -> pd.DataFrame:
    """
    delta: subject x features (Δ)
    method: "pearson" or "spearman"
    Returns results table with correlation, pvalue, FDR qvalue.
    """
    if target not in delta.columns:
        raise KeyError(f"Target {target} not found in delta columns.")

    y = pd.to_numeric(delta[target], errors="coerce")

    rows = []
    for feat in delta.columns:
        if feat == target:
            continue
        x = pd.to_numeric(delta[feat], errors="coerce")

        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < min_n:
            continue

        if method == "pearson":
            r, p = stats.pearsonr(x[mask].values, y[mask].values)
        elif method == "spearman":
            r, p = stats.spearmanr(x[mask].values, y[mask].values)
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        rows.append({"feature": feat, "n": n, "corr": r, "pvalue": p})

    res = pd.DataFrame(rows)
    if res.empty:
        raise RuntimeError("No correlations computed. Check min_n or missingness.")

    res["qvalue_fdr"] = multipletests(res["pvalue"].values, method="fdr_bh")[1]
    res = res.sort_values(["qvalue_fdr", "pvalue"]).reset_index(drop=True)
    return res


# -----------------------------
# 3) Plot correlation results (volcano-ish)
# -----------------------------
def plot_corr_results(res: pd.DataFrame, title: str, outfile: str, highlight: list[str] = None):
    """
    x = corr, y = -log10(p)
    """
    df = res.copy()
    df["neglog10p"] = -np.log10(np.clip(df["pvalue"].values, 1e-300, 1.0))

    plt.figure(figsize=(7, 5))
    plt.scatter(df["corr"], df["neglog10p"], alpha=0.6)
    plt.axhline(-np.log10(0.05), linestyle="--") ## FDR −log10(0.05) ≈ 1.301
    plt.axvline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Correlation with Δ target")
    plt.ylabel("-log10(p)")
    plt.tight_layout()

    if highlight:
        for h in highlight:
            hit = df[df["feature"] == h]
            if not hit.empty:
                plt.scatter(hit["corr"], hit["neglog10p"], s=80)
                plt.text(float(hit["corr"]), float(hit["neglog10p"]), h)

    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)


# -----------------------------
# 4) Optional: scatter plot for top hits
# -----------------------------
def plot_scatter(delta: pd.DataFrame, target: str, feature: str, title: str, outfile: str):
    y = delta[target]
    x = delta[feature]
    mask = x.notna() & y.notna()

    plt.figure(figsize=(5, 4))
    plt.scatter(x[mask], y[mask], alpha=0.8)
    # fit line
    if mask.sum() >= 3:
        m, b = np.polyfit(x[mask], y[mask], 1)
        xx = np.linspace(x[mask].min(), x[mask].max(), 100)
        plt.plot(xx, m*xx + b)

    plt.title(title)
    plt.xlabel(f"Δ {feature}")
    plt.ylabel(f"Δ {target}")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)


# ============================================================
# RUN: RNA (ΔGENE001 vs Δall genes)
# ============================================================

# These must exist from your pipeline:
# meta_rna: metadata aligned to rna_logcpm
# rna_logcpm: samples x genes, index=sample_id
# meta_prot: metadata aligned to prot_log2_f (or prot_log2)
# prot_log2_f: samples x proteins, index=sample_id

# --- RNA
delta_rna = compute_delta_wide(meta_rna, rna_logcpm, feature_kind="RNA genes")
rna_corr = correlate_target(delta_rna, target=GENE_TARGET, method="spearman", min_n=8)

rna_corr.to_csv(os.path.join(OUT_DIR, "corr_delta_GENE001_vs_all_genes.csv"), index=False)
print("\nTop RNA features correlated with ΔGENE001:")
print(rna_corr.head(15).to_string(index=False))

plot_corr_results(
    rna_corr,
    title="Correlation: ΔGENE001 vs ΔAll Other Genes (Spearman)",
    outfile=os.path.join(OUT_DIR, "corr_volcano_rna_GENE001.png"),
)

# Scatter plots for top 5 hits
for feat in rna_corr.head(5)["feature"].tolist():
    plot_scatter(
        delta_rna, GENE_TARGET, feat,
        title=f"Δ{GENE_TARGET} vs Δ{feat} (RNA)",
        outfile=os.path.join(OUT_DIR, f"scatter_rna_{GENE_TARGET}_vs_{feat}.png")
    )


# ============================================================
# RUN: Proteomics (ΔPROT001 vs Δall proteins)
# ============================================================

delta_prot = compute_delta_wide(meta_prot, prot_log2_f, feature_kind="Proteins")
prot_corr = correlate_target(delta_prot, target=PROT_TARGET, method="spearman", min_n=8)

prot_corr.to_csv(os.path.join(OUT_DIR, "corr_delta_PROT001_vs_all_proteins.csv"), index=False)
print("\nTop Proteins correlated with ΔPROT001:")
print(prot_corr.head(15).to_string(index=False))

plot_corr_results(
    prot_corr,
    title="Correlation: ΔPROT001 vs ΔAll Other Proteins (Spearman)",
    outfile=os.path.join(OUT_DIR, "corr_volcano_prot_PROT001.png"),
)

# Scatter plots for top 5 hits
for feat in prot_corr.head(5)["feature"].tolist():
    plot_scatter(
        delta_prot, PROT_TARGET, feat,
        title=f"Δ{PROT_TARGET} vs Δ{feat} (Protein)",
        outfile=os.path.join(OUT_DIR, f"scatter_prot_{PROT_TARGET}_vs_{feat}.png")
    )


### DIFF EXPRESSION - like Deseq2
import numpy as np
import pandas as pd

BASELINE = "baseline"
WEEK4 = "week4"

# build paired baseline/week4 matrices per arm
def keep_paired_subjects(meta: pd.DataFrame) -> pd.DataFrame:
    tab = (meta.groupby(["subject_id", "timepoint"]).size().unstack(fill_value=0))
    paired = tab[(tab.get(BASELINE, 0) > 0) & (tab.get(WEEK4, 0) > 0)].index
    return meta[meta["subject_id"].isin(paired)].copy()

def subset_arm(meta: pd.DataFrame, mat: pd.DataFrame, arm: str):
    m = meta[meta["treatment"].astype(str) == arm].copy()
    m = keep_paired_subjects(m)
    mat2 = mat.loc[m["sample_id"]].copy()
    return m, mat2

# Run paired DE for one arm (ASO or placebo) - not working
# from pydeseq2.dds import DeseqDataSet
# from pydeseq2.ds import DeseqStats
# 
# def deseq2_paired_within_arm(meta_arm: pd.DataFrame,
#                              counts_arm: pd.DataFrame,
#                              ref_timepoint: str = BASELINE,
#                              test_timepoint: str = WEEK4) -> pd.DataFrame:
#     """
#     Paired DE within a single arm: week4 vs baseline controlling for subject.
#     meta_arm: rows=samples with subject_id and timepoint
#     counts_arm: samples x genes (raw counts)
#     Returns: results table with log2FoldChange, pvalue, padj
#     """
# 
#     # Ensure only baseline/week4
#     meta_arm = meta_arm[meta_arm["timepoint"].isin([ref_timepoint, test_timepoint])].copy()
#     counts_arm = counts_arm.loc[meta_arm["sample_id"]].copy()
# 
#     # pydeseq2 expects counts as genes x samples OR samples x genes? -> use genes as columns works with their API via DataFrame
#     # We'll provide counts as samples x genes (common) and pydeseq2 handles it.
# 
#     clinical = meta_arm.set_index("sample_id")[["subject_id", "timepoint"]].copy()
#     # categorical timepoint with baseline as reference
#     clinical["timepoint"] = pd.Categorical(clinical["timepoint"], categories=[ref_timepoint, test_timepoint], ordered=True)
# 
#     dds = DeseqDataSet(
#         counts=counts_arm.astype(int),
#         clinical=clinical,
#         design_factors=["subject_id", "timepoint"],  # paired
#         design_factors=["subject_id", "batch", "timepoint"] ## if batch is present
#         refit_cooks=True,
#         n_cpus=1
#     )
#     dds.deseq2()
# 
#     stat_res = DeseqStats(dds, contrast=("timepoint", test_timepoint, ref_timepoint))
#     stat_res.summary()
# 
#     res = stat_res.results_df.copy()
#     # results_df index is gene ids
#     res = res.reset_index().rename(columns={"index": "gene"})
#     return res

# DE working -  Run paired DE for one arm (ASO or placebo) RNA
import pandas as pd
import inspect
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

BASELINE = "baseline"
WEEK4 = "week4"

def deseq2_paired_within_arm(meta_arm: pd.DataFrame,
                             counts_arm: pd.DataFrame,
                             ref_timepoint: str = BASELINE,
                             test_timepoint: str = WEEK4) -> pd.DataFrame:
    """
    Paired DE within one arm: week4 vs baseline controlling for subject_id.
    meta_arm: rows=samples with columns sample_id, subject_id, timepoint
    counts_arm: samples x genes (raw counts, int)
    """

    # keep only the two timepoints
    meta_arm = meta_arm[meta_arm["timepoint"].isin([ref_timepoint, test_timepoint])].copy()
    counts_arm = counts_arm.loc[meta_arm["sample_id"]].copy()

    # metadata indexed by sample_id
    meta_df = meta_arm.set_index("sample_id")[["subject_id", "timepoint"]].copy()
    meta_df["timepoint"] = pd.Categorical(
        meta_df["timepoint"],
        categories=[ref_timepoint, test_timepoint],
        ordered=True
    )

    # Detect correct keyword for metadata argument
    dds_sig = inspect.signature(DeseqDataSet)
    if "metadata" in dds_sig.parameters:
        meta_kw = {"metadata": meta_df}
    elif "clinical" in dds_sig.parameters:
        meta_kw = {"clinical": meta_df}
    elif "clinical_data" in dds_sig.parameters:
        meta_kw = {"clinical_data": meta_df}
    elif "obs" in dds_sig.parameters:
        meta_kw = {"obs": meta_df}
    else:
        raise RuntimeError(
            f"Could not find metadata argument name in DeseqDataSet signature: {dds_sig}"
        )

    dds = DeseqDataSet(
        counts=counts_arm.astype(int),
        design_factors=["subject_id", "timepoint"],  # paired blocking
        **meta_kw
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, contrast=("timepoint", test_timepoint, ref_timepoint))
    stat_res.summary()

    res = stat_res.results_df.copy().reset_index().rename(columns={"index": "gene"})
    return res


# run it 

# meta, rna_raw must exist
meta_aso_rna, rna_aso = subset_arm(meta, rna_raw, "ASO")
meta_pl_rna,  rna_pl  = subset_arm(meta, rna_raw, "placebo")

res_rna_aso = deseq2_paired_within_arm(meta_aso_rna, rna_aso)
res_rna_pl  = deseq2_paired_within_arm(meta_pl_rna,  rna_pl)

res_rna_aso.to_csv("outputs/deseq2_aso_week4_vs_baseline.csv", index=False)
res_rna_pl.to_csv("outputs/deseq2_placebo_week4_vs_baseline.csv", index=False)
print("Saved RNA DE results.")

## proteomics DE
from scipy import stats
from statsmodels.stats.multitest import multipletests

def delta_matrix_within_arm(meta_arm: pd.DataFrame, mat: pd.DataFrame) -> pd.DataFrame:
    """
    meta_arm: sample_id, subject_id, timepoint (baseline/week4)
    mat: samples x proteins (log2 intensity)
    Returns: subject x proteins delta = week4 - baseline
    """
    df = meta_arm[["sample_id", "subject_id", "timepoint"]].set_index("sample_id").join(mat, how="inner")
    base = df[df["timepoint"] == BASELINE].groupby("subject_id").mean(numeric_only=True)
    wk4  = df[df["timepoint"] == WEEK4].groupby("subject_id").mean(numeric_only=True)

    common = base.index.intersection(wk4.index)
    delta = wk4.loc[common] - base.loc[common]
    delta.index.name = "subject_id"
    return delta

def paired_proteomics_de(meta_arm: pd.DataFrame, prot_log2_arm: pd.DataFrame, min_n: int = 6) -> pd.DataFrame:
    """
    Test if mean delta != 0 per protein (paired within-subject).
    """
    delta = delta_matrix_within_arm(meta_arm, prot_log2_arm)

    rows = []
    for prot in delta.columns:
        d = pd.to_numeric(delta[prot], errors="coerce").dropna()
        if len(d) < min_n:
            continue
        t = stats.ttest_1samp(d.values, popmean=0.0, nan_policy="omit")
        rows.append({
            "protein": prot,
            "n": int(len(d)),
            "mean_delta": float(d.mean()),
            "pvalue": float(t.pvalue),
        })

    res = pd.DataFrame(rows)
    res["padj"] = multipletests(res["pvalue"].values, method="fdr_bh")[1]
    res = res.sort_values(["padj", "pvalue"]).reset_index(drop=True)
    return res


# meta, prot_log2_f (or prot_log2) must exist
meta_aso_p, prot_aso = subset_arm(meta, prot_log2_f, "ASO")
meta_pl_p,  prot_pl  = subset_arm(meta, prot_log2_f, "placebo")

res_prot_aso = paired_proteomics_de(meta_aso_p, prot_aso)
res_prot_pl  = paired_proteomics_de(meta_pl_p,  prot_pl)

res_prot_aso.to_csv("outputs/proteomics_aso_week4_vs_baseline_paired.csv", index=False)
res_prot_pl.to_csv("outputs/proteomics_placebo_week4_vs_baseline_paired.csv", index=False)
print("Saved proteomics DE results.")

# filter genes specific in ASO and not sig in placebo
def aso_specific_from_two_tables(res_aso: pd.DataFrame, res_pl: pd.DataFrame,
                                id_col: str,
                                padj_sig: float = 0.05,
                                padj_nonsig: float = 0.10,
                                lfc_col: str = "log2FoldChange",
                                lfc_min_abs: float = 0.5) -> pd.DataFrame:
    """
    Generic ASO-specific filter comparing ASO vs placebo within-arm DE results.
    """
    a = res_aso[[id_col, "padj", lfc_col]].rename(columns={"padj": "padj_aso", lfc_col: "effect_aso"})
    p = res_pl[[id_col, "padj", lfc_col]].rename(columns={"padj": "padj_pl",  lfc_col: "effect_pl"})
    m = a.merge(p, on=id_col, how="left")

    m["aso_sig"] = (m["padj_aso"] < padj_sig) & (m["effect_aso"].abs() >= lfc_min_abs)
    m["pl_nonsig"] = (m["padj_pl"].isna()) | (m["padj_pl"] >= padj_nonsig)

    out = m[m["aso_sig"] & m["pl_nonsig"]].copy()
    return out.sort_values("padj_aso")

aso_spec_rna = aso_specific_from_two_tables(
    res_rna_aso, res_rna_pl,
    id_col="gene",
    padj_sig=0.05,
    padj_nonsig=0.10,
    lfc_col="log2FoldChange",
    lfc_min_abs=0.5
)

aso_spec_rna.to_csv("outputs/aso_specific_genes.csv", index=False)
print("ASO-specific genes:", aso_spec_rna.shape[0])

# prot
aso_spec_prot = aso_specific_from_two_tables(
    res_prot_aso.rename(columns={"mean_delta": "log2FoldChange"}),  # reuse function
    res_prot_pl.rename(columns={"mean_delta": "log2FoldChange"}),
    id_col="protein",
    padj_sig=0.05,
    padj_nonsig=0.10,
    lfc_col="log2FoldChange",
    lfc_min_abs=0.1  # proteomics tends to have smaller log2 deltas; adjust as needed
)

aso_spec_prot.to_csv("outputs/aso_specific_proteins.csv", index=False)
print("ASO-specific proteins:", aso_spec_prot.shape[0])

## Venn
import os
import pandas as pd

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

def split_up_down_sets(
    res_aso: pd.DataFrame,
    res_pl: pd.DataFrame,
    id_col: str,
    fdr_thr: float = 0.05,
    lfc_thr: float = 0.5,
    lfc_col: str = "log2FoldChange",
):
    """
    Returns 4 sets:
      ASO_up, ASO_down, PL_up, PL_down
    based on padj <= fdr_thr and log2FC thresholds.
    """

    # keep only needed columns, drop NAs
    a = res_aso[[id_col, "padj", lfc_col]].dropna().copy()
    p = res_pl[[id_col, "padj", lfc_col]].dropna().copy()

    ASO_up   = set(a.loc[(a["padj"] <= fdr_thr) & (a[lfc_col] >=  lfc_thr), id_col].astype(str))
    ASO_down = set(a.loc[(a["padj"] <= fdr_thr) & (a[lfc_col] <= -lfc_thr), id_col].astype(str))

    PL_up    = set(p.loc[(p["padj"] <= fdr_thr) & (p[lfc_col] >=  lfc_thr), id_col].astype(str))
    PL_down  = set(p.loc[(p["padj"] <= fdr_thr) & (p[lfc_col] <= -lfc_thr), id_col].astype(str))

    return ASO_up, ASO_down, PL_up, PL_down

#  Plot “one figure per omic” (two Venns in one figure)
import matplotlib.pyplot as plt
from matplotlib_venn import venn2

def plot_up_down_venns(
    ASO_up, ASO_down, PL_up, PL_down,
    title: str,
    outfile: str
):
    """
    One figure containing:
      left: UP overlap (ASO_up vs Placebo_up)
      right: DOWN overlap (ASO_down vs Placebo_down)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    venn2([ASO_up, PL_up], set_labels=("ASO up", "Placebo up"), ax=axes[0])
    axes[0].set_title("UP (FDR & log2FC threshold)")

    venn2([ASO_down, PL_down], set_labels=("ASO down", "Placebo down"), ax=axes[1])
    axes[1].set_title("DOWN (FDR & log2FC threshold)")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print("Saved:", outfile)

# run RNA
FDR_THR = 0.05
LFC_THR = 0.5

ASO_up, ASO_down, PL_up, PL_down = split_up_down_sets(
    res_rna_aso, res_rna_pl,
    id_col="gene",
    fdr_thr=FDR_THR,
    lfc_thr=LFC_THR,
    lfc_col="log2FoldChange"
)

# Save the lists (picky reviewers love this)
pd.Series(sorted(ASO_up)).to_csv(f"{OUT_DIR}/rna_ASO_up.csv", index=False, header=["gene"])
pd.Series(sorted(ASO_down)).to_csv(f"{OUT_DIR}/rna_ASO_down.csv", index=False, header=["gene"])
pd.Series(sorted(PL_up)).to_csv(f"{OUT_DIR}/rna_placebo_up.csv", index=False, header=["gene"])
pd.Series(sorted(PL_down)).to_csv(f"{OUT_DIR}/rna_placebo_down.csv", index=False, header=["gene"])

plot_up_down_venns(
    ASO_up, ASO_down, PL_up, PL_down,
    title=f"RNA DE overlaps (FDR<={FDR_THR}, |log2FC|>={LFC_THR})",
    outfile=f"{OUT_DIR}/venn_rna_up_down.png"
)

# proteomics
#If your proteomics DE result uses mean_delta rather than log2FoldChange, do:

res_prot_aso2 = res_prot_aso.rename(columns={"mean_delta": "log2FoldChange"})
res_prot_pl2  = res_prot_pl.rename(columns={"mean_delta": "log2FoldChange"})

FDR_THR_PROT = 0.05
LFC_THR_PROT = 0.25  # often smaller for proteins; adjust to match your rubric

ASO_up_p, ASO_down_p, PL_up_p, PL_down_p = split_up_down_sets(
    res_prot_aso2, res_prot_pl2,
    id_col="protein",
    fdr_thr=FDR_THR_PROT,
    lfc_thr=LFC_THR_PROT,
    lfc_col="log2FoldChange"
)

pd.Series(sorted(ASO_up_p)).to_csv(f"{OUT_DIR}/prot_ASO_up.csv", index=False, header=["protein"])
pd.Series(sorted(ASO_down_p)).to_csv(f"{OUT_DIR}/prot_ASO_down.csv", index=False, header=["protein"])
pd.Series(sorted(PL_up_p)).to_csv(f"{OUT_DIR}/prot_placebo_up.csv", index=False, header=["protein"])
pd.Series(sorted(PL_down_p)).to_csv(f"{OUT_DIR}/prot_placebo_down.csv", index=False, header=["protein"])

plot_up_down_venns(
    ASO_up_p, ASO_down_p, PL_up_p, PL_down_p,
    title=f"Proteomics DE overlaps (FDR<={FDR_THR_PROT}, |log2FC|>={LFC_THR_PROT})",
    outfile=f"{OUT_DIR}/venn_proteomics_up_down.png"
)



#######
print(meta.shape)
print(rna.shape)
print(prot.shape)

print(meta.head())
print(meta["treatment"].value_counts())
print(meta["timepoint"].value_counts())

pd.crosstab(meta["subject_id"], meta["timepoint"]).head()

rna_log = np.log2(rna_mat + 1)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(rna_log)
pcs = PCA(n_components=2).fit_transform(X)

## cheat sheet 
# Load
meta = pd.read_csv('data/metadata.csv')
prot = pd.read_csv('data/proteomics.csv')
rna  = pd.read_csv('data/transcriptomics.csv')

print("Meta columns:", meta.columns)
print("Prot columns:", prot.columns)
print("Prot index name:", prot.index.name)

# 1. Prepare Proteomics: Set protein ID as index, then Transpose (.T)
# This turns columns (S001_baseline...) into the Index (rows)
prot_clean = prot.set_index('protein').T

# 2. Reset index to turn the sample names back into a column we can merge on
prot_clean = prot_clean.reset_index().rename(columns={'index': 'sample_id'})

# 3. Do the same for RNA (assuming it has the same structure)
rna_clean = rna.set_index('gene').T # Change 'gene_id' to whatever the first col is named
rna_clean = rna_clean.reset_index().rename(columns={'index': 'sample_id'})

# 4. NOW you can merge
# Check if sample_ids match exactly (e.g., 'S001_baseline' vs 'S001_baseline')
df = meta.merge(prot_clean, on='sample_id', how='left')
df = df.merge(rna_clean, on='sample_id', how='left')

print("Merged shape:", df.shape)
print(df.head())

## delta
# 1. Melt to long format (if not already)
# This stacks all proteins into one column 'value', indexed by protein 'variable'
# (Skip this if you prefer working with wide columns directly, see below)

# BETTER METHOD for Interviews (Pivot Table approach):
# Create a dataframe where rows are subjects, columns are Timepoints
# We want to subtract: Week4 - Baseline

# Extract list of protein columns (e.g., all starting with 'PROT')
prot_cols = [c for c in df.columns if c.startswith('PROT')]

# Pivot: Index=Subject/Treatment, Columns=Timepoint, Values=Proteins
df_piv = df.pivot_table(index=['subject_id', 'treatment'], 
                        columns='timepoint', 
                        values=prot_cols)

# Calculate Delta: (Week4 - Baseline)
# Note: df_piv has MultiIndex columns (Protein, Timepoint)
df_delta = df_piv.xs('week4', level=1, axis=1) - df_piv.xs('baseline', level=1, axis=1)

# Reset index to get treatment back as a column for plotting
df_delta = df_delta.reset_index()

import seaborn as sns
import matplotlib.pyplot as plt

# Sanity Check: Did PROT001 go down in ASO?
sns.boxplot(data=df_delta, x='treatment', y='PROT001')
sns.stripplot(data=df_delta, x='treatment', y='PROT001', color='k', alpha=0.5)
plt.title("Target Engagement: PROT001 Change")
plt.show()

## boxplot with timeline
import seaborn as sns
import matplotlib.pyplot as plt

# Use the original merged dataframe 'df' (before calculating deltas)
# It should have columns: 'treatment', 'timepoint', and 'PROT001'

plt.figure(figsize=(8, 6))

# Main Boxplot
# x = Grouping (Treatment)
# y = Value (Protein Level)
# hue = Sub-grouping (Timepoint)
sns.boxplot(data=df, x='treatment', y='PROT001', hue='timepoint')

# Add dots (Strip plot)
# Important: set dodge=True so the dots align with the split boxplots
sns.stripplot(data=df, x='treatment', y='PROT001', hue='timepoint', 
              dodge=True, color='black', alpha=0.5, legend=False)

plt.title("PROT001 Levels: Treatment vs Timepoint")
plt.ylabel("Raw Protein Expression")
plt.show()

## GEN001
gen_cols = [c for c in df.columns if c.startswith('GEN')]

# Pivot: Index=Subject/Treatment, Columns=Timepoint, Values=Proteins
df_piv = df.pivot_table(index=['subject_id', 'treatment'], 
                        columns='timepoint', 
                        values=gen_cols)

# Calculate Delta: (Week4 - Baseline)
# Note: df_piv has MultiIndex columns (Protein, Timepoint)
df_delta = df_piv.xs('week4', level=1, axis=1) - df_piv.xs('baseline', level=1, axis=1)

# Reset index to get treatment back as a column for plotting
df_delta = df_delta.reset_index()

# Main Boxplot
# x = Grouping (Treatment)
# y = Value (Protein Level)
# hue = Sub-grouping (Timepoint)
sns.boxplot(data=df, x='treatment', y='PROT001', hue='timepoint')

# Add dots (Strip plot)
# Important: set dodge=True so the dots align with the split boxplots
sns.stripplot(data=df, x='treatment', y='GEN001', hue='timepoint', 
              dodge=True, color='black', alpha=0.5, legend=False)

plt.title("GEN001 Levels: Treatment vs Timepoint")
plt.ylabel("Raw RNA Expression")
plt.show()
## pca
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Drop non-numeric info for PCA
X = df[prot_cols] 
# OPTIONAL: X = np.log1p(X) if data looks raw (large range)

# Scale & Run
pca = PCA(n_components=2)
coords = pca.fit_transform(StandardScaler().fit_transform(X))

# Plot
plt.scatter(coords[:,0], coords[:,1], c=df['batch'].map({'Batch1':0, 'Batch2':1}))
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.show()

## correaltion
# Correlate all protein deltas against the Target Delta (PROT001)
target_vec = df_delta['PROT001']
corrs = df_delta.corrwith(target_vec) # Pandas magic function!

# Sort best positive correlations (move together)
print(corrs.sort_values(ascending=False).head(5))

# Sort best negative correlations (move opposite)
print(corrs.sort_values(ascending=True).head(5))
