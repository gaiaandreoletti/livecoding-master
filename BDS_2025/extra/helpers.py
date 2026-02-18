# (1) Load data and basic inspection
import pandas as pd

meta = pd.read_csv("data/meta.csv")
prot = pd.read_csv("data/proteome.csv")
rna  = pd.read_csv("data/transcriptome.csv")

print("meta:", meta.shape)
print("proteome:", prot.shape)
print("transcriptome:", rna.shape)

meta.head()



# (2) Inspect study design (treatment / timepoint)
meta.groupby(["treatment", "timepoint"]).agg(
    n_samples=("sample_id", "nunique"),
    n_subjects=("subject_id", "nunique")
)



# (3) Join metadata and proteomics
prot_full = meta.merge(prot, on="sample_id", how="inner")
prot_full.head()



# (4) Compute within-subject baseline → week4 deltas
baseline = prot_full[prot_full["timepoint"] == "baseline"].copy()
week4    = prot_full[prot_full["timepoint"] == "week4"].copy()

baseline = baseline.sort_values("subject_id")
week4    = week4.sort_values("subject_id")

prot_cols = [c for c in prot_full.columns if c.startswith("PROT")]

import numpy as np

delta_vals = week4[prot_cols].values - baseline[prot_cols].values
delta = pd.DataFrame(delta_vals, columns=prot_cols)
delta["subject_id"] = baseline["subject_id"].values
delta["treatment"]  = baseline["treatment"].values

delta.head()



# (5) Check the target protein (PROT001)
from scipy import stats

aso = delta[delta["treatment"] == "ASO"]["PROT001"].dropna()
plac = delta[delta["treatment"] == "placebo"]["PROT001"].dropna()

print("Mean delta PROT001 (ASO):    ", aso.mean())
print("Mean delta PROT001 (placebo):", plac.mean())

stats.ttest_ind(aso, plac, equal_var=False)



# (6) Scan all proteins for ASO vs placebo differences
results = []
for p in prot_cols:
    aso_vals  = delta.loc[delta["treatment"]=="ASO", p].dropna()
    plac_vals = delta.loc[delta["treatment"]=="placebo", p].dropna()
    if len(aso_vals) < 2 or len(plac_vals) < 2:
        continue
    t, pval = stats.ttest_ind(aso_vals, plac_vals, equal_var=False)
    results.append({
        "protein": p,
        "mean_delta_ASO": aso_vals.mean(),
        "mean_delta_placebo": plac_vals.mean(),
        "difference_ASO_minus_placebo": aso_vals.mean() - plac_vals.mean(),
        "pvalue": pval
    })

prot_hits = pd.DataFrame(results).sort_values("pvalue").reset_index(drop=True)
prot_hits.head(10)



# (7) Volcano-style plot for proteins
import matplotlib.pyplot as plt
import seaborn as sns

df = prot_hits.copy()
df["neg_log10_p"] = -np.log10(df["pvalue"].replace(0, np.nan))

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=df,
    x="difference_ASO_minus_placebo",
    y="neg_log10_p",
    s=30
)

# Highlight target protein
if "PROT001" in df["protein"].values:
    target_row = df[df["protein"] == "PROT001"].iloc[0]
    plt.scatter(
        target_row["difference_ASO_minus_placebo"],
        target_row["neg_log10_p"],
        s=80,
        edgecolor="red",
        facecolor="none"
    )
    plt.text(
        target_row["difference_ASO_minus_placebo"],
        target_row["neg_log10_p"],
        " PROT001",
        va="center"
    )

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("Δ (ASO - placebo)")
plt.ylabel("-log10 p-value")
plt.title("CSF protein deltas: ASO vs placebo")
plt.tight_layout()
plt.show()



# (8) Filter to candidate biomarker proteins
hits_filtered = prot_hits[
    (prot_hits["pvalue"] < 0.05) &
    (prot_hits["difference_ASO_minus_placebo"] < 0)
].sort_values("pvalue")

hits_filtered.head(10)



# (9) (Optional) Similar approach for RNA
rna_full = meta.merge(rna, on="sample_id", how="inner")

baseline_rna = rna_full[rna_full["timepoint"] == "baseline"].copy()
week4_rna    = rna_full[rna_full["timepoint"] == "week4"].copy()

baseline_rna = baseline_rna.sort_values("subject_id")
week4_rna    = week4_rna.sort_values("subject_id")

rna_cols = [c for c in rna_full.columns if c.startswith("GENE")]

delta_vals_rna = week4_rna[rna_cols].values - baseline_rna[rna_cols].values
delta_rna = pd.DataFrame(delta_vals_rna, columns=rna_cols)
delta_rna["subject_id"] = baseline_rna["subject_id"].values
delta_rna["treatment"]  = baseline_rna["treatment"].values

delta_rna.head()
