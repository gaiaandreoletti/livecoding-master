import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_subjects = 20
treated = 12
placebo = n_subjects - treated
timepoints = ["baseline", "week4"]
n_genes = 500
n_proteins = 300

genes = [f"GENE{i:03d}" for i in range(1, n_genes+1)]
proteins = [f"PROT{i:03d}" for i in range(1, n_proteins+1)]

geneX_name = "GENE001"
protX_name = "PROT001"

# small panel of downstream features (still affected, but less important for the exercise)
affected_genes = [geneX_name] + [f"GENE{i:03d}" for i in range(2, 2+14)]
affected_prots = [protX_name] + [f"PROT{i:03d}" for i in range(2, 2+11)]

# ----------------------
# Metadata
# ----------------------
subject_ids = [f"SUBJ{i:03d}" for i in range(1, n_subjects+1)]
assign = ["ASO"]*treated + ["placebo"]*placebo
np.random.shuffle(assign)
assign_map = dict(zip(subject_ids, assign))

rows = []
for sid in subject_ids:
    for tp in timepoints:
        rows.append({
            "sample_id": f"{sid}_{tp}",
            "subject_id": sid,
            "treatment": assign_map[sid],
            "timepoint": tp
        })
meta = pd.DataFrame(rows)

meta["batch"] = np.where(
    (meta["treatment"] == "ASO") & (np.random.rand(len(meta)) < 0.65),
    "B", "A"
)
meta["site"] = np.random.choice([1,2,3], size=len(meta), p=[0.4,0.4,0.2])
meta["age"] = np.clip(np.round(np.random.normal(52,10,size=len(meta)),1), 30, 80)
meta["sex"] = np.random.choice(["F","M"], size=len(meta))
meta["dose"] = np.where(
    meta["treatment"]=="ASO",
    np.random.choice([50,100], size=len(meta), p=[0.6,0.4]),
    0
)
meta = meta.sort_values(["subject_id","timepoint"]).reset_index(drop=True)

# ----------------------
# Omics matrices
# ----------------------
n_samples = len(meta)
rna = pd.DataFrame(np.random.normal(0, 1, size=(n_samples, n_genes)), columns=genes)
prot = pd.DataFrame(np.random.normal(0, 1, size=(n_samples, n_proteins)), columns=proteins)

# Batch effects on random subsets, EXCLUDING the main target
batch_genes = np.random.choice([g for g in genes if g != geneX_name], size=50, replace=False)
batch_prots = np.random.choice([p for p in proteins if p != protX_name], size=30, replace=False)

rna.loc[meta["batch"]=="B", batch_genes] += 0.4
prot.loc[meta["batch"]=="B", batch_prots] += 0.4

# Subject-level random effects (paired structure)
subject_effect = {sid: np.random.normal(0, 0.5) for sid in subject_ids}
for i, row in meta.iterrows():
    sid = row["subject_id"]
    rna.iloc[i,:]  += subject_effect[sid]
    prot.iloc[i,:] += subject_effect[sid]*0.8

aso_w4  = (meta["treatment"]=="ASO") & (meta["timepoint"]=="week4")
plac_w4 = (meta["treatment"]=="placebo") & (meta["timepoint"]=="week4")

# ----------------------
# Treatment effects
# ----------------------

# Strong, clean target engagement on PROT001 and GENE001 for ASO at week4
# (choose something big relative to noise; -2 to -3 is fine)
rna.loc[aso_w4, geneX_name]  += -2.5
prot.loc[aso_w4, protX_name] += -2.0

# No systematic placebo drift on target itself (only baseline noise)
# (we leave rna/prot for placebo as originally drawn)

# Moderate effects on downstream panel
rna.loc[aso_w4,  affected_genes[1:]] += -0.7
prot.loc[aso_w4, affected_prots[1:]] += -0.6

# Mild random drift at week4 for everyone (smaller than before)
rna.loc[meta["timepoint"]=="week4", :]  += np.random.normal(
    0, 0.03, size=((meta["timepoint"]=="week4").sum(), n_genes)
)
prot.loc[meta["timepoint"]=="week4", :] += np.random.normal(
    0, 0.03, size=((meta["timepoint"]=="week4").sum(), n_proteins)
)

# ~3% missing in proteomics
mask = np.random.rand(*prot.shape) < 0.03
prot = prot.mask(mask)

# Attach sample_id
rna.insert(0, "sample_id", meta["sample_id"].values)
prot.insert(0, "sample_id", meta["sample_id"].values)

# ----------------------
# Save
# ----------------------
out_dir = "data"
os.makedirs(out_dir, exist_ok=True)
meta.to_csv(os.path.join(out_dir, "meta.csv"), index=False)
rna.to_csv(os.path.join(out_dir, "transcriptome.csv"), index=False)
prot.to_csv(os.path.join(out_dir, "proteome.csv"), index=False)

with open(os.path.join(out_dir, "README_data.txt"), "w") as f:
    f.write("Synthetic CSF ASO dataset for live coding (target effect tuned for clear PROT001 signal).\n")
