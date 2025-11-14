import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 120
})

FNAME_RE = re.compile(r'(.+)-(.+)-(\d+)-(\d+)\.npz$')   # dataset-model-seed-lvl
OUT_CSV  = "metrics_by_ratio.csv"

def extract_metrics(path: str):
    m = FNAME_RE.match(os.path.basename(path))
    if not m:
        print(f"{path}: name doesn't match pattern skipped"); return

    dataset, model, seed, lvl = m.groups()
    with np.load(path, allow_pickle=True) as data:
        result = data["result"].tolist()      # list: [mname, model_type, AUC, PR, …]
        auc, pr = result[2], result[3]
        try:
            auc = float(auc)
            pr  = float(pr)
        except ValueError:
            return

        contam_ratio = float(data["contam_ratio"])
        yield dict(dataset=dataset,
                   model=model,
                   seed=int(seed),
                   contam_lvl=int(lvl),
                   contam_ratio=contam_ratio,
                   auc=auc,
                   pr=pr)

def load_all(npz_dir: str) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(npz_dir, "*.npz")):
        rows.extend(extract_metrics(path) or [])
    if not rows:
        raise SystemExit("No valid .npz files found!")
    return pd.DataFrame(rows)

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± std across *seeds* for every dataset X model X contamination-ratio.
    """
    df_summary = (df.groupby(["dataset", "model", "contam_ratio"])
                    .agg(auc_mean=("auc", "mean"),
                         auc_std =("auc", "std"),
                         pr_mean =("pr", "mean"),
                         pr_std  =("pr", "std"))
                    .reset_index()
                    .sort_values(["dataset", "model", "contam_ratio"]))

    df_summary[["auc_std", "pr_std"]] = df_summary[["auc_std", "pr_std"]].fillna(0.0)
    return df_summary

def plot_dataset(df_ds: pd.DataFrame, out_dir: str, metric: str, 
                 ylabel: str, 
                 marker_map: dict):
    fig, ax = plt.subplots()
    for model, g_m in df_ds.groupby("model"):
        g_m = g_m.sort_values("contam_ratio")
        ax.errorbar(g_m["contam_ratio"], g_m[f"{metric}_mean"],
                    yerr=g_m[f"{metric}_std"],
                    marker=marker_map[model], 
                    capsize=3, label=model)
    ax.set(title=f"{df_ds.name} - {ylabel} vs contamination",
           xlabel="Contamination ratio",
           ylabel=ylabel)
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{df_ds.name}_{metric}.png"))
    plt.close(fig)

def plot_all(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # 11 distinct, easily readable markers
    MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', '8', '>']
    def make_marker_map(models):
        """
        Return {model_name: marker} with unique marker per model, cycling if > len(MARKERS).
        """
        return {m: MARKERS[i % len(MARKERS)] for i, m in enumerate(sorted(models))}
    marker_map = make_marker_map(df["model"].unique())
    for ds, g_ds in df.groupby("dataset"):
        g_ds = g_ds.copy(); g_ds.name = ds
        plot_dataset(g_ds, out_dir, "auc", "AUC ROC", marker_map)
        plot_dataset(g_ds, out_dir, "pr",  "Average precision", marker_map)


def _select_best_model(df_ds: pd.DataFrame) -> str:
    """Pick the model with the highest average AUC across contamination ratios."""
    return (df_ds.groupby("model")["auc_mean"]
                .mean()
                .idxmax())


def plot_grid(df: pd.DataFrame, out_dir: str, filename: str = "contamination_grid.pdf"):
    """
    Plot the eight dataset curves in a single 2x4 grid, overlaying AUC/PR in
    blue/red respectively. Uses the best-performing model per dataset and
    shades ±1 std to show variance across seeds.
    """
    datasets = sorted(df["dataset"].unique())
    if not datasets:
        return

    rows, cols = 2, 4
    if len(datasets) > rows * cols:
        raise ValueError("plot_grid currently supports at most 8 datasets in a 2x4 layout.")
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5), sharey=True)
    axes = axes.flatten()

    auc_color = "tab:blue"
    pr_color = "tab:red"

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        df_ds = df[df["dataset"] == ds]
        best_model = _select_best_model(df_ds)
        df_model = (df_ds[df_ds["model"] == best_model]
                        .sort_values("contam_ratio")
                        .copy())
        df_model["auc_std"].fillna(0, inplace=True)
        df_model["pr_std"].fillna(0, inplace=True)

        x = df_model["contam_ratio"].values

        # AUROC curve
        y_auc = df_model["auc_mean"].values
        y_auc_std = df_model["auc_std"].values
        ax.plot(x, y_auc, color=auc_color, marker='o', label="AUROC")
        ax.fill_between(x, y_auc - y_auc_std, y_auc + y_auc_std, color=auc_color, alpha=0.2)

        # AUPRC curve
        y_pr = df_model["pr_mean"].values
        y_pr_std = df_model["pr_std"].values
        ax.plot(x, y_pr, color=pr_color, marker='s', label="AUPRC")
        ax.fill_between(x, y_pr - y_pr_std, y_pr + y_pr_std, color=pr_color, alpha=0.2)

        ax.set_title(ds)
        ax.set_ylim(0, 1.05)
        ax.grid(True, linestyle='--', alpha=0.4)

        if ax_idx // cols == rows - 1:
            ax.set_xlabel("Abnormal Ratio in Training Set")
        if ax_idx % cols == 0:
            ax.set_ylabel("Score")

    for ax in axes[len(datasets):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False)
    fig.tight_layout(rect=[0, 0, 0.97, 0.97])

    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"✓ Grid figure saved → {out_path}")

if __name__ == "__main__":
    results_dir = "results_contam"
    output_dir = results_dir
    df_raw     = load_all(results_dir)
    df_summary = aggregate(df_raw)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, OUT_CSV)
    df_summary.to_csv(csv_path, index=False)
    print(f"CSV written to {csv_path}")

    plot_all(df_summary, output_dir)
    print(f"Plots saved in {output_dir}")

    plot_grid(df_summary, output_dir)
