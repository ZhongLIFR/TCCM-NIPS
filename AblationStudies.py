import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import load_adbench_npz
from FMAD.functions import SinusoidalTimeEmbedding, FlowMatching
import os

# Train Flow Matching
def train_flow_matching(model, train_loader, epochs=50, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            t = torch.rand(batch_x.shape[0], 1, device=batch_x.device)
            f_xt = model(batch_x, t)
            dx_dt = -batch_x
            
            loss = criterion(f_xt, dx_dt)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    return model

def train_flow_matching_noise(model, train_loader, epochs=50, lr=0.001, noise=False):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=1.0)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()

            # sample a random time t
            t = torch.rand(batch_x.shape[0], 1, device=batch_x.device) ** 2 

            if noise:
                epsilon = torch.randn_like(batch_x)
                interpolated_x = batch_x + t * epsilon
                f_xt = model(interpolated_x, t)
                dx_dt = -batch_x  # dx/dt
            else:
                f_xt = model(batch_x, t)  
                dx_dt = -batch_x  

            loss = criterion(f_xt, dx_dt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # avoid gradient explosion
            optimizer.step()
            total_loss += loss.item()
    return model


# ==========================
# Compute anomaly scores
# ==========================
def compute_anomaly_scores(model, X_test, y_test):
    if isinstance(X_test, np.ndarray):  
        X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test = X_test.to(next(model.parameters()).device)

    with torch.no_grad():
        t = torch.ones(X_test.shape[0], 1, device=X_test.device, dtype=torch.float32)
        f_xt = model(X_test, t)

        # Compute anomaly score S(x) = ||f(x, 1) + x||_2
        anomaly_scores = torch.norm(f_xt + X_test, dim=1)

    anomaly_scores = anomaly_scores.cpu().numpy()

    return anomaly_scores


def evaluate_anomaly_detection(y_test, anomaly_scores, method_name="TCCM"):
    """
    Evaluate the performance (AUROC and AUPRC)
    """
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.detach().cpu().numpy().astype(int)
    elif isinstance(y_test, np.ndarray):
        y_test = y_test.astype(int)

    if isinstance(anomaly_scores, torch.Tensor):
        anomaly_scores = anomaly_scores.detach().cpu().numpy()
    elif not isinstance(anomaly_scores, np.ndarray):
        anomaly_scores = np.array(anomaly_scores)

    # Calculate AUROC and AUPRC
    auc_score = roc_auc_score(y_test, anomaly_scores)
    pr_score = average_precision_score(y_test, anomaly_scores)

    print(f"📊 {method_name} - ROC-AUC Score: {auc_score:.4f}")
    print(f"📊 {method_name} - Precision-Recall AUC Score: {pr_score:.4f}")

    return auc_score, pr_score

# ==========================
# Global Configs for All Ablation Studies
# ==========================

def get_data_configs():
    return {
        # Small
        "29_Pima": {"batch_size": 512, "lr": 0.005, "epochs": 5},
        "18_Ionosphere": {"batch_size": 512, "lr": 0.005, "epochs": 10},
        # Medium
        "31_satimage-2": {"batch_size": 512, "lr": 0.005, "epochs": 5},
        "44_Wilt": {"batch_size": 512, "lr": 0.005, "epochs": 20},
        # Large
        "22_magic.gamma": {"batch_size": 1024, "lr": 0.005, "epochs": 10},
        "23_mammography": {"batch_size": 1024, "lr": 0.005, "epochs": 20},
        # High-dimensional
        "25_musk": {"batch_size": 512, "lr": 0.005, "epochs": 5},
        "17_InternetAds": {"batch_size": 512, "lr": 0.005, "epochs": 50},
    }

t_grid = np.linspace(0.0, 1.0, 11)
seed_list = [0, 1, 2, 3, 4]


# ==========================
# Helper: Compute Anomaly Score at Given t
# ==========================
def compute_score_at_t(model, X_test, t_val):
    t_tensor = torch.full((X_test.shape[0], 1), t_val, device=X_test.device)
    with torch.no_grad():
        f_xt = model(X_test, t_tensor)
        return torch.norm(f_xt + X_test, dim=1).cpu().numpy()


# ==========================
# Ablation Study 1: Sensitivity to Fixed t
# ==========================
def run_fixed_t_sensitivity_analysis():
    import matplotlib.pyplot as plt

    data_configs = get_data_configs()
    results = {}

    for dataset_name, config in data_configs.items():
        aucs, prs = [], []

        for t_val in t_grid:
            auc_scores, pr_scores = [], []

            for seed in seed_list:
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Load dataset
                X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed)
                X_train = torch.tensor(X_train, dtype=torch.float32)
                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

                # Model and training
                model = FlowMatching(input_dim=X_train.shape[1])
                train_loader = DataLoader(
                    TensorDataset(X_train, torch.zeros(len(X_train))),
                    batch_size=config["batch_size"], shuffle=True,
                )
                model = train_flow_matching(model, train_loader, epochs=config["epochs"], lr=config["lr"])

                # Evaluation
                scores = compute_score_at_t(model, X_test, t_val)
                auc, pr = evaluate_anomaly_detection(y_test, scores, method_name=f"{dataset_name}_t={t_val:.1f}")
                auc_scores.append(auc)
                pr_scores.append(pr)

            aucs.append((np.mean(auc_scores), np.std(auc_scores)))
            prs.append((np.mean(pr_scores), np.std(pr_scores)))

        results[dataset_name] = {"auroc": aucs, "auprc": prs}

    # ================
    # Plotting: All 8 datasets in 2x4 layout
    # ================
    fig, axs = plt.subplots(2, 4, figsize=(28, 10), sharey=True)
    for i, dataset_name in enumerate(results.keys()):
        row, col = divmod(i, 4)
        ax = axs[row][col]

        auroc = np.array(results[dataset_name]["auroc"])
        auprc = np.array(results[dataset_name]["auprc"])

        ax.plot(t_grid, auroc[:, 0], label="AUROC", color="blue", marker='d')
        ax.fill_between(t_grid, auroc[:, 0] - auroc[:, 1], auroc[:, 0] + auroc[:, 1], color="blue", alpha=0.2)

        ax.plot(t_grid, auprc[:, 0], label="AUPRC", color="red", marker='o')
        ax.fill_between(t_grid, auprc[:, 0] - auprc[:, 1], auprc[:, 0] + auprc[:, 1], color="red", alpha=0.2)

        ax.set_title(dataset_name)
        ax.set_xlabel("Fixed Time Value $t$ at Inference")
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        if col == 0:
            ax.set_ylabel("Anomaly Score")
        if row == 0 and col == 3:
            ax.legend(loc="lower right")

    plt.tight_layout()
    os.makedirs("./results_ablation/", exist_ok=True)
    plt.savefig("./results_ablation/Sensitivity_t_Figure_13.pdf", bbox_inches='tight')
    plt.show()

# ==========================
# Ablation Study 2: Time Embedding Methods (t=1.0)
# ==========================
# Univeral setups
data_configs = get_data_configs()

# Generic TCCM class
class TCCMAblation(nn.Module):
    def __init__(self, input_dim, time_embed_dim=128, embedding_type="Sinusoidal"):
        super().__init__()
        self.embedding_type = embedding_type

        if embedding_type == "LinearSin":
            self.time_embedding = nn.Linear(1, time_embed_dim)
            self.encode_time = lambda t: torch.sin(self.time_embedding(t))
        elif embedding_type == "Sinusoidal":
            self.encode_time = SinusoidalTimeEmbedding(time_embed_dim)
        elif embedding_type == "SinMLP":
            self.encode_time = nn.Sequential(
                SinusoidalTimeEmbedding(time_embed_dim),
                nn.Linear(time_embed_dim, time_embed_dim),
                nn.ReLU(),
                nn.Linear(time_embed_dim, time_embed_dim)
            )
        else:
            raise ValueError(f"Unsupported embedding_type: {embedding_type}")

        self.model = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, t):
        t_emb = self.encode_time(t.view(-1, 1))
        x_t = torch.cat([x, t_emb], dim=1)
        return self.model(x_t)

# Main function of the ablation study of varies time embedding methods
def run_time_embedding_ablation():
    embedding_types = ["LinearSin", "Sinusoidal", "SinMLP"]
    seed_list = [0, 1, 2, 3, 4]
    results_dict = {}

    for dataset_name, config in data_configs.items():
        print(f"\n Dataset: {dataset_name}")
        method_scores = {}

        for emb_type in embedding_types:
            print(f"Embedding: {emb_type}")
            aucs, prcs = [], []

            for seed in seed_list:
                torch.manual_seed(seed)
                np.random.seed(seed)

                X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed)
                X_train = torch.tensor(X_train, dtype=torch.float32)
                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

                model = TCCMAblation(input_dim=X_train.shape[1], embedding_type=emb_type)
                train_loader = DataLoader(
                    TensorDataset(X_train, torch.zeros(len(X_train))),
                    batch_size=config["batch_size"],
                    shuffle=True,
                )
                model = train_flow_matching(model, train_loader, epochs=config["epochs"], lr=config["lr"])

                t_tensor = torch.ones(X_test.shape[0], 1, device=X_test.device)
                with torch.no_grad():
                    f_xt = model(X_test.to(t_tensor.device), t_tensor)
                    anomaly_scores = torch.norm(f_xt + X_test.to(t_tensor.device), dim=1).cpu().numpy()

                auc, pr = evaluate_anomaly_detection(y_test, anomaly_scores, method_name=emb_type)
                aucs.append(auc)
                prcs.append(pr)

            method_scores[emb_type] = (np.mean(aucs), np.std(aucs), np.mean(prcs), np.std(prcs))

        results_dict[dataset_name] = method_scores

    return results_dict

# Visualization: plot AUROC / AUPRC with bar plots
def plot_time_embedding_ablation(results_dict):
    """
    Create plots for AUROC and AUPRC:
    - Left: AUROC under different time embedding methods
    - Right: AUPRC under different time embedding methods
    """
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ["LinearSin", "Sinusoidal", "SinMLP"]
    method_labels = ["Linear+Sin", "Sinusoidal (default)", "Sinusoidal+MLP"]
    colors = ["#4E79A7", "#59A14F", "#E15759"]  # ColorBrewer-inspired

    datasets = list(results_dict.keys())
    x = np.arange(len(datasets))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

    bars = []
    for i, method in enumerate(methods):
        auroc_means = [results_dict[ds][method][0] for ds in datasets]
        auroc_stds = [results_dict[ds][method][1] for ds in datasets]
        auprc_means = [results_dict[ds][method][2] for ds in datasets]
        auprc_stds = [results_dict[ds][method][3] for ds in datasets]

        bar1 = ax1.bar(x + i * width, auroc_means, width, yerr=auroc_stds,
                       color=colors[i], capsize=5)
        bar2 = ax2.bar(x + i * width, auprc_means, width, yerr=auprc_stds,
                       color=colors[i], capsize=5)
        bars.append(bar1)

    for ax, metric in zip([ax1, ax2], ["AUROC", "AUPRC"]):
        ax.set_ylabel(f"{metric} Score")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets, rotation=30, ha="right")
        ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    ax1.set_title("Time Embedding Ablation - AUROC")
    ax2.set_title("Time Embedding Ablation - AUPRC")
    ax1.set_xlabel("Dataset")
    ax2.set_xlabel("Dataset")

    fig.legend(bars, method_labels, loc="upper center", ncol=len(methods), frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs("./results_ablation/", exist_ok=True)
    plt.savefig("./results_ablation/Time_Embedding_Figure_12.pdf")
    plt.show()

# ==========================
# Ablation Study 3: Training with or without Noise
# ==========================

# Global Setups
data_configs = get_data_configs()
seed_list = [0, 1, 2, 3, 4]

def run_noise_ablation(data_configs, seed_list):
    noise_results = {}

    for dataset_name, config in data_configs.items():
        print(f"\n Dataset: {dataset_name}")
        result = {}

        for noise_flag in [False, True]:
            mode = "NoNoise" if not noise_flag else "WithNoise"
            print(f"  Mode: {mode}")
            aucs, prs = [], []

            for seed in seed_list:
                torch.manual_seed(seed)
                np.random.seed(seed)

                X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed)
                X_train = torch.tensor(X_train, dtype=torch.float32)
                X_test = torch.tensor(X_test, dtype=torch.float32)
                y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

                model = FlowMatching(input_dim=X_train.shape[1])  # The default Sinusoidal
                train_loader = DataLoader(
                    TensorDataset(X_train, torch.zeros(len(X_train))),
                    batch_size=config["batch_size"],
                    shuffle=True,
                )
                model = train_flow_matching_noise(
                    model, train_loader, epochs=config["epochs"],
                    lr=config["lr"], noise=noise_flag
                )

                with torch.no_grad():
                    t = torch.ones(X_test.shape[0], 1, device=X_test.device)
                    f_xt = model(X_test.to(t.device), t)
                    anomaly_scores = torch.norm(f_xt + X_test.to(t.device), dim=1).cpu().numpy()

                auc, pr = evaluate_anomaly_detection(y_test, anomaly_scores, method_name=mode)
                aucs.append(auc)
                prs.append(pr)

            result[mode] = (np.mean(aucs), np.std(aucs), np.mean(prs), np.std(prs))

        noise_results[dataset_name] = result

    return noise_results


def plot_noise_ablation(noise_results):
    """
    Visualization: AUROC and AUPRC of adding noises vs no noises 
    """
    modes = ["NoNoise", "WithNoise"]
    mode_labels = ["No Noise (default)", "With Noise"]
    colors = ["#4C72B0", "#C44E52"]

    datasets = list(noise_results.keys())
    x = np.arange(len(datasets))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6), sharex=True)

    for i, mode in enumerate(modes):
        auroc_means = [noise_results[ds][mode][0] for ds in datasets]
        auroc_stds = [noise_results[ds][mode][1] for ds in datasets]
        auprc_means = [noise_results[ds][mode][2] for ds in datasets]
        auprc_stds = [noise_results[ds][mode][3] for ds in datasets]

        ax1.bar(x + i * width, auroc_means, width, yerr=auroc_stds, color=colors[i],
                capsize=5, label=mode_labels[i])
        ax2.bar(x + i * width, auprc_means, width, yerr=auprc_stds, color=colors[i],
                capsize=5, label=mode_labels[i])

    for ax, metric in zip([ax1, ax2], ["AUROC", "AUPRC"]):
        ax.set_ylabel(f"{metric} Score")
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(datasets, rotation=30, ha="right")

    ax1.set_title("Effect of Noise on AUROC")
    ax2.set_title("Effect of Noise on AUPRC")
    ax1.set_xlabel("Dataset")
    ax2.set_xlabel("Dataset")

    fig.legend(loc="upper center", ncol=2, fontsize=12, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs("./results_ablation/", exist_ok=True)
    plt.savefig("./results_ablation/Noise_Injection_Figure_14.pdf")
    plt.show()

# ==========================

# ==========================
# Ablation Study 4: Contamination Ratio Sensitivity (Per-Dataset Max) for TCCM
# More extensive comtamination study is provided at ContaminationStudies.py
# ==========================

# Dataset configuration
seed_list = [0, 1, 2, 3, 4]

def run_training_contamination_ablation_dynamic_fixed_split():
    data_configs = get_data_configs()
    all_results = {}
    all_contam_levels = {}

    for dataset_name, config in data_configs.items():
        print(f"\n Dataset: {dataset_name}")
        auroc_all = []
        auprc_all = []

        # ============================
        # Split normal / anomaly using fixed random seed 42
        # ============================
        X_train_full, y_train_full, X_test_full, y_test_full = load_adbench_npz(dataset_name, test_size=0.5, random_state=42)
        X_all = np.vstack([X_train_full, X_test_full])
        y_all = np.concatenate([y_train_full, y_test_full])

        X_normal = X_all[y_all == 0]
        X_abnormal = X_all[y_all == 1]

        from sklearn.model_selection import train_test_split
        
        X_train_normal, X_test_normal = train_test_split(X_normal, test_size=0.5, random_state=42, stratify=None)
        X_train_abnormal_full, X_test_abnormal = train_test_split(X_abnormal, test_size=0.5, random_state=42, stratify=None)

        X_test = np.vstack([X_test_normal, X_test_abnormal])
        y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_abnormal))])
        y_test = torch.tensor(y_test, dtype=torch.long)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        n_train_normal = len(X_train_normal)
        n_train_abnormal_max = len(X_train_abnormal_full)
        max_abnormal_ratio = n_train_abnormal_max / (n_train_abnormal_max + n_train_normal)

        contamination_levels = np.linspace(0.001, max_abnormal_ratio, 10)
        all_contam_levels[dataset_name] = contamination_levels

        for contam_ratio in contamination_levels:
            aucs, prs = [], []

            for seed in seed_list:
                torch.manual_seed(seed)
                np.random.seed(seed)

                n_ab = int(contam_ratio * n_train_normal / (1 - contam_ratio))
                n_ab = min(n_ab, n_train_abnormal_max)

                selected_ab = X_train_abnormal_full[:n_ab]  # Can be replaced by random sampling if needed
                X_train = np.vstack([X_train_normal, selected_ab])
                y_train = np.zeros(len(X_train))  # label unused

                X_train = torch.tensor(X_train, dtype=torch.float32)

                model = FlowMatching(input_dim=X_train.shape[1])
                train_loader = DataLoader(
                    TensorDataset(X_train, torch.tensor(y_train)),
                    batch_size=config["batch_size"],
                    shuffle=True,
                )
                model = train_flow_matching(model, train_loader, epochs=config["epochs"], lr=config["lr"])

                with torch.no_grad():
                    t = torch.ones(X_test.shape[0], 1, device=X_test.device)
                    f_xt = model(X_test.to(t.device), t)
                    scores = torch.norm(f_xt + X_test.to(t.device), dim=1).cpu().numpy()

                auc = roc_auc_score(y_test, scores)
                pr = average_precision_score(y_test, scores)
                aucs.append(auc)
                prs.append(pr)

            auroc_all.append((np.mean(aucs), np.std(aucs)))
            auprc_all.append((np.mean(prs), np.std(prs)))

        all_results[dataset_name] = {
            "auroc": auroc_all,
            "auprc": auprc_all,
        }

    return all_results, all_contam_levels


# Create the line plots
def plot_training_contamination_ablation_dynamic(results, contamination_levels_dict):
    fig, axs = plt.subplots(2, 4, figsize=(28, 10), sharey=True)
    dataset_list = list(results.keys())

    for i, dataset_name in enumerate(dataset_list):
        row, col = divmod(i, 4)
        ax = axs[row][col]

        auroc = np.array(results[dataset_name]["auroc"])
        auprc = np.array(results[dataset_name]["auprc"])
        contam_levels = contamination_levels_dict[dataset_name]

        ax.plot(contam_levels, auroc[:, 0], label="AUROC", color="blue", marker='o')
        ax.fill_between(contam_levels,
                        auroc[:, 0] - auroc[:, 1],
                        auroc[:, 0] + auroc[:, 1],
                        color="blue", alpha=0.2)

        ax.plot(contam_levels, auprc[:, 0], label="AUPRC", color="red", marker='s')
        ax.fill_between(contam_levels,
                        auprc[:, 0] - auprc[:, 1],
                        auprc[:, 0] + auprc[:, 1],
                        color="red", alpha=0.2)

        ax.set_title(dataset_name)
        ax.set_xlabel("Abnormal Ratio in Training Set")
        ax.set_ylim(0, 1.05)
        ax.grid(True)

        if col == 0:
            ax.set_ylabel("Accuracy Score")

    axs[0][3].legend(loc="upper center", bbox_to_anchor=(1.15, 1.1), fontsize="large")
    plt.tight_layout()
    os.makedirs("./results_ablation/", exist_ok=True)
    plt.savefig("./results_ablation/Contamination_Figure_TCCM.pdf")
    plt.show()


if __name__=="__main__":
    # ======================================================== #
    """
    Ablation Study 1: Sensitivity to Fixed t
    """
    run_fixed_t_sensitivity_analysis()
    # ======================================================== #
    # ======================================================== #
    """
    Ablation Study 2: Time Embedding Methods (t=1.0)
    """
    results = run_time_embedding_ablation()
    plot_time_embedding_ablation(results)
    # ======================================================== #
    # ======================================================== #
    """
    Ablation Study 3: Training with or without Noise
    """
    noise_results = run_noise_ablation(data_configs, seed_list)
    plot_noise_ablation(noise_results)
    # ======================================================== #
    # ======================================================== #
    """
    Ablation Study 4: Contamination Ratio Sensitivity (Per-Dataset Max)
    """
    results, contamination_levels_dict= run_training_contamination_ablation_dynamic_fixed_split()
    plot_training_contamination_ablation_dynamic(results, contamination_levels_dict)
    # ======================================================== #




