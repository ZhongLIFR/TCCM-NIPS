import os
# disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
# **Suppress all warnings**
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
import re

from contextlib import redirect_stdout
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from utils import set_seed

# **Deep Learning Models (Require Training)**
deep_models = {
    "TCCMRobust": "FMAD.FlowMatchingAD",
}
MODEL_NAMES = list(deep_models.keys())

# The range of accepted perturbation level testing robustness
EPS_RANGES = list(np.arange(0.1, 3.1, 0.1, dtype=float)) # The data is standardized (std=1), so this is equal to percentage of perturbations
EPS_MINI_STEP = 0.01

def sample_gmm_data(normal_modes_values=[-1, 0, 1], anomaly_modes_values=[-1, 1], d=2, n_train=5000, n_test_normal=4000, n_test_anom=1000, 
                    mu_sep=3, anom_sep=9, sigma=1.0, seed=42):
    rng = np.random.default_rng(seed+d)
    
    # Means for normal components
    normal_modes = len(normal_modes_values)
    mu_r = [np.ones(d) * mu_sep * (i) for i in normal_modes_values]
    pi_r = np.ones(normal_modes) / normal_modes

    # Means for anomalous components, disjoint from normal as described in Proposition 2
    anomaly_modes = len(anomaly_modes_values)
    nu_s = [np.ones(d) * anom_sep * (i) for i in anomaly_modes_values]
    eta_s = np.ones(anomaly_modes) / anomaly_modes

    # Covariance matrix ()
    cov = (sigma**2) * np.eye(d)

    # Sample training data (normal only)
    X_train = []
    for _ in range(n_train):
        r = rng.choice(normal_modes, p=pi_r)
        x = rng.multivariate_normal(mu_r[r], cov)
        X_train.append(x)
    X_train = np.array(X_train)

    # Sample normal test data
    X_test_normal = []
    for _ in range(n_test_normal):
        r = rng.choice(normal_modes, p=pi_r)
        x = rng.multivariate_normal(mu_r[r], cov)
        X_test_normal.append(x)
    X_test_normal = np.array(X_test_normal)

    # Sample anomalous test data
    X_test_anom = []
    for _ in range(n_test_anom):
        s = rng.choice(anomaly_modes, p=eta_s)
        x = rng.multivariate_normal(nu_s[s], cov)
        X_test_anom.append(x)
    X_test_anom = np.array(X_test_anom)
    print(f"Shape of the data: {X_train.shape}")
    return X_train, X_test_normal, X_test_anom

def plot_robustness(ax, key, arrays, is_FP, show_ylabel=False):
    if is_FP:
        title = "Normal to Anomaly Evasion Attack on GMM"
    else:
        title = "Anomaly to Normal Evasion Attack on GMM"

    records = []
    for run_id, array in enumerate(arrays):
        arr = array['result_false_negative'] if is_FP else array['result_false_positive']
        df = pd.DataFrame(arr, columns=["x", "ROC", "PR"])
        df["Run"] = run_id
        df_long = df.melt(id_vars=["x", "Run"],
                          value_vars=["ROC", "PR"],
                          var_name="Values", value_name="Value")
        records.append(df_long)

    df_all = pd.concat(records, ignore_index=True)

    sns.lineplot(
        data=df_all.round(4),
        x="x",
        y="Value",
        hue="Values",
        ax=ax,
        estimator="mean",
        marker="o"
    )

    ax.set_title(f"Dim: {str(key).split('-')[-1]}", fontsize=12)
    ax.set_xlabel("Max Budget")
    if show_ylabel:
        ax.set_ylabel("AUC")
    else:
        ax.set_ylabel("")
        # ax.set_yticklabels([])

    ax.grid(True)
    ax.legend_.remove()

def plot_all_robustness(res_dict, root_path, is_FP=True):
    n = len(res_dict)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 3), sharey=True)

    if n == 1:
        axs = [axs]

    for i, (ax, (key, arrays)) in enumerate(zip(axs, res_dict.items())):
        show_ylabel = (i == 0)
        plot_robustness(ax, key, arrays, is_FP, show_ylabel=show_ylabel)

    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=1, fontsize='x-large')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(f"{root_path}/plots", exist_ok=True)
    plt.savefig(os.path.join(f"{root_path}/plots", f"combined_FP_{is_FP}.pdf"))
    plt.show()

if __name__ == "__main__":
    import argparse
    from utils import load_adbench_npz, set_seed
    import importlib

    import sys
    import os

    # Get the absolute path of the project root dynamically
    project_root = os.path.abspath(os.path.dirname(__file__))

    print(f"Add the directory {project_root} into system path of python. Please make sure the directory FMAD is under this directory.")
    # Add the project root to sys.path temporarily
    if project_root not in sys.path:
        sys.path.append(project_root)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--dname", type=str, default="synthesis")
    parser.add_argument("-i", "--dimensions", type=int, default=2)
    parser.add_argument("-r", "--random_seed", type=int, default=42)
    parser.add_argument("-p", "--path_result", type=str, default="./results_robustness", help="The path to store the results")
    args = parser.parse_args()

    dataset_name = args.dname
    model_index = 0
    seed = args.random_seed
    dimensions = args.dimensions
    set_seed(seed)

    print(f"Loading dataset {dataset_name}...")
    if dataset_name=="synthesis":
        # Generate simulated GMM data
        X_train, X_test_normal, X_test_anom = sample_gmm_data(d=dimensions, seed=seed)
        y_train = np.zeros(shape=X_train.shape[0]).astype(int)
        y_test = np.concatenate([np.zeros(X_test_normal.shape[0]).astype(int), np.ones(X_test_anom.shape[0]).astype(int)])
        X_test = np.vstack((X_test_normal, X_test_anom))
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed)

    mname = MODEL_NAMES[model_index]

    if mname in deep_models:
        module_path = deep_models[mname]
    else:
        import sys
        print(f"Model {mname} is not supported.")
        sys.exit(0)
    
    # Dynamically import the module only if needed
    module = importlib.import_module(module_path)
    model_class = getattr(module, mname)  # Get class dynamically

    parameters = {}
    
    # Initialize the model
    if dataset_name != "synthesis":
        raise NotImplementedError
    else:
        hyparam = {
            "epochs": 50,
            "learning_rate": 0.005,
            "batch_size": 1024
            }
        parameters.update({"n_features": X_train.shape[1]})
        parameters.update(hyparam)
        print(f"""Using epoch size: {parameters['epochs']}, learning rate: {parameters['learning_rate']}, batch_size: {parameters['batch_size']}
              on dataset {dataset_name}""")
    
    model_type = ""
    is_transductive = False
    model = model_class(**parameters)

    print(f"Evaluating Model: {mname}")
    path_prefix = args.path_result
    os.makedirs(path_prefix, exist_ok=True)

    filepath = os.path.join(path_prefix, f"models")
    os.makedirs(filepath, exist_ok=True)

    # Fit the TCCM detector
    if os.path.isfile(f"{filepath}/{dataset_name}-{mname}-{dimensions}-{seed}.pth"):
        state_dict = torch.load(f"{filepath}/{dataset_name}-{mname}-{dimensions}-{seed}.pth")
        model.load_state_dict(state_dict)
    else:
        with open(os.devnull, 'w') as f, redirect_stdout(f):  # Suppress output
            model.fit(X_train)
        torch.save(model.state_dict(), f"{filepath}/{dataset_name}-{mname}-{dimensions}-{seed}.pth")
    model.eval()
    if hasattr(model, 'predict_score') and callable(getattr(model, 'predict_score')):
        scores = model.predict_score(X_test)
    else:
        scores = model.decision_function(X_test)

    # Replace NaN and Inf values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    result = [mname, model_type]

    auc = roc_auc_score(y_test, scores)
    pr = average_precision_score(y_test, scores)
    print(f"Finish original {mname} AUC: {round(auc, 4)}, PR: {round(pr, 4)}")

    # PGD Attack on Anomaly Samples (drag them towards normal) or on Normal Samples (drag them towards anomaly)
    # x_adv_std = x_test_anom_std.clone() # or x_test_normal_std, depending on FN or FP attack
    def PGD_attack(FN_attack = True):
        target = 1 if FN_attack else 0
        result = np.zeros(shape=(len(EPS_RANGES)+1, 3))
        result[0, :] = [0, auc, pr]
        perturbation_holder = []
        for id_eps, eps in tqdm(enumerate(EPS_RANGES), desc=f"Is False Negative Attack? {FN_attack}. If True, change anomaly to normal"):
            x_adv = torch.from_numpy(X_test[y_test==target]).to(torch.float32)
            x_adv_copy = torch.from_numpy(X_test[y_test==target]).to(torch.float32)
            num_steps = int(round(eps/EPS_MINI_STEP * 2))
            if id_eps==0 and dimensions == 2:
                perturbation_holder.append(x_adv.cpu().numpy())
            for _ in range(num_steps):
                x_adv.requires_grad = True
                score = model(x_adv)
                loss = score.mean() if FN_attack else -score.mean()
                loss.backward()
                
                with torch.no_grad():
                    step = EPS_MINI_STEP * x_adv.grad.sign()
                    x_adv = x_adv - step if FN_attack else x_adv + step
                    x_adv = torch.clamp(x_adv, x_adv_copy - eps, x_adv_copy + eps) # L_infinity norm
            x_adv.requires_grad = False
            if hasattr(model, 'predict_score') and callable(getattr(model, 'predict_score')):
                scores_adv = model.predict_score(x_adv)
            else:
                scores_adv = model.decision_function(x_adv)
            scores_clone = scores.copy()
            scores_clone[y_test==target] = scores_adv

            auc_adv = roc_auc_score(y_test, scores_clone)
            pr_adv = average_precision_score(y_test, scores_clone)

            result[id_eps+1, :] = [eps, auc_adv, pr_adv]
            if dimensions == 2:
                perturbation_holder.append(x_adv.cpu().numpy())
        return result, perturbation_holder
    
    res_FN, pert_Xs_FN = PGD_attack()
    res_FP, pert_Xs_FP = PGD_attack(False)
        
    filepath = os.path.join(path_prefix, f"{dataset_name}-{mname}-{dimensions}-{seed}")
    try:
        np.savez_compressed(filepath + ".npz", result_false_negative=res_FN, result_false_positive=res_FP, pert_Xs_FN=pert_Xs_FN, pert_Xs_FP=pert_Xs_FP)
    except Exception as e:
        print(e)
        import pickle
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(dict(
                result=result,
                y_test=y_test,
                scores=scores
            ), f)
    
    print(f"[{dataset_name}] Done with evaluating ROBUSTNESS {model_type} Model {mname}.")

    files = os.listdir(path_prefix)
    res_dict = {}
    for fname in files:
        res_path = os.path.join(path_prefix, fname)
        if not os.path.isfile(res_path):
            continue
        names = res_path.split("-")
        dname = names[0].split("/")[-1]
        alg_name = names[1]
        dimension = names[2]
        seed = int(names[3].split(".")[0])
        
        result = np.load(res_path)
        res_dict.setdefault(f"{dname}-{alg_name}-{dimension}", []).append(result)

    sorted_dict = dict(sorted(res_dict.items(), key=lambda x: int(re.findall(r'\d+$', x[0])[0])))

    """
    Plot false positive attack, perturb normal samples to anomaly samples
    """
    plot_all_robustness(sorted_dict, root_path=path_prefix)

    """
    Plot false negative attack, perturb anomaly samples to normal samples
    """
    plot_all_robustness(sorted_dict, root_path=path_prefix, is_FP=False)