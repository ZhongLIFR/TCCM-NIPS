import numpy as np
import os
import pandas as pd
import argparse

DATASETS = {
    "small": ["4_breastw", "14_glass", "15_Hepatitis", "18_Ionosphere", "21_Lymphography",
              "29_Pima", "37_Stamps", "39_vertebral", "42_WBC", "43_WDBC", "45_wine", "46_WPBC"], 
    "medium": ["2_annthyroid","6_cardio","7_Cardiotocography","12_fault","19_landsat","20_letter",
               "27_PageBlocks","28_pendigits","30_satellite","31_satimage-2","38_thyroid","40_vowels",
               "41_Waveform","44_Wilt","47_yeast"],
    "high_dim": ['3_backdoor', '5_campaign', '9_census', '17_InternetAds', 
                 '24_mnist', '25_musk', '26_optdigits', '35_SpamBase', '36_speech'],
    "large": ['1_ALOI', '8_celeba', '10_cover', '11_donors', '13_fraud', '16_http', '22_magic.gamma', 
              '23_mammography', '32_shuttle', '33_skin', '34_smtp']
}

# Transductive Models (Require full dataset during prediction, do NOT generalize)
transductive_models = {
    "ABOD": "pyod.models.abod",       # Angle-Based method, needs full dataset
    "COF": "pyod.models.cof",         # Connectivity-Based, like LOF
    "LOF": "pyod.models.lof",         # Local Outlier Factor, relies on neighbors
    "PCA": "pyod.models.pca",         # Principal Component Analysis (score depends on dataset)
    "KPCA": "pyod.models.kpca",       # Kernel PCA (like PCA but non-linear)
    "KNN": "pyod.models.knn",         # k-Nearest Neighbors (needs full dataset)
    "INNE": "pyod.models.inne",       # Isolation Nearest Neighbors (similar to KNN)
}

force_inductive_models = {
    "ABOD_semisup": "pyod.models.abod",       # Angle-Based method, needs full dataset
    "COF_semisup": "pyod.models.cof",         # Connectivity-Based, like LOF
    "LOF_semisup": "pyod.models.lof",         # Local Outlier Factor, relies on neighbors
    "PCA_semisup": "pyod.models.pca",         # Principal Component Analysis (score depends on dataset)
    "KPCA_semisup": "pyod.models.kpca",       # Kernel PCA (like PCA but non-linear)
    "KNN_semisup": "pyod.models.knn",         # k-Nearest Neighbors (needs full dataset)
    "INNE_semisup": "pyod.models.inne",       # Isolation Nearest Neighbors (similar to KNN)
}

# Inductive Models (Learn a function/model that generalizes to unseen data)
inductive_models = {
    "CBLOF": "pyod.models.cblof",     # Clustering-Based, applies to new data
    "IForest": "pyod.models.iforest", # Isolation Forest, learns a decision model
    "LODA": "pyod.models.loda",       # Histogram-Based, generalizes anomaly scores
    "FeatureBagging": "pyod.models.feature_bagging",  # Ensemble method, applies trained models
    "Sampling": "pyod.models.sampling",  # Statistical-based sampling for anomalies
    "MCD": "pyod.models.mcd",         # Robust covariance estimator, applies to new data
    "CD": "pyod.models.cd",           # Class Distribution, uses decision rules
    "ECOD": "pyod.models.ecod",       # Empirical CDF-based, estimates probability function
    "HBOS": "pyod.models.hbos",       # Histogram-Based, fits probability density
    "OCSVM": "pyod.models.ocsvm",     # One-Class SVM, learns a separating hyperplane
    "KDE": "pyod.models.kde",         # Kernel Density Estimation, fits a probability distribution
    "GMM": "pyod.models.gmm",         # Gaussian Mixture Model, learns statistical distribution
    "QMCD": "pyod.models.qmcd",       # Quantile-Based CDF Estimation
    "LMDD": "pyod.models.lmdd",       # Limit Distribution Difference
}

# **Deep Learning Models (Require Training)**
deep_models = {
    "TCCM": "FMAD.FlowMatchingAD",              # Our method
    "VAE": "pyod.models.vae",                   # Variational Autoencoder
    "SO_GAAL": "pyod.models.so_gaal",           # Single-Objective Generative Adversarial Active Learning
    "MO_GAAL": "pyod.models.mo_gaal",           # Multi-Objective Generative Adversarial Active Learning
    "AutoEncoder": "pyod.models.auto_encoder",  # AutoEncoder
    "DeepSVDD": "pyod.models.deep_svdd",        # Deep Support Vector Data Description, a deep one-class classification
    "LUNAR": "pyod.models.lunar",               # Learnable Unified Neighbourhood-based Anomaly Ranking, a GNN method
    "DIF": "pyod.models.dif",                   # Deep Isolation Forest
    "ALAD": "pyod.models.alad",                 # Adversarially Learned Anomaly Detection
    "AE1SVM": "pyod.models.ae1svm",             # Autoencoder + One-Class SVM hybrid
    "AnoGAN": "pyod.models.anogan"              # Anomaly detection with Generative Adversarial Networks
}

additional_models = {
    "DAGMM": "baselines.dagmm",                 # Deep Autoencoding Gaussian Mixture Model
    "GANomaly": "baselines.ganomaly",           # Semi-Supervised Anomaly Detection via Adversarial Training, not to be confused with AnoGAN
    "NormalizingFlow": "baselines.normalizingFlow", # Normalizing Flows for Anomaly Detection
    "DROCC": "baselines.drocc",                 # Deep Robust One-Class Classification
    "DTEDDPM": "baselines.diffusion.ddpm",      # Diffusion-based method
    "DTEGaussian": "baselines.diffusion.dte",   # Diffusion Time Estimation Gaussian variant
    "DTEInverseGamma": "baselines.diffusion.dte",   # Diffusion Time Estimation Inverse Gamma variant
    "DTECategorical": "baselines.diffusion.dte",    # Diffusion Time Estimation Categorical variant
    "DTENonParametric": "baselines.diffusion.dteNonParam",  # Diffusion Time Estimation Non-Parametric variant
    "ICL": "baselines.icl",                     # Internal Contrastive Learning for Anomaly Detection
    "GOAD": "baselines.goad",                   # Classification-Based Anomaly Detection for General Data
    "SLAD": "baselines.slad",                   # Scale Learning for Anomaly Detection
    "MCM": "baselines.mcm"                      # Masked Cell Modeling for Anomaly Detection
}

ALL_MODELS = list(deep_models.keys()) + \
    list(transductive_models.keys()) + \
             list(inductive_models.keys()) + \
             list(additional_models.keys()) + \
                list(force_inductive_models.keys())

if __name__ == "__main__":
    # Directory containing all .npz files
    data_dir = "./results_all" # By default
    base_metric_path = "./final_metrics"
    # Dictionary to organize files
    exp_dict = {}
    argp = argparse.ArgumentParser()
    argp.add_argument("-s", "--split_index", type=int, default=0, help="Which splits: all(0), small (1), medium (2), high dimensional (3), large (4).")
    argp.add_argument("--semi_only", action="store_true", help="If true, only semi-supervised models are considered.")
    argp.set_defaults(semi_only=True)

    args = argp.parse_args()
    # split = "all"  # default split
    split = ["all", "small", "medium", "high_dim", "large"][args.split_index]

    # Loop through the directory
    for filename in os.listdir(data_dir):
        if filename.endswith(".npz"):
            file_path = os.path.join(data_dir, filename)
            
            # Splitting filename into meaningful categories
            parts = filename.split("-")
            if "31_satimage-2" in filename:
                dataset_name = parts[0] + "-" + parts[1]
                model_name = parts[2]
                random_seed = int(parts[3].replace(".npz", ""))
            else:
                dataset_name = parts[0]  # e.g., "1_ALOI", "3_backdoor", etc.
                model_name = parts[1]  # Extract model name
                random_seed = int(parts[2].replace(".npz", ""))  # Extract model name
            if model_name not in ALL_MODELS:
                continue
            if args.semi_only:
                if model_name in transductive_models:
                    continue
            else:
                if model_name in force_inductive_models:
                    continue

            with open(file_path, "rb") as f:
                temp = np.load(f, allow_pickle=True)
                data = {key: temp[key] for key in temp}

            if random_seed not in exp_dict:
                exp_dict[random_seed]={}
            if dataset_name not in exp_dict[random_seed]:
                exp_dict[random_seed][dataset_name] = {}
            exp_dict[random_seed][dataset_name][model_name] = data # It has three keys results, y_test, scores

    # Print the available categories and models
    for random_seed, dataset_dict in exp_dict.items():
        print(f"Random Seed: {random_seed}")
        for data_name in list(dataset_dict.keys()):
            # print(f"Dataset: {data_name}")
            if data_name not in DATASETS['large'] and data_name not in DATASETS['small'] and data_name not in DATASETS['medium'] and data_name not in DATASETS['high_dim']:
                dataset_dict.pop(data_name, None)
                continue
            for model_name in ALL_MODELS:
                if args.semi_only:
                    if model_name in transductive_models:
                        continue
                if model_name not in dataset_dict[data_name]:
                    dataset_dict[data_name][model_name] = {"AUC": None, "PR": None, "~ExecTimeSeconds": None}
                else:
                    summary = dataset_dict[data_name][model_name]
                    res = summary['result'][2:]
                    dataset_dict[data_name][model_name] = {"AUC": res[0], "PR": res[1], "~ExecTimeSeconds": res[2], "~TrainTimeSeconds": res[3], "~TestTimeSeconds": res[4]}
        dataset_dict = dict(sorted(dataset_dict.items()))
        exp_dict.update({random_seed: dataset_dict})

    os.makedirs(f"{base_metric_path}/{split}", exist_ok=True)
    dfs_by_seeds = []
    pd.DataFrame()
    for seed_idx, seed in enumerate(exp_dict):
        df = pd.DataFrame.from_dict(
             {(category, model): metrics for category, models in exp_dict[seed].items() for model, metrics in models.items()},
             orient='index'
        )
        sorted_df = df.groupby(level=0, group_keys=False).apply(lambda x: x.sort_index(level=1))
        sorted_df.to_csv(f"{base_metric_path}/{split}/Experiments_random_seed_{seed}.csv")
        
        
        sorted_df['AUC'] = pd.to_numeric(sorted_df['AUC'], errors="coerce")
        sorted_df['PR'] = pd.to_numeric(sorted_df['PR'], errors="coerce")
        sorted_df['~ExecTimeSeconds'] = pd.to_numeric(sorted_df['~ExecTimeSeconds'], errors="coerce")
        sorted_df['~TrainTimeSeconds'] = pd.to_numeric(sorted_df['~TrainTimeSeconds'], errors="coerce")
        sorted_df['~TestTimeSeconds'] = pd.to_numeric(sorted_df['~TestTimeSeconds'], errors='coerce')
        dfs_by_seeds.append(sorted_df)

    # -----------------------------
    # Load & Preprocess CSVs
    # -----------------------------
    file_template = "Experiments_random_seed_{}.csv"
    seeds = [0, 1, 2, 3, 4]
    df_list = []

    for seed in seeds:
        file_path = os.path.join(f"{base_metric_path}/{split}", file_template.format(seed))
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df.rename(columns={df.columns[0]: "Dataset", df.columns[1]: "Detector"}, inplace=True)

            # Convert relevant columns to numeric safely
            df["PR"] = pd.to_numeric(df["PR"], errors="coerce").fillna(-1)
            df["AUC"] = pd.to_numeric(df["AUC"], errors="coerce").fillna(-1)
            df['~TrainTimeSeconds'] = pd.to_numeric(df['~TrainTimeSeconds'], errors="coerce").fillna(300000)
            df['~TestTimeSeconds'] = pd.to_numeric(df['~TestTimeSeconds'], errors='coerce').fillna(300000)
            df["Seed"] = seed

            # Compute per-seed, per-dataset ranking (lower rank = better)
            # df["Rank_PR"] = df.groupby(["Seed", "Dataset"])["PR"].rank(ascending=False, method="min")
            # df["Rank_AUC"] = df.groupby(["Seed", "Dataset"])["AUC"].rank(ascending=False, method="min")
            df["Rank_PR"] = df.groupby(["Seed", "Dataset"])["PR"].rank(ascending=False, method="min")
            df["Rank_AUC"] = df.groupby(["Seed", "Dataset"])["AUC"].rank(ascending=False, method="min")

            df_list.append(df)
        else:
            print(f"File missing for Seed {seed}: {file_path}")

    # -----------------------------
    # Aggregate Metrics across seeds
    # -----------------------------
    if df_list:
        df_combined = pd.concat(df_list, ignore_index=True)

        # Compute mean/std of metrics for each Detector x Dataset combo
        df_aggregated = df_combined.groupby(["Dataset", "Detector"]).agg(
            Mean_PR=("PR", "mean"),
            Mean_AUC=("AUC", "mean"),
            Std_PR=("PR", "std"),
            Std_AUC=("AUC", "std"),
            Mean_ExecTime=("~ExecTimeSeconds", "mean"),
            Std_ExecTime=("~ExecTimeSeconds", "std"),
            Mean_TrainTime=("~TrainTimeSeconds", "mean"),
            Std_TrainTime=("~TrainTimeSeconds", "std"),
            Mean_TestTime=("~TestTimeSeconds", "mean"),
            Std_TestTime=("~TestTimeSeconds", "std"),
        ).reset_index()

        # Round numeric values for cleaner display
        df_aggregated[df_aggregated.select_dtypes(include=[np.number]).columns] = \
            df_aggregated.select_dtypes(include=[np.number]).round(3)

        # Rank detectors by Mean PR and AUC within each dataset
        df_aggregated["Rank_PR"] = df_aggregated.groupby("Dataset")["Mean_PR"].rank(ascending=False, method="min")
        df_aggregated["Rank_AUC"] = df_aggregated.groupby("Dataset")["Mean_AUC"].rank(ascending=False, method="min")

        # Rename AUC → ROC to avoid confusion in plots
        df_aggregated.rename(columns={
            "Mean_AUC": "Mean_ROC",
            "Std_AUC": "Std_ROC",
            "Rank_AUC": "Rank_ROC"
        }, inplace=True)

    # -----------------------------
    # Define Detector Categories & Colors
    # -----------------------------
    transductive_detectors = ["ABOD", "COF", "LOF", "PCA", "KPCA", "KNN", "INNE"]
    force_inductive_detectors = ["ABOD_semisup", "COF_semisup", "LOF_semisup", "PCA_semisup", "KPCA_semisup", "KNN_semisup", "INNE_semisup"]
    inductive_detectors = [
        "CBLOF", "IForest", "LODA", "FeatureBagging", "Sampling", "MCD", "CD",
        "ECOD", "HBOS", "OCSVM", "KDE", "GMM", "QMCD", "LMDD"
    ]

    def assign_category(detector):
        if detector in transductive_detectors:
            return "Transductive"
        elif detector in force_inductive_detectors:
            return "Force Inductive"
        elif detector in inductive_detectors:
            return "Inductive"
        else:
            return "Deep Learning"

    df_aggregated["Category"] = df_aggregated["Detector"].apply(assign_category)
    df_aggregated.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}.csv"))
    order_roc = df_aggregated.groupby("Detector")["Rank_ROC"].mean().sort_values()
    order_roc.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank.csv"))
    order_pr = df_aggregated.groupby("Detector")["Rank_PR"].mean().sort_values()
    order_pr.to_csv(os.path.join(f"{base_metric_path}/{split}", f"Results_{split}_rank_PR.csv"))