import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
# **Suppress all warnings**
warnings.filterwarnings("ignore")


import time
import pandas as pd
from contextlib import redirect_stdout
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import set_seed

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
    "INNE_semisup": "pyod.models.inne",       # Isolation Nearest Neighbors (similar to KNN)
    "KNN_semisup": "pyod.models.knn",         # k-Nearest Neighbors (needs full dataset)
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
    "TCCM": "FMAD.FlowMatchingAD", # Our method
    "VAE": "pyod.models.vae",
    "SO_GAAL": "pyod.models.so_gaal",
    "MO_GAAL": "pyod.models.mo_gaal",
    "AutoEncoder": "pyod.models.auto_encoder", # verbose=0
    "DeepSVDD": "pyod.models.deep_svdd", # n_features=n_features
    "LUNAR": "pyod.models.lunar",
    "DIF": "pyod.models.dif",
    "ALAD": "pyod.models.alad",
    "AE1SVM": "pyod.models.ae1svm",
    "AnoGAN": "pyod.models.anogan"
}

additional_models = {
    "DAGMM": "baselines.dagmm", # Use it in semi-supervised manner
    "GANomaly": "baselines.ganomaly",
    "NormalizingFlow": "baselines.normalizingFlow",
    "DROCC": "baselines.drocc",
    "DTEDDPM": "baselines.diffusion.ddpm",
    "DTEGaussian": "baselines.diffusion.dte",
    "DTEInverseGamma": "baselines.diffusion.dte",
    "DTECategorical": "baselines.diffusion.dte",
    "DTENonParametric": "baselines.diffusion.dteNonParam",
    "ICL": "baselines.icl",
    "GOAD": "baselines.goad",
    "SLAD": "baselines.slad",
    "MCM": "baselines.mcm"
}

MODEL_NAMES = list(deep_models.keys()) + \
    list(transductive_models.keys()) + \
        list(inductive_models.keys()) + \
            list(additional_models.keys()) + \
                list(force_inductive_models.keys())

import time
import multiprocessing
import numpy as np
import pandas as pd
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score

def limit_memory(max_mem_gb):
    """Restricts memory usage for the process (Linux only). This does not work well on MacOS."""
    try:
        import resource  # For memory limiting on Linux
    except ImportError:
        resource = None
    
    if resource:
        max_mem_bytes = int(max_mem_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        resource.setrlimit(resource.RLIMIT_AS, (max_mem_bytes, max_mem_bytes))

def model_worker(queue, func, args, mem_limit):
    """ Worker function that runs inside a multiprocessing process. """
    try:
        # Set memory limit for the worker process (Linux only)
        limit_memory(mem_limit) # Hard limit, kill the process if exceeded memory limit
        result = func(*args)
        queue.put(result)
        print("Finish training and evaluation.")
    except Exception as e:
        print("ERROR!")
        queue.put(("ModERROR", str(e)))

def run_model_with_timeout(func, args, time_limit, mem_limit):
    """ Run a function with a strict time limit using multiprocessing. """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=model_worker, args=(queue, func, args, mem_limit))
    process.start()
    start_time = time.time()
    
    interval = 10 # Perform a check every 10 seconds
    content = ["MemoryLimitReachedERROR", None, None]
    while True:
        if not queue.empty():
            status, result, train_time, test_time = queue.get()
            if status == "OK":
                content = [result, train_time, test_time]
            else:
                content = [status + ": " + result, None, None]
            break
        if time.time() - start_time > time_limit:
            print(f"Timeout! Skipping (>{time_limit} sec)")
            process.terminate()
            process.join()
            content = ["Timeout", None, None]  # Reach time limits
            break
        time.sleep(interval) # Perform another check after 'interval' seconds

    if process.is_alive():
        process.terminate()
        process.join()
    return content

def train_and_eval(model, X_train, X_test, is_transductive, random_seed=None):
    """
    Train and evaluate a model.
    
    - If `is_transductive` is True, train on X_test (unsupervised).
    - Otherwise, train on X_train and test on X_test.
    """
    train_time = None
    test_time = None
    if random_seed is not None:
        set_seed(random_seed)
    try:
        start_time = time.time()
        if is_transductive:
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                model.fit(X_test)
        else:
            start_time = time.time()
            with open(os.devnull, 'w') as f, redirect_stdout(f):
                model.fit(X_train)
        train_time = time.time() - start_time
        print(f"Finish training of {str(model)}")
    except Exception as e:
        return ("ModERROR", str(e), train_time, test_time)
    else:
        try:
            start_time_test = time.time()
            if hasattr(model, 'predict_score') and callable(getattr(model, 'predict_score')):
                result = model.predict_score(X_test)
            else:
                result = model.decision_function(X_test)
            test_time = time.time() - start_time_test
        except Exception as e:
            return ("ModERROR", str(e), train_time, test_time)
        else:
            return ("OK", result, train_time, test_time)

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
    parser.add_argument("-d", "--dname", type=str)
    parser.add_argument("-i", "--model_index", type=int)
    parser.add_argument("-r", "--random_seed", type=int, default=0)
    parser.add_argument("-t", "--time_limit", type=int, default=7200, help="Time limit in second")
    parser.add_argument("-m", "--memory_limit", type=float, default=None, help="Memory limit in GB")
    parser.add_argument("-p", "--path_result", type=str, default="./results_all", help="The path to store the results")
    args = parser.parse_args()

    dataset_name = args.dname
    model_index = args.model_index
    seed = args.random_seed
    set_seed(seed)

    print(f"Loading dataset {dataset_name}...")
    X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed) # By default, we use 0
    
    mname = MODEL_NAMES[model_index] # All models

    if mname in transductive_models:
        module_path = transductive_models[mname]
    elif mname in force_inductive_models:
        module_path = force_inductive_models[mname]
    elif mname in inductive_models:
        module_path = inductive_models[mname]
    elif mname in deep_models:
        module_path = deep_models[mname]
    elif mname in additional_models:
        module_path = additional_models[mname]
    else:
        import sys
        print(f"Model {mname} is not supported.")
        sys.exit(0)
    
    # Dynamically import the module only if needed
    module = importlib.import_module(module_path)
    if mname in force_inductive_models:
        model_name = mname[:mname.rfind("_semisup")]
        model_class = getattr(module, model_name)
    else:
        model_class = getattr(module, mname)  # Get class dynamically

    parameters = {}
    # Initialize the model
    if mname == "AutoEncoder":
        parameters.update({"verbose": 0})
    elif mname == "DeepSVDD":
        parameters.update({"n_features": X_train.shape[1]})
    elif mname == "TCCM":
        from FMAD.functions import determine_FMAD_hyperparameters
        hyparam = determine_FMAD_hyperparameters(dataset_name)
        parameters.update({"n_features": X_train.shape[1]})
        parameters.update(hyparam)
            
    elif mname in additional_models:
        if mname == "MCM":
            parameters = {"n_features": X_train.shape[1]}
        elif mname == "GOAD":
            parameters = {"n_epoch": 25}
        else:
            parameters = {}
    
    model_type = ""
    is_transductive = False
    if mname in transductive_models:
        is_transductive = True
        model_type = "Transductive"
    elif mname in force_inductive_models:
        model_type = "Force Inductive"
    elif mname in inductive_models:
        model_type = "Inductive"
    elif mname in deep_models:
        model_type = "Deep Learning"
    elif mname in additional_models:
        model_type = "SOTA"
    model = model_class(**parameters)

    time_limit = args.time_limit
    mem_limit_gb = args.memory_limit
    print(f"Evaluating {model_type} Model: {mname} with time limit {time_limit} seconds, and memory limit {mem_limit_gb}GB.")
    model_func = partial(train_and_eval, model, X_train, X_test, is_transductive, random_seed = seed)

    if mem_limit_gb is None:
        mem_limit_gb = 9999 # No limit

    start_time = time.time()

    summary = run_model_with_timeout(model_func, (), time_limit, mem_limit_gb)
    scores = summary[0]
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)

    path_prefix = args.path_result
    os.makedirs(path_prefix, exist_ok=True)
    # Replace NaN and Inf values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
    result = [mname, model_type]
    if isinstance(scores, str):
        print(f"Skipping {mname} due to error: {scores}")
        result+=[scores, scores, elapsed_time]
        scores = None
        auc = None
        pr = None
    else:
        auc = roc_auc_score(y_test, scores)
        pr = average_precision_score(y_test, scores)
        result+=[auc, pr, elapsed_time]
    result+=[summary[1], summary[2]]
    filepath = os.path.join(path_prefix, f"{dataset_name}-{mname}-{seed}")
    try:
        np.savez_compressed(filepath + ".npz", result=result, y_test=y_test, scores=scores)
    except Exception as e:
        print(e)
        import pickle
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(dict(
                result=result,
                y_test=y_test,
                scores=scores
            ), f)
    
    print(f"[{dataset_name}] Done with evaluating {model_type} Model {mname}.")
    print(f"Elapsed time of fitting {summary[1]} seconds, testing {summary[2]} seconds, and in total {elapsed_time} seconds, with AUC: {auc}, PR: {pr}")