import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import warnings
# **Suppress all warnings**
warnings.filterwarnings("ignore")


import time
import multiprocessing
import numpy as np
from utils import set_seed
from functools import partial

from contextlib import redirect_stdout
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

###############################################
# We only test the top 10 performed models based on AUROC on the ADBench dataset
###############################################
top10_models = {
    "TCCM": "FMAD.FlowMatchingAD", # Our method
    "DTENonParametric": "baselines.diffusion.dteNonParam", # Diffusion Time Estimation Non-Parametric variant
    "LUNAR": "pyod.models.lunar",     # Unifying Local Outlier Detection Methods via Graph Neural Networks
    "KDE": "pyod.models.kde",         # Kernel Density Estimation, fits a probability distribution
    "AutoEncoder": "pyod.models.auto_encoder", # AutoEncoder, deep reconstruction-based
    "CBLOF": "pyod.models.cblof",     # Clustering-Based, applies to new data
    "DTECategorical": "baselines.diffusion.dte", # Diffusion Time Estimation Categorical variant
    "GMM": "pyod.models.gmm",         # Gaussian Mixture Model, learns statistical distribution
    "OCSVM": "pyod.models.ocsvm",     # One-Class SVM, learns a separating hyperplane
    "DTEGaussian": "baselines.diffusion.dte", # Diffusion Time Estimation Gaussian variant
}

MODEL_NAMES = list(top10_models.keys())

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
            with open(os.devnull, 'w') as f, redirect_stdout(f):  # Suppress output, namely omitting logs
                model.fit(X_test)
        else:
            start_time = time.time()
            with open(os.devnull, 'w') as f, redirect_stdout(f):  # Suppress output
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
    parser.add_argument("-p", "--path_result", type=str, default="./results_contam", help="The path to store the results")
    parser.add_argument("-c", "--contam_lv_idx", type=int, default=0)
    args = parser.parse_args()

    dataset_name = args.dname
    model_index = args.model_index
    seed = args.random_seed
    set_seed(seed)

    print(f"Loading dataset {dataset_name}...")
    X_train, y_train, X_test, y_test = load_adbench_npz(dataset_name, random_state=seed) # By default, we use 0

    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])

    X_normal = X_all[y_all == 0]
    X_abnormal = X_all[y_all == 1]
    
    X_train_normal, X_test_normal = train_test_split(X_normal, test_size=0.5, random_state=42, stratify=None)
    X_train_abnormal_full, X_test_abnormal = train_test_split(X_abnormal, test_size=0.5, random_state=42, stratify=None)

    X_test = np.vstack([X_test_normal, X_test_abnormal])
    y_test = np.concatenate([np.zeros(len(X_test_normal)), np.ones(len(X_test_abnormal))])

    n_train_normal = len(X_train_normal)
    n_train_abnormal_max = len(X_train_abnormal_full)
    max_abnormal_ratio = n_train_abnormal_max / (n_train_abnormal_max + n_train_normal)

    contamination_levels = np.linspace(0.001, max_abnormal_ratio, 10)
    contam_lv_idx = args.contam_lv_idx
    
    contam_ratio = contamination_levels[contam_lv_idx]

    n_ab = int(contam_ratio * n_train_normal / (1 - contam_ratio))
    n_ab = min(n_ab, n_train_abnormal_max)

    selected_ab = X_train_abnormal_full[:n_ab]  # TODO: Can be replaced by random sample
    X_train = np.vstack([X_train_normal, selected_ab])
    y_train = np.zeros(len(X_train))  # Not used

    mname = MODEL_NAMES[model_index] # All models

    if mname in top10_models:
        module_path = top10_models[mname]
    else:
        import sys
        print(f"Model {mname} is not supported.")
        sys.exit(0)
    
    # Dynamically import the module only if needed
    module = importlib.import_module(module_path)
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
    
    model_type = ""
    is_transductive = False
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
    filepath = os.path.join(path_prefix, f"{dataset_name}-{mname}-{seed}-{contam_lv_idx}")
    try:
        np.savez_compressed(filepath + ".npz", 
                            result=result, 
                            y_test=y_test, 
                            scores=scores,
                            contam_ratio=contam_ratio)
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