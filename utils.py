import random
import os
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# The utility function to load one specific dataset from the ADbench datasets
def load_adbench_npz(dataset_name="42_WBC", test_size=0.5, random_state=0):
    base_paths = ["./datasets/small/", "./datasets/medium/", "./datasets/high_dim/", "./datasets/large/"]
    file_path = None
    for b_path in base_paths:
        if os.path.isfile(f"{b_path}{dataset_name}.npz"):
            file_path = f"{b_path}{dataset_name}.npz"
    if file_path is None:
        raise FileNotFoundError(f"Dataset {dataset_name}.npz not found!")
    # if not os.path.exists(file_path):
    #     raise FileNotFoundError(f"Dataset {dataset_name}.npz not found!")

    data = np.load(file_path, allow_pickle=True)
    X, y = data["X"], data["y"].astype(int)

    # Train using only normal samples
    X_normal, X_anomalous = X[y == 0], X[y == 1]
    y_normal, y_anomalous = y[y == 0], y[y == 1]

    X_train, X_test_normal, y_train, y_test_normal = train_test_split(
        X_normal, y_normal, test_size=test_size, random_state=random_state, stratify=y_normal
    )

    # Test set contains both normal and abnormal data
    X_test = np.vstack((X_test_normal, X_anomalous))
    y_test = np.concatenate((y_test_normal, y_anomalous))

    # Data standardization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Print dataset information
    print(f"Dataset {dataset_name} loaded successfully!")
    print(f"Training data: {X_train.shape}, Normal: {len(y_train)}")
    print(f"Test data: {X_test.shape}, Normal: {sum(y_test == 0)}, Anomalies: {sum(y_test == 1)}")

    return X_train, y_train, X_test, y_test