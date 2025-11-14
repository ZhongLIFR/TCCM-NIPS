# TCCM-NIPS
This is the official repository of the NeurIPS 2025 paper entitled: [Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](https://neurips.cc/virtual/2025/poster/116486)


# Updates

2025.10.22: We finished the camera-ready version and uploaded it on [ArXiv](https://arxiv.org/abs/2510.18328)

2025.10.16: We are cleaning the code and will upload it soon.

2025.11.11: The majority of the code is ready.

## Instructions on reproducing the main results
### Setup Instructions
The codebase is established on python 3.9.21 and works on Linux OS.

1. Download or cloning this repository to your local PC or cluster (preferable)

2. Install the required python and packages
```bash
pip install -r requirements.txt
```

### Datasets
The used ADBench datasets are placed under the ./datasets directory

All datasets are split into four categories: small, medium, high-dimensional (high_dim), and large (Table 1)

### Main Results
#### To reproduce the benchmark results showed in Figure 2 (AUROC and AUPRC)
RUN FROM THE ROOT DIRECTORY OF THIS REPO:

```bash
chmod u+x ./bash_files/run_main.sh
./bash_files/run_main.sh
```

The log of each run is stored under the directory "./logs" as:

"./logs/seed_```$RANDOM_SEED```/run_```{dataset_ID_NAME}```_model_```{model_ID}```.log"

Once all the runs are done, you will see the message:

```LARGE datasets processed.``` at the end of the log file: ./logs/All_log.log

##### NOTE: As requested by the reviewers, we also performed additional experiments on enforcing transductive models to behave inductively.

```bash
chmod u+x ./bash_files/run_semisupervise.sh
./bash_files/run_semisupervise.sh
```

All experimental results shall be saved in: ./results_all

#### Retrieve results and visualization
```bash
# Delete the --semi_only flag if you would like the transductive models to run in their standard mode.
python AggregateResults.py --semi_only
# [PENDING] We are actively working on making clean code for visualizing all plots
# python Visualization.py
```
The AUROC and AUPRC plots will be saved as ```Rank_ROC.pdf``` and ```Rank_PR.pdf```.

The dataset-wise tabular results as well as the rankings of models are saved under ```./final_metrics/all```.

NOTES:
- All experiment runs are executed on CPU only.
- For each dataset, we allow each algorithm to use a maximum of 10 GB of RAM and a maximum runtime of 3 days.
    - You can change the ```MEMORY_LIMIT``` and ```TIME_LIMIT``` in ```run_everything.sh``` accordingly.
- By default, we run 45 × (```K``` = 3) = 135 jobs in parallel. Please adjust the value of ```K``` as needed.
- By default, the experiments are executed on all datasets for all models, which may result in a significantly long execution time.
- The performance may differ slightly (regarding numerical precision) from the results reported in the paper due to randomness introduced by CPU specifications. However, **we have verified the results on at least two very different clusters, and the overall ranking of models remains consistent**.

### Ablation Studies
To reproduce the results of ablation studies as shown in Appendix D.3

Please run the following command from the root directory of this project:
```bash
python AblationStudies.py
```

The results (plots) will be saved under the folder ```./results_ablation```:
- Time_Embedding_Figure_12.pdf: Study 1 Time Embedding Variants used in TCCM.
- Sensitivity_t_Figure_13.pdf: Study 2 Sensitivity to Fixed Time t during Inference.
- Noise_Injection_Figure_14.pdf: Study 3 Effect of Noise Injection during Training.
- Contamination_Figure_TCCM.pdf: Study 4 Effect of Contamination in Training Data for TCCM.

### Empirical Robustness Verification
To reproduce the results of empirical robustness verification for TCCM as shown in Appendix D.4.

Please run the following command from the root directory of this project:
```bash
chmod u+x ./bash_files/run_robustness.sh
./bash_files/run_robustness.sh
```

The results (both raw data and plots) are located in the ```./results_robustness``` directory:
- combined_FP_False.pdf: False Negative Attack – aims to make anomaly samples appear as normal.
- combined_FP_True.pdf: False Positive Attack – aims to make normal samples appear as anomalies.

### Empirical Experiments of Robustness under Contaminated Training Data
To reproduce the results of empirical experiments of robustness under contaminated training data for top-10 performed models.

Please run the following command from the root directory of this project:
```bash
chmod u+x ./bash_files/run_contamination.sh
./bash_files/run_contamination.sh
# Extract and visualize results
python ProcessContamination.py
```
The results (raw data, CSV table, and plots) are located in the ```./results_contam``` directory.
