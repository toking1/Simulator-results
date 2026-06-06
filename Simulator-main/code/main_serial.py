# 全仿真实验运行脚本

import os
import time
from datetime import datetime

from node import Node
from plot_table_cloud import append_row

from run_simulation_d_sarsa import run_simulation as run_simulation_d_sarsa
from run_simulation_d_sarsa_cloud import run_simulation as run_simulation_d_sarsa_cloud
from run_simulations_no_learning import run_simulation as run_simulation_no_learning
from run_simulations_no_learning_cloud import run_simulation as run_simulation_no_learning_cloud
from run_simulation_d_sarsa_cloud_failure import run_simulation as run_simulation_failure

ALPHA_INCREMENT = 0.05
SESSION_ID = datetime.now().strftime("%Y%m%d")

PATH_RESULTS = "../results"
PATH_RESULTS_DATA = f"{PATH_RESULTS}/data"
PATH_RESULTS_TABLE = f"{PATH_RESULTS}/table"

os.makedirs(PATH_RESULTS_DATA, exist_ok=True)
os.makedirs(PATH_RESULTS_TABLE, exist_ok=True)


def timed(label, func, *args):
    """简单的计时包装器"""
    print(f"[{datetime.now()}] START {label}")
    t0 = time.time()
    func(*args)
    dt = time.time() - t0
    print(f"[{datetime.now()}] END   {label} | elapsed = {dt:.1f} s\n")


def run_all_experiments_serial():
    # 1) D-SARSA（只有边缘/worker 的版本）
    alpha_values = [i * ALPHA_INCREMENT for i in range(0, 21)]  # 0.00, 0.05, ..., 1.00
    total = len(alpha_values)

    for idx, alpha in enumerate(alpha_values, 1):
        label = f"D-SARSA alpha={alpha:.2f} ({idx}/{total})"
        timed(label, run_simulation_d_sarsa, alpha)

    # 2) 无学习策略（只有边缘/worker）
    policies = [
        Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,
        Node.NoLearningPolicy.MAXIMUM_LIFESPANE,
        Node.NoLearningPolicy.RANDOM,
    ]
    total_p = len(policies)

    for idx, policy in enumerate(policies, 1):
        label = f"No-learning policy={policy.name} ({idx}/{total_p})"
        timed(label, run_simulation_no_learning, policy)

    # 3) D-SARSA + Cloud
    for idx, alpha in enumerate(alpha_values, 1):
        label = f"D-SARSA-CLOUD alpha={alpha:.2f} ({idx}/{total})"
        timed(label, run_simulation_d_sarsa_cloud, alpha)

    # 4) 无学习策略 + Cloud
    for idx, policy in enumerate(policies, 1):
        label = f"No-learning-CLOUD policy={policy.name} ({idx}/{total_p})"
        timed(label, run_simulation_no_learning_cloud, policy)

    # 5) 故障场景（作者只用 alpha=1.0 和 0.0）
    failure_alphas = [1.0, 0.0]
    for idx, alpha in enumerate(failure_alphas, 1):
        label = f"Failure alpha={alpha:.2f} ({idx}/{len(failure_alphas)})"
        timed(label, run_simulation_failure, alpha)

    # 6) 汇总结果表
    build_summary_table()


def build_summary_table():
    # 和原始入口一样，从结果目录中汇总表格数据
    base_folder = f"{PATH_RESULTS_DATA}/_log/learning/D_SARSA/WORKERS_OR_CLOUD"
    alpha_range = [i / 100 for i in range(0, 101, 5)]  # 0.00, 0.05, ..., 1.00

    rows = []

    for alpha in alpha_range:
        folder_name = f"{base_folder}/{SESSION_ID}_{alpha:.2f}"
        print(f"[{datetime.now()}] scan learning folder: {folder_name}")
        if os.path.exists(folder_name):
            rows.append(append_row(folder_name, f"{alpha:.2f}"))

    base_folder_no = f"{PATH_RESULTS_DATA}/_log/no-learning"
    policies = [
        Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,
        Node.NoLearningPolicy.MAXIMUM_LIFESPANE,
        Node.NoLearningPolicy.RANDOM,
    ]

    for policy in policies:
        folder_name = f"{base_folder_no}/{policy.name}/{SESSION_ID}_{policy.name}_WORKERS_OR_CLOUD"
        print(f"[{datetime.now()}] scan no-learning folder: {folder_name}")
        if os.path.exists(folder_name):
            rows.append(append_row(folder_name, policy.name))

    os.makedirs(PATH_RESULTS_TABLE, exist_ok=True)
    file_path = f"{PATH_RESULTS_TABLE}/table_data_WORKERS_OR_CLOUD.txt"

    header = [
        "Alpha",
        "Mean Difference Batteries",
        "Max - Min batteries when first node die",
        "Min Lifespan",
        "Max Lifespan",
        "deadline / total job",
        "deadline / total type_0",
        "deadline / total type_1",
        "deadline / total type_2",
        "Time Service",
    ]

    with open(file_path, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

    print(f"[{datetime.now()}] Summary table written to {file_path}")


if __name__ == "__main__":
    print(f"[{datetime.now()}] ==== START ALL EXPERIMENTS (serial) ====")
    t0 = time.time()
    run_all_experiments_serial()
    dt = time.time() - t0
    print(f"[{datetime.now()}] ==== ALL DONE (serial) | total elapsed = {dt/3600:.2f} h ====")