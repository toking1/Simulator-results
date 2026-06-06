# 全仿真实验运行脚本

import os
import subprocess
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
PATH_RESULTS_PLOT = f"{PATH_RESULTS}/plot"

os.makedirs(PATH_RESULTS_DATA, exist_ok=True)
os.makedirs(PATH_RESULTS_TABLE, exist_ok=True)
os.makedirs(PATH_RESULTS_PLOT, exist_ok=True)


def timed(label, func, *args):
    """简单的计时包装器"""
    print(f"[{datetime.now()}] START {label}")
    t0 = time.time()
    func(*args)
    dt = time.time() - t0
    print(f"[{datetime.now()}] END   {label} | elapsed = {dt:.1f} s\n")


def run_all_experiments_serial():
    alpha_values = [i * ALPHA_INCREMENT for i in range(0, 21)]  # 0.00, 0.05, ..., 1.00
    total = len(alpha_values)

    # -------------------------------------------------------
    # 1) D-SARSA（ONLY_WORKERS，无 cloud）
    # -------------------------------------------------------
    for idx, alpha in enumerate(alpha_values, 1):
        label = f"D-SARSA alpha={alpha:.2f} ({idx}/{total})"
        timed(label, run_simulation_d_sarsa, alpha)

    # -------------------------------------------------------
    # 2) 无学习策略（ONLY_WORKERS，无 cloud）
    #    注意：LEAST_LOADED_AWARE_CLOUD 必须配合 WORKERS_OR_CLOUD，
    #    不能放在 ONLY_WORKERS 场景中，否则会触发 RuntimeError。
    # -------------------------------------------------------
    policies_only_workers = [
        Node.NoLearningPolicy.LEAST_LOADED_NOT_AWARE,
        Node.NoLearningPolicy.MAXIMUM_LIFESPANE,
        Node.NoLearningPolicy.RANDOM,
    ]
    total_p = len(policies_only_workers)
    for idx, policy in enumerate(policies_only_workers, 1):
        label = f"No-learning ONLY_WORKERS policy={policy.name} ({idx}/{total_p})"
        timed(label, run_simulation_no_learning, policy)

    # -------------------------------------------------------
    # 3) D-SARSA + Cloud（WORKERS_OR_CLOUD）
    # -------------------------------------------------------
    for idx, alpha in enumerate(alpha_values, 1):
        label = f"D-SARSA-CLOUD alpha={alpha:.2f} ({idx}/{total})"
        timed(label, run_simulation_d_sarsa_cloud, alpha)

    # -------------------------------------------------------
    # 4) 无学习策略 + Cloud（WORKERS_OR_CLOUD）
    #    这里才使用 LEAST_LOADED_AWARE_CLOUD，动作空间包含 cloud，合法。
    # -------------------------------------------------------
    policies_with_cloud = [
        Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,
        Node.NoLearningPolicy.MAXIMUM_LIFESPANE,
        Node.NoLearningPolicy.RANDOM,
    ]
    total_p_cloud = len(policies_with_cloud)
    for idx, policy in enumerate(policies_with_cloud, 1):
        label = f"No-learning-CLOUD policy={policy.name} ({idx}/{total_p_cloud})"
        timed(label, run_simulation_no_learning_cloud, policy)

    # -------------------------------------------------------
    # 5) 故障场景（alpha=1.0 和 0.0）
    # -------------------------------------------------------
    failure_alphas = [1.0, 0.0]
    for idx, alpha in enumerate(failure_alphas, 1):
        label = f"Failure alpha={alpha:.2f} ({idx}/{len(failure_alphas)})"
        timed(label, run_simulation_failure, alpha)

    # -------------------------------------------------------
    # 6) 汇总结果表
    # -------------------------------------------------------
    build_summary_table()

    # -------------------------------------------------------
    # 7) 自动绘制所有结果图
    # -------------------------------------------------------
    run_all_plots()


def build_summary_table():
    base_folder = f"{PATH_RESULTS_DATA}/_log/learning/D_SARSA/WORKERS_OR_CLOUD"
    alpha_range = [i / 100 for i in range(0, 101, 5)]  # 0.00, 0.05, ..., 1.00

    rows = []

    for alpha in alpha_range:
        folder_name = f"{base_folder}/{SESSION_ID}_{alpha:.2f}"
        print(f"[{datetime.now()}] scan learning folder: {folder_name}")
        if os.path.exists(folder_name):
            rows.append(append_row(folder_name, f"{alpha:.2f}"))

    base_folder_no = f"{PATH_RESULTS_DATA}/_log/no-learning"
    for policy in [
        Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD,
        Node.NoLearningPolicy.MAXIMUM_LIFESPANE,
        Node.NoLearningPolicy.RANDOM,
    ]:
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


def run_all_plots():
    """依次运行所有绘图脚本，输出图像到 results/plot/"""
    plot_scripts = [
        "plot_reward_over_time.py",
        "plot_batteries_and_fps.py",
        "plot_stacked_performances.py",
        "plot_actions_over_time.py",
        "plot_table_cloud.py",
        "plot_table_only_workers.py",
        "plot_tables.py",
    ]

    print(f"\n[{datetime.now()}] ==== START PLOTTING ====")
    for script in plot_scripts:
        print(f"[{datetime.now()}] Running: {script}")
        try:
            result = subprocess.run(
                ["python", script],
                capture_output=True,
                text=True,
                timeout=300,  # 单个脚本最多等 5 分钟
            )
            if result.returncode == 0:
                print(f"[{datetime.now()}] OK: {script}")
            else:
                print(f"[{datetime.now()}] WARN: {script} exited with code {result.returncode}")
                if result.stderr:
                    print(result.stderr[:500])
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now()}] TIMEOUT: {script} skipped")
        except Exception as e:
            print(f"[{datetime.now()}] ERROR: {script} -> {e}")

    print(f"[{datetime.now()}] ==== PLOTTING DONE ====")
    print(f"[{datetime.now()}] 图像保存在: {os.path.abspath(PATH_RESULTS_PLOT)}")


if __name__ == "__main__":
    print(f"[{datetime.now()}] ==== START ALL EXPERIMENTS (serial) ====")
    t0 = time.time()
    run_all_experiments_serial()
    dt = time.time() - t0
    print(f"[{datetime.now()}] ==== ALL DONE (serial) | total elapsed = {dt/3600:.2f} h ====")
