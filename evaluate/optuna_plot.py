import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna


def get_values(study_name, db_dir="outputs/optuna_storage"):
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_dir}/optuna.db")
    values = [trial.values for trial in study.trials]

    return np.array(values)


def get_all_values(data_model_name, seed_min, seed_max, db_dir):
    all_values = []
    for seed in range(seed_min, seed_max + 1):
        for cv in range(0, 10):
            study_name = f"{data_model_name}-{seed}_{cv}"
            all_values.append(get_values(study_name, db_dir))

    return np.concatenate(all_values, axis=0)


def get_k_best(values, alpha=1.0):
    k_best = -np.inf
    for trial in values:
        auc, inv_r = trial
        k = inv_r + alpha * auc
        if k > k_best:
            k_best = k

    return k_best


def save_scatter(values, data_model_name, output_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(values[:, 1], values[:, 0], alpha=0.6)
    for alpha in [1.0, 1.5, 2.0, 2.5, 3.0]:
        alpha_ = 2**alpha
        k_best = get_k_best(values, alpha_)
        auc = np.linspace(-10, 1.0, 10)
        inv_r = k_best - alpha_ * auc
        plt.plot(inv_r, auc, label=f"alpha={alpha:.1f}")

    plt.title(data_model_name)
    plt.xlabel(r"$1/N_{rules}$")
    plt.ylabel(r"$auc$")
    plt.xlim([0.0, 0.55])
    plt.ylim([0.45, 1.0])
    # plt.xscale("log")
    plt.legend()
    plt.grid()

    if output_path != Path("outputs/results.png"):
        output_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path)
    plt.close()


def main(args):
    data_model_name = args.data_model_name
    seed_min = args.seed_min
    seed_max = args.seed_max
    output_path = args.output_path

    values = get_all_values(data_model_name, seed_min, seed_max, args.db_dir)
    values = values[values[:, 1] != 1.0]
    save_scatter(values, data_model_name, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-model-name", "-d", default="aust-rerx", type=str)
    parser.add_argument("--db-dir", default=Path("outputs/optuna_storage"), type=Path)
    parser.add_argument("--seed-min", "-min", default=0, type=int)
    parser.add_argument("--seed-max", "-max", default=9, type=int)
    parser.add_argument("--output-path", default=Path("outputs/results.png"), type=Path)
    args = parser.parse_args()

    main(args)
