import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna


def get_values(study_name, db_dir="outputs/optuna_storage"):
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_dir}/optuna.db")
    values = [trial.values for trial in study.best_trials]

    return np.array(values)


def remove_same_points(a, b):
    """
    同じ点を削除する（ただし１つは残す）
    """
    n = len(a)
    mask = np.ones(n, dtype=bool)
    appeared = set()
    for i in range(n):
        if (a[i], b[i]) in appeared:
            mask[i] = False
        else:
            appeared.add((a[i], b[i]))

    return a[mask], b[mask]


def pareto_front(a, b):
    n = len(a)
    print(n)
    pareto_mask = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                if a[j] >= a[i] and b[j] >= b[i]:
                    pareto_mask[i] = False
                    break

    pareto_a = a[pareto_mask]
    pareto_b = b[pareto_mask]

    pf = np.array([pareto_a, pareto_b]).T
    pf = pf[pf[:, 0].argsort()]
    return pf


def save_pareto_front(method_values, title, output_path):
    plt.figure(figsize=(6, 6))
    font_size = 14

    # plt.title(title)
    for method, pf in method_values.items():
        plt.scatter(pf[:, 0], pf[:, 1], alpha=0.6)
        plt.plot(pf[:, 0], pf[:, 1], label=method)

    plt.xlabel(r"$1/N_{rules}$", fontsize=font_size)
    plt.ylabel(r"$auc$", fontsize=font_size)
    # plt.xlim([0.0, 0.55])
    # plt.ylim([0.45, 1.0])
    plt.xscale("log")
    plt.legend(loc="lower left", fontsize=font_size)
    plt.tight_layout()
    plt.grid()

    if output_path != Path("outputs/results.png"):
        output_path.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(output_path)
    plt.close()


def main(args):
    data = args.data
    methods = args.methods
    output_path = args.output_path
    seed = args.seed
    cv = args.cv

    method_values = {}
    for method in methods:
        data_model_name = f"{data}-{method}"
        study_name = f"{data_model_name}-{seed}_{cv}"
        values = get_values(study_name, args.db_dir)
        values = values[values[:, 1] != 1.0]
        inv_r, auc = values[:, 1], values[:, 0]
        inv_r, auc = remove_same_points(inv_r, auc)
        pf = pareto_front(inv_r, auc)
        print(f"{method}: {pf.shape}")
        if method == "rerx3":
            method = "Re-Rx with J48graft"
        elif method == "j48graft2":
            method = "J48graft"
        elif method == "dt":
            method = "DT"
        elif method == "fbts":
            method = "FBT"
        elif method == "rulecosi":
            method = "RuleCOSI+"
        method_values[method] = pf

    save_pareto_front(method_values, f"Pareto front in {data} data.", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument(
        "--methods",
        "-m",
        type=str,
        nargs="+",
        help="List of methods to evaluate",
        default=["fbts", "rulecosi", "rerx3", "j48graft2", "dt"],
    )
    parser.add_argument("--db-dir", "-d", default=Path("outputs/optuna_storage"), type=Path)
    parser.add_argument("--seed", "-s", default=0, type=int)
    parser.add_argument("--cv", "-c", default=0, type=int)
    parser.add_argument("--output-path", "-o", default=Path("outputs/results.png"), type=Path)
    args = parser.parse_args()

    main(args)
