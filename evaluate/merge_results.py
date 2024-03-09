import sys

from scipy import stats

sys.path.append("./")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.utils import load_json


def calculate_metrics_summary(directory_path, file_name):
    # Get a list of all results.json files in the specified directory using pathlib and glob
    json_files = list(directory_path.glob(f"*/{file_name}"))
    if len(json_files) < 10:
        print(f"results.json files are less than 10 in {directory_path}. {len(json_files)} files found.")

    if not json_files:
        print("No results.json files found in the directory.")
        return

    # Read the first results.json file to obtain metric names
    first_json_file = json_files[0]
    first_data = load_json(str(first_json_file))
    metric_names = list(first_data.keys())

    # Initialize dictionaries to store metric sums, averages, and variances
    metrics_all = {metric: [] for metric in metric_names}

    for json_file in json_files:
        # Load each results.json file
        data = load_json(json_file)

        # Add metrics to the sums dictionary
        for metric, values in data.items():
            metrics_all[metric].extend(values)
        if len(data[metric]) < 10:
            print(f"{json_file} has less than 10 metrics.")

    # Calculate the average and variance of each metric
    for metric, values in metrics_all.items():
        metrics_all[metric] = np.array(values)

    return metrics_all


# metric に対する method のランキングのdictを生成
# ただし、同じ値の場合は同じ順位とする
def get_ranking(scores, less_is_better):
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=less_is_better)
    ranking = {}
    rank = 1
    prev_score = None
    for method, score in sorted_scores:
        if prev_score is None:
            ranking[method] = rank
        elif prev_score != score:
            rank += 1
            ranking[method] = rank
        else:
            ranking[method] = rank
        prev_score = score
    return ranking


def get_results(methods, directory_path, file_name, median=False):
    metric_method_scores = None
    for method in methods:
        method_path = directory_path / method
        # Calculate the average and variance of metrics
        scores_dict = calculate_metrics_summary(method_path, file_name)
        if metric_method_scores is None:
            for metric, scores in scores_dict.items():
                metric_method_scores = {metric: {method: scores} for metric, scores in scores_dict.items()}
        else:
            for metric, scores in scores_dict.items():
                metric_method_scores[metric][method] = scores
        scores_data = {}

        # Output the results
        for metric, scores in scores_dict.items():
            if metric not in ["Num of Rules", "Ave. ante.", "Time"]:
                scores = scores * 100
            # print(f"{metric}: ${scores.mean():.1f} \pm {scores.std():.1f}$")
            if median:
                scores_data[metric] = [f"${np.median(scores):.1f}$"]
            else:
                scores_data[metric] = [f"${scores.mean():.1f} \pm {scores.std():.1f}$"]

        # scores to dataframe
        # print(pd.DataFrame(scores_data).to_markdown())
    results = {}
    ranking = {}
    for metric, method_scores in metric_method_scores.items():
        if metric not in ["Num of Rules", "Ave. ante.", "CREP", "Time"]:
            less_is_better = True
        else:
            less_is_better = False
        mean_score = {k: score.mean() for k, score in method_scores.items()}
        best_method = max(mean_score, key=mean_score.get) if less_is_better else min(mean_score, key=mean_score.get)
        ranking[metric] = get_ranking(mean_score, less_is_better=less_is_better)

        scores_data = {}
        for method, scores in method_scores.items():
            if method == best_method:
                if metric not in ["Num of Rules", "Ave. ante.", "CREP", "Time"]:
                    scores = scores * 100
                if median:
                    scores_data[method] = "$\mathbf{" + f"{np.median(scores):.1f}" + "}$"
                else:
                    scores_data[method] = (
                        "$\mathbf{" + f"{scores.mean():.1f}" + "}" + " \pm \mathbf{" + f"{scores.std():.1f}" + "}$"
                    )
            else:
                # p > 0.05 means no significant difference.
                p_value = stats.ttest_ind(method_scores[best_method], scores, equal_var=False).pvalue
                p_nan = np.isnan(p_value) and scores.mean() == method_scores[best_method].mean()
                if metric not in ["Num of Rules", "Ave. ante.", "CREP", "Time"]:
                    scores = scores * 100
                if p_nan or p_value > 0.05:
                    if median:
                        scores_data[method] = "$\mathbf{" + f"{np.median(scores):.1f}" + "}$"
                    else:
                        scores_data[method] = (
                            "$\mathbf{" + f"{scores.mean():.1f}" + "}" + " \pm \mathbf{" + f"{scores.std():.1f}" + "}$"
                        )
                else:
                    if median:
                        scores_data[method] = f"${np.median(scores):.1f}$"
                    else:
                        scores_data[method] = f"${scores.mean():.1f} \pm {scores.std():.1f}$"
        results[metric] = scores_data
    return results, ranking, metric_method_scores


def merge_results(results_data, ranking_data, methods):
    merged_metrics_result = {}
    merged_metrics_ranking = {}
    for data in results_data.keys():
        for metric in results_data[data].keys():
            if metric not in merged_metrics_result:
                merged_metrics_result[metric] = {}
                merged_metrics_ranking[metric] = {}
            merged_metrics_result[metric][data] = results_data[data][metric]
            merged_metrics_ranking[metric][data] = ranking_data[data][metric]

    # ranking の平均を計算
    mean_ranking = {}
    for metric, ranking in merged_metrics_ranking.items():
        mean_ranking[metric] = {}
        for method in methods:
            mean_ranking[metric][method] = np.mean([ranking[data][method] for data in ranking.keys()])

    return merged_metrics_result, mean_ranking


def merge_data(metric_method_score_data, methods, median=False):
    metric_method_scores = {}
    for data in metric_method_score_data.keys():
        for metric in metric_method_score_data[data].keys():
            for method in methods:
                if metric not in metric_method_scores:
                    metric_method_scores[metric] = {}
                if method not in metric_method_scores[metric]:
                    metric_method_scores[metric][method] = []
                metric_method_scores[metric][method].extend(metric_method_score_data[data][metric][method])

    # 有意差があるかどうかを判定する
    for metric, method_scores in metric_method_scores.items():
        # top method を求める
        if metric not in ["Num of Rules", "Ave. ante.", "CREP", "Time"]:
            less_is_better = True
        else:
            less_is_better = False
        mean_score = {k: np.array(score).mean() for k, score in method_scores.items()}
        best_method = max(mean_score, key=mean_score.get) if less_is_better else min(mean_score, key=mean_score.get)
        best_score = np.array(method_scores[best_method])
        for method, scores in method_scores.items():
            scores = np.array(scores)
            if method == best_method:
                if metric not in ["Num of Rules", "Ave. ante.", "CREP", "Time"]:
                    scores = scores * 100
                if median:
                    metric_method_scores[metric][method] = "$\mathbf{" + f"{np.median(scores):.1f}" + "}$"
                else:
                    metric_method_scores[metric][method] = (
                        "$\mathbf{" + f"{scores.mean():.1f}" + "}" + " \pm \mathbf{" + f"{scores.std():.1f}" + "}$"
                    )
            else:
                # p > 0.05 means no significant difference.
                p_value = stats.ttest_ind(best_score, scores, equal_var=False).pvalue
                p_nan = np.isnan(p_value) and scores.mean() == method_scores[best_method].mean()
                if metric not in ["Num of Rules", "Ave. ante.", "CREP", "Time"]:
                    scores = scores * 100
                if p_nan or p_value > 0.05:
                    if median:
                        metric_method_scores[metric][method] = "$\mathbf{" + f"{np.median(scores):.1f}" + "}$"
                    else:
                        metric_method_scores[metric][method] = (
                            "$\mathbf{" + f"{scores.mean():.1f}" + "}" + " \pm \mathbf{" + f"{scores.std():.1f}" + "}$"
                        )
                else:
                    if median:
                        metric_method_scores[metric][method] = f"${np.median(scores):.1f}$"
                    else:
                        metric_method_scores[metric][method] = f"${scores.mean():.1f} \pm {scores.std():.1f}$"

    return metric_method_scores


# 各 metrics について、ranking を含む 表を出力する
def show_results(results, ranking, methods, datasets):
    for metric in results.keys():
        print(f"### {metric}")
        df = pd.DataFrame(results[metric], index=methods)
        df = df[datasets]
        df["ranking"] = [f"{ranking[metric][method]:.1f}" for method in methods]
        df = df.T
        print(df.to_latex())


# 横軸が metrics, 縦軸が methods の表を出力する
def show_summary(metric_method_scores, methods, metrics=["AUC", "Num of Rules", "CREP"]):
    df = pd.DataFrame(metric_method_scores, index=methods)
    df = df[metrics]
    print(df.to_latex())


def main(args):
    # Get the directory path from the command-line arguments
    directory_path = args.directory_path
    file_name = args.file_name
    methods = args.methods
    datasets = args.datasets
    results_data = {}
    ranking_data = {}
    metric_method_score_data = {}
    for dataset in datasets:
        print(dataset)
        dataset_path = directory_path / dataset
        results, ranking, metric_method_scores = get_results(methods, dataset_path, file_name, median=args.median)
        dataset = f"\{dataset}".replace("-", "")
        results_data[dataset] = results
        ranking_data[dataset] = ranking
        metric_method_score_data[dataset] = metric_method_scores
    results, ranking = merge_results(results_data, ranking_data, methods)
    metric_method_scores = merge_data(metric_method_score_data, methods, median=args.median)

    datasets = [f"\{dataset}".replace("-", "") for dataset in datasets]
    show_results(results, ranking, methods, datasets)
    show_summary(metric_method_scores, methods)


if __name__ == "__main__":
    # Set the directory path using pathlib
    parser = argparse.ArgumentParser(description="Calculate average and variance of metrics from results.json files")
    parser.add_argument("datasets", type=str, nargs="+", help="List of datasets to evaluate")
    parser.add_argument("--directory_path", "-d", type=Path, help="Path to the directory", default=Path("./outputs"))
    parser.add_argument(
        "--methods",
        "-m",
        type=str,
        nargs="+",
        help="List of methods to evaluate",
        default=["fbts", "rulecosi", "rerx3", "j48graft", "dt"],
    )
    parser.add_argument("--median", "-md", action="store_true", help="Output median instead of mean")
    parser.add_argument("--file-name", "-f", type=str, help="results.json file name", default="results.json")

    # Parse the command-line arguments
    args = parser.parse_args()
    main(args)
