"""
This module contains several functions that are used in various stages of the process
"""
import json
import operator as op
import os
import pickle
import random
from typing import Dict, Union

import numpy as np
import tensorflow as tf
from fbts import FBT
from rerx.rule import RuleSet
from rerx.rule.rule_extraction import DecisionTreeRuleExtractor
from sklearn.metrics import accuracy_score, roc_auc_score

from dataset.utils import feature_name_restorer

RANDOM_SEED = 1


def set_seed(seed: int = 42):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_json(data: Dict[str, Union[int, float, str]], save_dir: str = "./"):
    with open(os.path.join(save_dir, "results.json"), mode="wt", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path) -> Dict[str, Union[int, float, str]]:
    with open(path, mode="rt", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_object(obj, output_path: str):
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def save_ruleset(ruleset: RuleSet, save_dir: str, file_name: str):
    os.makedirs(save_dir, exist_ok=True)
    save_object(ruleset, os.path.join(save_dir, file_name) + ".pkl")
    with open(os.path.join(save_dir, file_name) + ".txt", "w") as f:
        f.write(feature_name_restorer(str(ruleset)))


def cal_auc_score(model, data, feature_cols, label_col):
    pred_proba = model.predict_proba(data[feature_cols])
    if data[label_col].nunique() == 2:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba[:, 1])
    else:
        auc = roc_auc_score(data[label_col].values.tolist(), pred_proba, multi_class="ovr")
    return auc


def cal_acc_score(model, data, feature_cols, label_col):
    pred = model.predict(data[feature_cols])
    acc = accuracy_score(data[label_col], pred)
    return acc


def cal_metrics(model, data, feature_cols, label_col):
    acc = cal_acc_score(model, data, feature_cols, label_col)
    auc = cal_auc_score(model, data, feature_cols, label_col)
    return {"ACC": acc, "AUC": auc}


def crep(ruleset: RuleSet, X, is_decision_list=False):
    crep = 0
    conditions = 0
    covered_mask = np.zeros((X.shape[0],), dtype=bool)  # records the records that are
    for rule in ruleset:
        _, r_mask = rule.predict(X)
        # update the covered_mask with the records covered by this rule
        remaining_covered_mask = ~covered_mask & r_mask
        p_rule = remaining_covered_mask.sum() / len(X)
        if is_decision_list:
            conditions += len(rule.A)
        else:
            conditions = len(rule.A)
        crep += p_rule * conditions
        covered_mask = covered_mask | r_mask

    return crep


def set_categories_in_rule(ruleset, categories_dict):
    ruleset.set_categories(categories_dict)


def softmax(x):
    return np.array([np.exp(x) / np.sum(np.exp(x))])


class FBTsRuleExtractor(DecisionTreeRuleExtractor):
    class TMPClass:
        feature_names_in_ = None

    def __init__(self, fbts: FBT, _column_names, classes_, X, y, float_threshold):
        _tree = self.TMPClass()
        _tree.feature_names_in_ = _column_names
        super().__init__(_tree, _column_names, classes_, X, y, float_threshold)
        self.fbts = fbts

    def get_tree_dict(self, base_tree, n_nodes=0):
        self.node_idx = -1
        keys = [
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "n_samples",
            "n_nodes",
        ]
        self.tree_dict = {k: [] for k in keys}
        self._convert(self.fbts.tree)
        self.tree_dict["n_nodes"] = self.node_idx
        return self.tree_dict

    def get_split_operators(self):
        op_left = op.ge
        op_right = op.lt
        return op_left, op_right

    def _convert(self, tree):
        self.node_idx += 1
        feature = tree.selected_feature

        if feature is None:
            self.tree_dict["children_left"].append(-1)
            self.tree_dict["children_right"].append(-1)
            self.tree_dict["feature"].append(-2)
            self.tree_dict["threshold"].append(-2)
            self.tree_dict["value"].append(np.array([softmax(c.label_probas) for c in tree.conjunctions]).mean(0))
            self.tree_dict["n_samples"].append(None)
        else:
            this_node = self.node_idx
            right = tree.right
            left = tree.left
            threshold = tree.selected_value
            self.tree_dict["feature"].append(feature)
            self.tree_dict["threshold"].append(threshold)
            self.tree_dict["value"].append(None)
            self.tree_dict["n_samples"].append(None)
            self.tree_dict["children_left"].append(self.node_idx + 1)
            self.tree_dict["children_right"].append(None)
            self._convert(left)
            self.tree_dict["children_right"][this_node] = self.node_idx + 1
            self._convert(right)
