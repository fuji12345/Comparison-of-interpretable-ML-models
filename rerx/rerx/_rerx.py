from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from .base_model import BaseModel
from .rule import Rule, RuleExtractorFactory, RuleSet
from .tree import BaseTree
from .utils import Logger


class ReRx(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        base_model: BaseModel,
        tree: BaseTree,
        output_dim: int = 2,
        pruning_lamda: float = 0.01,
        pruning_step: float = 0.01,
        is_increasing_decision_score: bool = True,
        delta_1: float = 1.0,
        delta_2: float = 1.0,
        min_instance: int = 10,
        mode: str = "ori",
        is_pruning: bool = True,
        is_eval: bool = False,
        verbose=1,
        log_func="print",
    ) -> None:
        super().__init__()
        self.logger = Logger("ReRx", verbose, log_func=log_func)
        self.base_model = base_model
        self.tree = tree

        self.output_dim = output_dim

        self.pruning_lamda = pruning_lamda
        self.pruning_step = pruning_step
        self.is_increasing_decision_score = is_increasing_decision_score

        self.delta_1 = delta_1
        self.delta_2 = delta_2

        self.min_instance = min_instance

        self.mode = mode
        self.is_pruning = is_pruning
        self.is_eval = is_eval

    def _stop_pruning(self, original_score: float, pos_score: float):
        if self.is_increasing_decision_score:
            return self.pruning_lamda <= original_score - pos_score
        else:
            return self.pruning_lamda >= pos_score - original_score

    def _pruning_base_model(self, X: pd.DataFrame, y: np.ndarray):
        self.logger("Pruning base_model ...")
        original_score = self.base_model.get_current_score(X, y)
        original_weight = self.base_model.get_weights()

        previous_weight = None
        for ratio in np.arange(0, 1.0, self.pruning_step):
            self.base_model.set_weights(original_weight)

            self.base_model.pruning(ratio)

            pos_score = self.base_model.get_current_score(X, y)
            if self._stop_pruning(original_score, pos_score):
                self.logger("pruning_rate is : " + str(ratio))
                break
            previous_weight = self.base_model.get_weights()

        if ratio + self.pruning_step == 1.0:
            self.logger("Warning: pruning threshold is too high. The model may not perform properly.")

        if previous_weight is not None:
            self.base_model.set_weights(previous_weight)

    def subdivision(
        self,
        subdivision_columns: List[str],
        subdivision_mask: np.ndarray,
        rule: Rule,
    ):
        # node i に分類したデータのみを抽出
        X_next = self.X[subdivision_mask].reset_index(drop=True)
        y_next = self.y[subdivision_mask]

        # node i に達する際に使用しなかった属性情報のみを選択
        X_next = X_next[subdivision_columns].drop(rule.get_attrs_in_rule(), axis=1)
        self.logger(f"X_train_next : {X_next.shape}")
        self.logger(f"y_train_next : {y_next.shape}")
        if self.eval_set is not None:
            X_val = self.eval_set[0][list(X_next.columns)]
            eval_set = (X_val, self.eval_set[1])
            self.logger(f"X_val_next : {X_val.shape}")

        # 再帰呼び出し
        self.logger("Rules extraction is called in subdivisions.")
        sub_ruleset = self.get_ruleset_recursive(X_next, y_next, subdivision_mask, eval_set)
        return sub_ruleset

    def rule_extract(
        self,
        ruleset: RuleSet,
        stop_recursion: bool,
        subdivision_columns: List[str],
        subdivision_mask: Optional[np.ndarray] = None,
    ) -> RuleSet:
        self.logger("call rule_extract_recursive")
        new_ruleset = []
        new_condition_map = ruleset.condition_map

        if subdivision_mask is None:
            subdivision_mask = np.ones(self.X.shape[0], dtype=bool)
        for i, rule in enumerate(ruleset):
            r_pred, r_mask = rule.predict(self.X.values)
            mask = subdivision_mask * r_mask
            r_pred = r_pred.squeeze()[mask]
            r_y = self.y[mask]
            num_data_in_rule = mask.sum()
            num_wrong = (r_y != r_pred).sum()
            self.logger(f"rule index                        : {i}")
            self.logger(f"rule                              : {rule}")
            self.logger(f"num of data included in this rule : {num_data_in_rule}")
            self.logger(f"num of wrong data                 : {num_wrong}")

            support = mask.sum() / self.num_data
            self.logger(f"support : {support}")

            if num_data_in_rule == 0:
                error = np.nan
                self.logger("error    : NaN (no data in this rule)")
            else:
                error = num_wrong / num_data_in_rule
                self.logger(f"error   : {error}")

            if (
                not (support > self.delta_1 and error > self.delta_2)
                or stop_recursion
                or len(subdivision_columns) == len(rule.get_attrs_in_rule())
            ):
                new_ruleset.append(rule)
                self.logger("Add to new RuleSet.")
            else:
                sub_ruleset = self.subdivision(subdivision_columns, mask, rule)
                self.logger(f"sub_ruleset shape is : {len(sub_ruleset)}")
                self.logger(f"ruleset shape is     : {len(ruleset)}")

                if len(sub_ruleset) == 1:
                    new_ruleset.append(rule)
                    self.logger("Original rule added due to failure of rule generation in subdivision.")
                    continue

                self.logger("Successful rule generation in subdivision.")
                for sub_rule in sub_ruleset:
                    new_rule = deepcopy(rule)
                    new_rule.y = sub_rule.y
                    new_rule.class_dist = sub_rule.class_dist
                    new_rule.class_index = sub_rule.class_index
                    new_rule.add_conditions(sub_rule)

                    new_ruleset.append(new_rule)
                    new_condition_map.update(new_rule.get_condition_map())

                    self.logger(f"Rule : {new_rule}")

        self.logger("Finished Rule extraction.")
        self.logger(f"\tNum of rule in the previous RuleSet     : {len(ruleset)}")
        self.logger(f"\tNum of rule in the new RuleSet          : {len(new_ruleset)}")

        return RuleSet(new_ruleset, new_condition_map, classes=ruleset.classes)

    def _get_correctly_classified_data(
        self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        correct_idx = np.where(y == y_pred)[0]
        X = X.iloc[correct_idx]
        y = y[correct_idx]
        return X, y

    def get_label_map(self, y):
        """
        Create a dictionary that converts y labels to consecutive integers starting from 0.
        However, if the label is already a continuous integer starting from 0, return None.
        """
        label_map = {label: i for i, label in enumerate(np.unique(y))}
        if all([key == value for key, value in label_map.items()]):
            return None
        else:
            return label_map

    def map_y(self, y, label_map):
        mapped_y = np.zeros_like(y)
        for i, label in enumerate(label_map.keys()):
            mapped_y[y == label] = i

        return mapped_y

    def unmap_y(self, maped_y, label_map):
        unmapped_y = np.zeros_like(maped_y)
        for original_label, mapped_label in label_map.items():
            unmapped_y[maped_y == mapped_label] = original_label
        return unmapped_y

    def update_eval_set(self, eval_set: Tuple, label_map: Dict[int, int]):
        """
        Remove labels that are not in the label_map from the eval_set.
        Also, map the labels to consecutive integers starting from 0.
        """
        mask = np.isin(eval_set[1], list(label_map.keys()))
        return (eval_set[0][mask], self.map_y(eval_set[1][mask], label_map))

    def fit_base_model(self, X, y, eval_set):
        self.base_model.fit(X, y, eval_set)

        if self.is_eval:
            score = self.base_model.evaluate(X, y)
            self.logger.log_score(score, "Base model train score (before pruning)")
        if eval_set is not None and self.is_eval:
            val_score = self.base_model.evaluate(eval_set[0], eval_set[1])
            self.logger.log_score(val_score, "Base model val score (before pruning)")

    def get_single_class_ruleset(self, y) -> RuleSet:
        class_dist = np.zeros(self.output_dim)
        class_dist[y[0]] = 1
        classes = np.arange(self.output_dim)
        rule = Rule(
            conditions=frozenset([]),
            y=np.array([y[0]]),
            class_dist=class_dist,
            classes=classes,
        )
        return RuleSet(rules=[rule], classes=classes)

    def tree2ruleset(self, y):
        extractor = RuleExtractorFactory.get_rule_extractor(
            self.tree, self.feature_names_in_, np.arange(self.output_dim), None, y, 0
        )
        ruleset, _ = extractor.extract_rules()
        return ruleset

    def fit_tree(self, X, y, num_labels) -> Tuple[RuleSet, bool]:
        X_cate = X.select_dtypes(include=int)
        X_cont = X.select_dtypes(include=float)
        if not X_cate.empty:
            self.tree.fit(X_cate, y)
            ruleset_cate = self.tree2ruleset(y)
            if len(ruleset_cate.unique_label()) != num_labels and not X_cont.empty:
                self.logger("Decision tree could not generate rules for categorical sets.")
                self.tree.fit(X_cont, y)
                ruleset_cont = self.tree2ruleset(y)
                if len(ruleset_cont.unique_label()) != num_labels:
                    return ruleset_cate, True
                return ruleset_cont, True
            return ruleset_cate, False
        else:
            self.tree.fit(X_cont, y)
            return self.tree2ruleset(y), True

    def get_ruleset_recursive(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        subdivision_mask: Optional[np.ndarray] = None,
        eval_set: Optional[Tuple] = None,
    ) -> RuleSet:
        label_map = self.get_label_map(y)
        if label_map is not None:
            y = self.map_y(y, label_map)
            if eval_set is not None:
                eval_set = self.update_eval_set(eval_set, label_map)

        self.fit_base_model(X, y, eval_set)
        if self.is_pruning:
            y_pred = self.base_model.predict(X)
            if len(np.unique(y_pred)) == 1:
                self.logger.info(
                    "Before Pruning: The number of classes correctly predicted by the NN does not match the expected number of classes."
                )
                return self.get_single_class_ruleset(y_pred)

            self._pruning_base_model(X, y)

            if eval_set is not None and self.is_eval:
                val_score = self.base_model.evaluate(eval_set[0], eval_set[1])
                self.logger.log_score(val_score, "Base model val score (after pruning)")

        y_pred = self.base_model.predict(X)
        X_tree, y_tree = self._get_correctly_classified_data(X, y, y_pred)

        if label_map is not None:
            y_tree = self.unmap_y(y_tree, label_map)

        unique_label_tree = unique_labels(y_tree)

        if len(unique_label_tree) == 1:
            self.logger.info(
                "The number of classes correctly predicted by the NN does not match the expected number of classes."
            )
            return self.get_single_class_ruleset(y_tree)

        if self.is_pruning:
            droped_columns = self.base_model.get_droped_columns(X)
            X = X.drop(droped_columns, axis=1, inplace=False)
            X_tree = X_tree.drop(droped_columns, axis=1, inplace=False)
            self.logger(f"Deleted columns : {droped_columns}")
            self.logger(f"The shape of new X : {X.shape}")
            self.logger(f"The shape of X provided to Tree is : {X_tree.shape}")

        if len(X_tree) < self.min_instance:
            self.logger("The number of instances is less than the minimum number of instances.")
            # Return the ruleset that predicts the class with the largest number of occurrences
            single_label = np.argmax(np.bincount(y_tree))
            return self.get_single_class_ruleset([single_label])

        ruleset, stop_recursion = self.fit_tree(X_tree, y_tree, len(unique_label_tree))

        if len(ruleset) == 1:
            self.logger("Decision tree could not generate rules for both categorical and numeric sets.")
            return ruleset

        if stop_recursion:
            self.logger("Stop recursion in this loop because the categorical set is empty.")

        ruleset = self.rule_extract(ruleset, stop_recursion, list(X_tree.columns), subdivision_mask)
        return ruleset

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None):
        self.num_data = len(y)
        self.logger("Rules extraction is called in root.")
        self.X = X
        self.y = y
        self.eval_set = eval_set
        self.feature_names_in_ = list(X.columns)
        self.ruleset = self.get_ruleset_recursive(X, y, eval_set=eval_set)

    def predict(self, X):
        check_is_fitted(self, "ruleset")
        return self.ruleset.predict(X.values)

    def predict_proba(self, X):
        return np.eye(self.output_dim)[self.predict(X)]

    def evaluate(self, X, y) -> Dict[str, float]:
        check_is_fitted(self, "ruleset")
        results = {}
        y_pred = self.predict(X)
        n_rules, _, n_total_ant = self.ruleset.compute_interpretability_measures()

        results["Num of Rules"] = n_rules
        results["Ave. ante."] = n_total_ant / n_rules
        results["ACC"] = accuracy_score(y, y_pred)
        if self.output_dim == 2:
            results["AUC"] = roc_auc_score(y, y_pred)
            results["Precision"] = precision_score(y, y_pred)
            results["Recall"] = recall_score(y, y_pred)
            results["F1"] = f1_score(y, y_pred)
        return results


if __name__ == "__main__":
    check_estimator(ReRx())
