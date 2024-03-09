from copy import deepcopy
from typing import List

import numpy as np
import xgboost as xgb
from fbts import FBT
from rerx import MLP, J48graft, ReRx
from rerx.rule import RuleExtractorFactory, RuleSet
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y

import rulecosi
import rulecosi.rule_extraction as rulecosi_rule_extraction

from .utils import FBTsRuleExtractor, crep, set_seed


def compute_emsemble_interpretability_measures(rulesets: List[RuleSet]):
    num_rules, condition_maps, n_total_ants = 0, {}, 0
    for ruleset in rulesets:
        num_rule, _, n_total_ant = ruleset.compute_interpretability_measures()
        num_rules += num_rule
        condition_maps.update(ruleset.condition_map)
        n_total_ants += n_total_ant
    return num_rules, len(condition_maps), n_total_ants


class BaseClassifier:
    def __init__(
        self, input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study=False, pre_model=None
    ) -> None:
        self.ruleset: RuleSet = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = model_config
        _, counts = np.unique(init_y, return_counts=True)
        self.class_ratio = counts.min() / counts.max()
        self.classes_ = unique_labels(init_y)
        self.onehoter = onehoter
        self.verbose = verbose
        self.pre_study = pre_study
        self.pre_model = pre_model

    def pre_fit(self, X, y, eval_set):
        raise NotImplementedError()

    def fit(self, X, y, eval_set):
        raise NotImplementedError()

    def predict_proba_pre(self, X):
        return self.pre_model.predict_proba(X.values)

    def predict_pre(self, X):
        return self.pre_model.predict(X.values)

    def predict_proba(self, X):
        return self.ruleset.predict_proba(X.values)

    def predict(self, X):
        return self.ruleset.predict(X.values)

    def compute_interpretability_measures(self):
        return self.ruleset.compute_interpretability_measures()

    def pre_evaluate(self, X, y):
        y_pred = self.predict_pre(X)
        y_score = self.predict_proba_pre(X)
        results = {}
        results["CREP"] = crep
        results["ACC"] = accuracy_score(y, y_pred)
        if self.output_dim == 2:
            results["AUC"] = roc_auc_score(y, y_score[:, 1])
            results["Precision"] = precision_score(y, y_pred, zero_division=0)
            results["Recall"] = recall_score(y, y_pred)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred)
            results["F1"] = f1_score(y, y_pred, zero_division=0)
        else:
            results["AUC"] = roc_auc_score(y, y_score, multi_class="ovr")
            results["Precision"] = precision_score(y, y_pred, average="macro", zero_division=0)
            results["Recall"] = recall_score(y, y_pred, average="macro", zero_division=0)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="macro", zero_division=0)
            results["F1"] = f1_score(y, y_pred, average="macro", zero_division=0)
        return results

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        n_rules, _, n_total_ant = self.compute_interpretability_measures()
        crep = self.compute_crep(X)
        results = {}
        results["Num of Rules"] = n_rules
        results["Ave. ante."] = n_total_ant / n_rules
        results["CREP"] = crep
        results["ACC"] = accuracy_score(y, y_pred)
        if self.output_dim == 2:
            results["AUC"] = roc_auc_score(y, y_pred)
            results["Precision"] = precision_score(y, y_pred, zero_division=0)
            results["Recall"] = recall_score(y, y_pred)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred)
            results["F1"] = f1_score(y, y_pred, zero_division=0)
        else:
            y_score = self.predict_proba(X)
            results["AUC"] = roc_auc_score(y, y_score, multi_class="ovr")
            results["Precision"] = precision_score(y, y_pred, average="macro", zero_division=0)
            results["Recall"] = recall_score(y, y_pred, average="macro", zero_division=0)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred, average="macro", zero_division=0)
            results["F1"] = f1_score(y, y_pred, average="macro", zero_division=0)
        return results

    def compute_crep(self, X):
        crep_result = crep(self.ruleset, X.values)
        return crep_result


class ReRxClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        rerx_config = self.model_config.rerx
        mlp_config = self.model_config.mlp
        tree_config = self.model_config.tree
        self.mlp = MLP(
            epochs=200,
            early_stop=True,
            use_output_bias=True,
            verbose=self.verbose,
            onehoter=self.onehoter,
            **mlp_config,
        )
        tree = J48graft(**tree_config, verbose=self.verbose)
        self.rerx = ReRx(
            base_model=self.mlp,
            tree=tree,
            output_dim=self.output_dim,
            is_eval=self.verbose > 0,
            verbose=self.verbose,
            **rerx_config,
        )

    def pre_fit(self, X, y, eval_set):
        label_map = self.rerx.get_label_map(y)
        if label_map is not None:
            y = self.rerx.map_y(y, label_map)
            if eval_set is not None:
                eval_set = self.rerx.update_eval_set(eval_set, label_map)
        self.mlp.fit(X, y, eval_set=eval_set)
        self.pre_model = self.mlp

    def predict_pre(self, X):
        return self.pre_model.predict(X)

    def predict_proba_pre(self, X):
        return self.pre_model.predict_proba(X)

    def fit(self, X, y, eval_set):
        self.rerx.fit(X, y, eval_set=eval_set)
        self.ruleset = self.rerx.ruleset


class J48graftClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tree = J48graft(**self.model_config, verbose=self.verbose)

    def fit(self, X, y, eval_set):
        self.tree.fit(X, y, eval_set=eval_set)
        extractor = RuleExtractorFactory.get_rule_extractor(
            self.tree, X.columns.to_list(), unique_labels(y), None, y, 0
        )
        self.ruleset, _ = extractor.extract_rules()


class DTClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tree = DecisionTreeClassifier(**self.model_config)

    def fit(self, X, y, eval_set):
        self.tree.fit(X, y)
        extractor = RuleExtractorFactory.get_rule_extractor(
            self.tree, X.columns.to_list(), unique_labels(y), None, y, 0
        )
        self.ruleset, _ = extractor.extract_rules()


class FBTsClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xgb_config = self.model_config.ensemble
        fbts_config = self.model_config.fbts

        if self.output_dim == 2:
            xgb_config["objective"] = "binary:logitraw"
        else:
            xgb_config["objective"] = "multi:softmax"

        if self.pre_model is not None:
            self.ens = self.pre_model
        else:
            self.ens = xgb.XGBClassifier(
                **xgb_config,
                num_class=self.output_dim if self.output_dim > 2 else None,
                eval_metric="auc",
                early_stopping_rounds=10,
        )
        self.fbts = FBT(**fbts_config, verbose=self.verbose)

    def pre_fit(self, X, y, eval_set):
        self.pre_model = self.ens.fit(X, y, eval_set=[eval_set], verbose=False)

    def fit(self, X, y, eval_set):
        if self.pre_model is None:
            self.ens.fit(X, y, eval_set=[eval_set], verbose=False)

        data = deepcopy(X)
        columns = list(data.columns)
        data["class"] = y
        self.fbts.fit(data, columns, "class", self.ens)
        extractor = FBTsRuleExtractor(self.fbts, columns, unique_labels(y), None, y, 0)
        self.ruleset, _ = extractor.extract_rules()


class RuleCOSIClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        xgb_config = self.model_config.ensemble
        rulecosi_config = self.model_config.rulecosi

        if self.output_dim == 2:
            xgb_config["objective"] = "binary:logitraw"
        else:
            xgb_config["objective"] = "multi:softmax"

        if self.pre_model is not None:
            self.ens = self.pre_model
        else:
            self.ens = xgb.XGBClassifier(
                **xgb_config,
                num_class=self.output_dim if self.output_dim > 2 else None,
                eval_metric="auc",
                early_stopping_rounds=10,
            )
        self.rulecosi = rulecosi.RuleCOSIClassifier(
            metric="auc",
            **rulecosi_config,
            verbose=self.verbose,
        )

    def pre_fit(self, X, y, eval_set):
        X_xgb, y_xgb = check_X_y(X, y)
        eval_set_xgb = check_X_y(*eval_set)
        self.pre_model = self.ens.fit(X_xgb, y_xgb, eval_set=[eval_set_xgb], verbose=False)

    def fit(self, X, y, eval_set):
        if self.pre_model is None:
            X_xgb, y_xgb = check_X_y(X, y)
            eval_set_xgb = check_X_y(*eval_set)
            self.ens.fit(X_xgb, y_xgb, eval_set=[eval_set_xgb], verbose=False)

        self.rulecosi.fit(X, y)
        self.ruleset = self.rulecosi.simplified_ruleset_

    def predict_proba(self, X):
        return softmax(self.ruleset.predict_proba(X.values), axis=1)

    def compute_crep(self, X):
        crep_result = crep(self.ruleset, X.values, is_decision_list=True)
        return crep_result


class XGBoostClassifier(BaseClassifier):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if self.output_dim == 2:
            self.model_config["objective"] = "binary:logitraw"
        else:
            self.model_config["objective"] = "multi:softmax"

        self.xgb = xgb.XGBClassifier(
            **self.model_config,
            num_class=self.output_dim if self.output_dim > 2 else None,
            eval_metric="auc",
            early_stopping_rounds=10,
        )

    def fit(self, X, y, eval_set):
        self._column_names = X.columns
        X, y = check_X_y(X, y)
        eval_set = check_X_y(*eval_set)

        self.xgb.fit(X, y, eval_set=[eval_set], verbose=False)
        extractor = rulecosi_rule_extraction.RuleExtractorFactory.get_rule_extractor(
            self.xgb,
            self._column_names,
            self.classes_,
            X,
            y,
            1e-6,
        )
        self.rulesets, _ = extractor.extract_rules()

    def compute_crep(self, X):
        crep_sum = 0
        for ruleset in self.rulesets:
            crep_sum += crep(ruleset, X.values)

        return crep_sum

    def predict_proba(self, X):
        return self.xgb.predict_proba(X)

    def predict(self, X):
        return self.xgb.predict(X)

    def compute_interpretability_measures(self):
        return compute_emsemble_interpretability_measures(self.rulesets)


def get_classifier(
    name,
    *,
    input_dim,
    output_dim,
    model_config,
    init_y,
    onehoter,
    pre_study=False,
    pre_model=None,
    seed=42,
    verbose=0,
):
    set_seed(seed=seed)
    if name == "rerx":
        return ReRxClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study)
    elif name == "rulecosi":
        return RuleCOSIClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study, pre_model)
    elif name == "fbts":
        return FBTsClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose, pre_study, pre_model)
    elif name == "xgboost":
        return XGBoostClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    elif name == "j48graft":
        return J48graftClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    elif name == "dt":
        return DTClassifier(input_dim, output_dim, model_config, init_y, onehoter, verbose)
    else:
        raise KeyError(f"{name} is not defined.")
