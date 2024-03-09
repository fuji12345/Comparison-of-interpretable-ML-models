import logging
import os
from copy import deepcopy
from statistics import mean

import numpy as np
import optuna
from hydra.utils import to_absolute_path
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import StratifiedKFold

from .classifier import get_classifier

logger = logging.getLogger(__name__)


def xgboost_config(trial: optuna.Trial, model_config, name=""):
    model_config.max_depth = trial.suggest_int(f"{name}max_depth", 1, 10)
    model_config.eta = trial.suggest_float(f"{name}eta", 1e-4, 1.0, log=True)
    model_config.n_estimators = 250
    return model_config


def j48graft_config(trial: optuna.Trial, model_config, name=""):
    min_instance_power = trial.suggest_int(f"{name}min_instance_power", 1, 8)
    model_config.min_instance = 2**min_instance_power
    model_config.pruning_conf = trial.suggest_float(f"{name}pruning_conf", 0.1, 0.5)
    return model_config


def rerx_config(trial: optuna.Trial, model_config, pre_study=False):
    if pre_study:
        model_config.mlp.h_dim = trial.suggest_int("mlp.h_dim", 1, 5)
        model_config.mlp.lr = trial.suggest_float("mlp.lr", 5e-3, 0.1, log=True)
        model_config.mlp.lr = trial.suggest_float("mlp.weight_decay", 1e-6, 1e-2, log=True)
    else:
        model_config.tree = j48graft_config(trial, model_config.tree, name="tree.")
        model_config.rerx.pruning_lamda = trial.suggest_float("rerx.pruning_lamda", 0.001, 0.25, log=True)
        model_config.rerx.delta_1 = trial.suggest_float("rerx.delta_1", 0.05, 0.4)
        model_config.rerx.delta_2 = trial.suggest_float("rerx.delta_2", 0.05, 0.4)
    return model_config


def rulecosi_config(trial: optuna.Trial, model_config, pre_study=False):
    if pre_study:
        model_config.ensemble = xgboost_config(trial, model_config.ensemble, name="ensemble.")
    else:
        model_config.rulecosi.conf_threshold = trial.suggest_float("rulecosi.conf_threshold", 0.0, 0.95)
        model_config.rulecosi.cov_threshold = trial.suggest_float("rulecosi.cov_threshold", 0.0, 0.5)
        model_config.rulecosi.c = trial.suggest_float("rulecosi.c", 0.1, 0.5)
    return model_config


def fbts_config(trial: optuna.Trial, model_config, pre_study=False):
    if pre_study:
        model_config.ensemble = xgboost_config(trial, model_config.ensemble, name="ensemble.")
    else:
        model_config.fbts.max_depth = trial.suggest_int("fbts.max_depth", 1, 10)
        model_config.fbts.pruning_method = trial.suggest_categorical("fbts.pruning_method", [None, "auc"])
    return model_config


def dt_config(trial: optuna.Trial, model_config):
    model_config.max_depth = trial.suggest_int("max_depth", 1, 10)
    model_config.min_samples_split = trial.suggest_float("min_samples_split", 0.0, 0.5)
    model_config.min_samples_leaf = trial.suggest_float("min_samples_leaf", 0.0, 0.5)
    return model_config


def get_model_config(model_name):
    if model_name == "rerx":
        return rerx_config
    elif model_name == "rulecosi":
        return rulecosi_config
    elif model_name == "fbts":
        return fbts_config
    elif model_name == "xgboost":
        return xgboost_config
    elif model_name == "j48graft":
        return j48graft_config
    elif model_name == "dt":
        return dt_config
    else:
        raise ValueError()


def update_model_cofig(default_config, best_config):
    for _p, v in best_config.items():
        current_dict = default_config
        _p = _p.split(".")
        for p in _p[:-1]:
            if p not in current_dict:
                current_dict[p] = {}
            current_dict = current_dict[p]
        last_key = _p[-1]
        current_dict[last_key] = v


class OptimParam:
    def __init__(
        self,
        model_name,
        default_config,
        input_dim,
        output_dim,
        X,
        y,
        val_data,
        columns,
        target_column,
        onehoter,
        n_trials,
        n_startup_trials,
        storage,
        study_name,
        cv=True,
        seed=42,
        alpha=1,
    ) -> None:
        self.model_name = model_name
        self.default_config = deepcopy(default_config)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_config = get_model_config(model_name)
        self.X = X
        self.y = y
        self.val_data = val_data
        self.columns = columns
        self.target_column = target_column
        self.onehoter = onehoter
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.storage = to_absolute_path(storage) if storage is not None else None
        self.study_name = study_name
        self.cv = cv
        self.seed = seed
        self.alpha = alpha
        self.pre_study = False
        if "rerx" in self.model_name or "rulecosi" in self.model_name or "fbts" in self.model_name:
            self.pre_study = True
            self.n_trials = self.n_trials // 2

    def fit(self, model_config, X_train, y_train, X_val=None, y_val=None):
        if X_val is None and y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()

        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            init_y=self.y,
            onehoter=self.onehoter,
            seed=self.seed,
            pre_study=self.pre_study,
            pre_model=deepcopy(self.pre_model) if hasattr(self, "pre_model") else None,
        )
        fit = model.pre_fit if self.pre_study else model.fit
        fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )
        evaluate = model.pre_evaluate if self.pre_study else model.evaluate
        score = evaluate(
            self.val_data[self.columns],
            self.val_data[self.target_column].values.squeeze(),
        )
        return score

    def fit_pre_model(self, model_config, X_train, y_train, X_val=None, y_val=None):
        logger.info("Fitting pre model...")
        if X_val is None and y_val is None:
            X_val = self.val_data[self.columns]
            y_val = self.val_data[self.target_column].values.squeeze()

        model = get_classifier(
            self.model_name,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            model_config=model_config,
            init_y=self.y,
            onehoter=self.onehoter,
            seed=self.seed,
            pre_study=True,
        )
        model.pre_fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
        )
        self.pre_model = model.pre_model

    def cross_validation(self, model_config):
        skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
        ave_auc = []
        ave_inv_rules = []
        for _, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train, y_train = self.X.iloc[train_idx], self.y[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y[val_idx]
            score = self.fit(model_config, X_train, y_train, X_val, y_val)
            ave_auc.append(score["AUC"])
            ave_inv_rules.append(1 / score["Num of Rules"])
        return mean(ave_auc), mean(ave_inv_rules)

    def one_shot(self, model_config):
        score = self.fit(model_config, self.X, self.y)
        return score["AUC"], 1 / score["Num of Rules"]

    def pre_objective(self, trial):
        _model_config = self.model_config(trial, deepcopy(self.default_config), pre_study=True)
        score = self.fit(_model_config, self.X, self.y)
        return score["AUC"]

    def objective(self, trial):
        _model_config = self.model_config(trial, deepcopy(self.default_config))
        if self.cv:
            auc, inv_rules = self.cross_validation(_model_config)
        else:
            auc, inv_rules = self.one_shot(_model_config)
        return auc, inv_rules

    def _get_best_params(self, study: optuna.Study):
        best_trials = study.best_trials
        best_trials = [trial for trial in best_trials if trial.values[0] != 0.5]
        if len(best_trials) == 0:
            best_trials = [study.best_trials[0]]
        best_trial = best_trials[0]
        k_best = -np.inf
        for trial in best_trials:
            x, y = trial.values
            k = np.log2(y) + 100 * self.alpha * x
            # print(f"{k:.3f}, {y:.3f}, {np.log2(y):.1f}, {x:.3f}")
            if k > k_best:
                k_best = k
                best_trial = trial
        logger.info(f"Accepted trial: {best_trial.number}")
        logger.info(f"AUC: {best_trial.values[0]}, 1/Rules: {best_trial.values[1]}, Rules: {1/best_trial.values[1]}")
        logger.info(f"Parameters: {best_trial.params}")
        return best_trial.params

    def get_n_complete(self, study: optuna.Study):
        n_complete = len([trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
        return n_complete

    def get_best_config(self):
        if self.storage is not None:
            os.makedirs(self.storage, exist_ok=True)
            self.storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{self.storage}/optuna.db",
            )
        if "rerx" in self.model_name or "rulecosi" in self.model_name or "fbts" in self.model_name:
            study_name = f"{self.study_name}_pre"
            if "fbts" in study_name or "rulecosi" in study_name:
                study_name = study_name.replace(self.model_name, "xgboost")
            elif "rerx3" in study_name:
                study_name = study_name.replace("rerx3", "rerx")

            pre_study = optuna.create_study(
                storage=self.storage,
                study_name=study_name,
                direction="maximize",
                sampler=optuna.samplers.TPESampler(
                    seed=self.seed,
                    n_startup_trials=self.n_startup_trials,
                ),
                load_if_exists=True,
            )
            n_complete = self.get_n_complete(pre_study)
            if self.n_trials > n_complete:
                pre_study.optimize(
                    self.pre_objective,
                    n_jobs=1,
                    callbacks=[MaxTrialsCallback(self.n_trials, states=(TrialState.COMPLETE,))],
                )
            best_params = pre_study.best_params
            update_model_cofig(self.default_config, best_params)
            self.pre_study = False
            if self.model_name != "rerx":
                self.fit_pre_model(self.default_config, self.X, self.y)

        study = optuna.create_study(
            storage=self.storage,
            study_name=self.study_name,
            directions=["maximize", "maximize"],
            sampler=optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=self.n_startup_trials,
            ),
            load_if_exists=True,
        )
        n_complete = self.get_n_complete(study)
        ########################################
        if self.model_name == "fbts":
            self.n_trials = 20
        #######################################
        if self.n_trials > n_complete:
            study.optimize(
                self.objective,
                n_jobs=1,
                callbacks=[MaxTrialsCallback(self.n_trials, states=(TrialState.COMPLETE,))],
            )
        best_params = self._get_best_params(study)
        if "j48graft" in self.model_name:
            best_params["min_instance"] = 2 ** best_params["min_instance_power"]
            del best_params["min_instance_power"]
        if "rerx" in self.model_name:
            best_params["tree.min_instance"] = 2 ** best_params["tree.min_instance_power"]
            del best_params["tree.min_instance_power"]
        update_model_cofig(self.default_config, best_params)
        return self.default_config
