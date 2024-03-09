import logging
import os
from time import time

import numpy as np
import optuna
import pandas as pd
from hydra.utils import to_absolute_path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import dataset
from dataset import TabularDataFrame

from .classifier import get_classifier
from .optuna import OptimParam
from .utils import (
    cal_metrics,
    load_json,
    save_json,
    save_ruleset,
    set_categories_in_rule,
    set_seed,
)

logger = logging.getLogger(__name__)


class ExpBase:
    def __init__(self, config):
        set_seed(config.seed)

        self.n_splits = config.n_splits
        self.model_name = config.model.name

        self.model_config = config.model.params
        self.exp_config = config.exp
        self.data_config = config.data

        dataframe: TabularDataFrame = getattr(dataset, self.data_config.name)(seed=config.seed, **self.data_config)
        dfs = dataframe.processed_dataframes()
        self.categories_dict = dataframe.get_categories_dict()
        self.train, self.test = dfs["train"], dfs["test"]
        self.columns = dataframe.all_columns
        self.target_column = dataframe.target_column

        # Onehotter for MLP in ReRx
        if self.model_name == "rerx":
            all_cate = pd.concat([self.train, self.test])[dataframe.categorical_columns]
            self.onehoter = OneHotEncoder(sparse_output=False).fit(all_cate) if len(all_cate.columns) != 0 else None
        else:
            self.onehoter = None

        self.seed = config.seed
        self.init_writer()

    def init_writer(self):
        metrics = [
            "fold",
            "ACC",
            "AUC",
            "Num of Rules",
            "Ave. ante.",
            "CREP",
            "Precision",
            "Recall",
            "Specificity",
            "F1",
            "Time",
        ]
        self.writer = {m: [] for m in metrics}
        if os.path.exists("results.json"):
            _writer = load_json("results.json")
            for k, v in _writer.items():
                self.writer[k] = v
            self.writer["fold"] = list(range(len(_writer["Time"])))

    def add_results(self, i_fold, scores: dict, time):
        self.writer["fold"].append(i_fold)
        self.writer["Time"].append(time)
        for m, score in scores.items():
            self.writer[m].append(score)
        save_json({k: v for k, v in self.writer.items() if k != "fold"})

    def each_fold(self, i_fold, train_data, val_data):
        uniq = self.get_unique(train_data)
        x, y = self.get_x_y(train_data)

        model_config = self.get_model_config(i_fold=i_fold, x=x, y=y, val_data=val_data, uniq=uniq)
        model = get_classifier(
            self.model_name,
            input_dim=len(self.columns),
            output_dim=len(uniq),
            model_config=model_config,
            init_y=y,
            onehoter=self.onehoter,
            verbose=self.exp_config.verbose,
            seed=self.seed,
        )
        start = time()
        model.fit(
            x,
            y,
            eval_set=(val_data[self.columns], val_data[self.target_column].values.squeeze()),
        )
        end = time() - start
        logger.info(f"[Fit {self.model_name}] Time: {end}")
        return model, end

    def run(self):
        skf = StratifiedKFold(n_splits=self.n_splits)
        for i_fold, (train_idx, val_idx) in enumerate(skf.split(self.train, self.train[self.target_column])):
            if len(self.writer["fold"]) != 0 and self.writer["fold"][-1] >= i_fold:
                logger.info(f"Skip {i_fold + 1} fold. Already finished.")
                continue

            train_data, val_data = self.train.iloc[train_idx], self.train.iloc[val_idx]
            model, time = self.each_fold(i_fold, train_data, val_data)

            score = cal_metrics(model, val_data, self.columns, self.target_column)
            score.update(model.evaluate(val_data[self.columns], val_data[self.target_column].values.squeeze()))
            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] val/ACC: {score['ACC']:.4f} | val/AUC: {score['AUC']:.4f} | "
                f"val/Rules: {score['Num of Rules']}"
            )

            score = cal_metrics(model, self.test, self.columns, self.target_column)
            score.update(model.evaluate(self.test[self.columns], self.test[self.target_column].values.squeeze()))

            logger.info(
                f"[{self.model_name} results ({i_fold+1} / {self.n_splits})] test/ACC: {score['ACC']:.4f} | test/AUC: {score['AUC']:.4f} | "
                f"test/Rules: {score['Num of Rules']}"
            )
            self.add_results(i_fold, score, time)

            if self.model_name == "rerx" and self.categories_dict is not None:
                set_categories_in_rule(model.ruleset, self.categories_dict)

            save_ruleset(model.ruleset, save_dir="ruleset", file_name=f"ruleset_{i_fold+1}")

        logger.info(f"[{self.model_name} Test Results]")
        mean_std_score = {}
        score_list_dict = {}
        for k, score_list in self.writer.items():
            if k == "fold":
                continue
            score = np.array(score_list)
            mean_std_score[k] = f"{score.mean(): .4f} ±{score.std(ddof=1): .4f}"
            score_list_dict[k] = score_list
            logger.info(f"[{self.model_name} {k}]: {mean_std_score[k]}")
        save_json(score_list_dict)

    def get_model_config(self, *args, **kwargs):
        raise NotImplementedError()

    def get_unique(self, train_data):
        uniq = np.unique(train_data[self.target_column])
        return uniq

    def get_x_y(self, train_data):
        x, y = train_data[self.columns], train_data[self.target_column].values.squeeze()
        return x, y


class ExpSimple(ExpBase):
    def __init__(self, config):
        super().__init__(config)

    def get_model_config(self, *args, **kwargs):
        return self.model_config


class ExpOptuna(ExpBase):
    def __init__(self, config):
        super().__init__(config)
        self.n_trials = config.exp.n_trials
        self.n_startup_trials = config.exp.n_startup_trials

        self.storage = config.exp.storage
        self.study_name = config.exp.study_name
        self.cv = config.exp.cv
        self.alpha = config.exp.alpha

    def run(self):
        if self.exp_config.delete_study:
            for i in range(self.n_splits):
                study_name = f"{self.study_name}_{i}"
                try:
                    optuna.delete_study(
                        study_name=study_name,
                        storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                    )
                    print(f"Successfully deleted study {study_name}")
                except:
                    print(f"study {study_name} not found.")
                if self.model_name == "rerx":
                    try:
                        study_name = f"{study_name}_pre"
                        optuna.delete_study(
                            study_name=study_name,
                            storage=f"sqlite:///{to_absolute_path(self.exp_config.storage)}/optuna.db",
                        )
                        print(f"Successfully deleted study {study_name}")
                    except:
                        print(f"study {study_name} not found.")
                
            return
        super().run()

    def get_model_config(self, i_fold, x, y, val_data, uniq, *args, **kwargs):
        op = OptimParam(
            self.model_name,
            default_config=self.model_config,
            input_dim=len(self.columns),
            output_dim=len(uniq),
            X=x,
            y=y,
            val_data=val_data,
            columns=self.columns,
            target_column=self.target_column,
            onehoter=self.onehoter,
            n_trials=self.n_trials,
            n_startup_trials=self.n_startup_trials,
            storage=self.storage,
            study_name=f"{self.study_name}_{i_fold}",
            cv=self.cv,
            seed=self.seed,
            alpha=self.alpha,
        )
        return op.get_best_config()
