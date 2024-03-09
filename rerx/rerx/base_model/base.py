from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.initializers import Constant
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD, Adam, AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from ..utils import Logger


class BaseModel(metaclass=ABCMeta):
    def initialize(self, input_dim: int, y=None):
        NotImplementedError()

    @abstractmethod
    def get_weights(self):
        ...

    @abstractmethod
    def set_weights(self, weights):
        ...

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None,
        *args,
        **kwargs,
    ) -> None:
        NotImplementedError()

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: np.ndarray, *args, **kwargs) -> Dict[str, float]:
        NotImplementedError()

    @abstractmethod
    def predict(self, X: pd.DataFrame):
        NotImplementedError()

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame):
        NotImplementedError()

    @abstractmethod
    def get_current_score(self, X: pd.DataFrame, y: np.ndarray):
        NotImplementedError()

    @abstractmethod
    def pruning(self, ratio: float):
        NotImplementedError()

    @abstractmethod
    def get_droped_columns(self, X):
        NotImplementedError()


def get_scaler(name: str) -> Union[StandardScaler, MinMaxScaler]:
    if name == "standard":
        return StandardScaler()
    elif name == "minmax":
        return MinMaxScaler()
    else:
        raise ValueError(name)


class MLP(BaseModel):
    def __init__(
        self,
        use_output_bias=True,
        h_dim=1,
        act="relu",
        lr=0.01,
        momentum=0.99,
        weight_decay=5e-4,
        epochs=50,
        batch_size="auto",
        eval_batch_size=4096,
        optimizer="adam",
        pruning_metric="acc",
        scaler=None,
        onehot=True,
        onehoter: OneHotEncoder = None,
        early_stop=False,
        verbose=0,
        log_func="print",
    ) -> None:
        super().__init__()
        self.logger = Logger("MLP", verbose=verbose, log_func=log_func)

        self.use_output_bias = use_output_bias

        self.h_dim = h_dim
        self.act = act

        self.optimizer = optimizer
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.epochs = epochs
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.pruning_metric = pruning_metric

        self.cont_scaler = get_scaler(scaler) if scaler is not None else None
        self.cate_onehoter = None
        if onehot:
            if onehoter is None:
                self.cate_onehoter = OneHotEncoder(sparse_output=False)
            else:
                self.cate_onehoter = onehoter
                self.categories = {k: list(v) for k, v in zip(onehoter.feature_names_in_, onehoter.categories_)}

        self.early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, verbose=verbose) if early_stop else None

        self.verbose = verbose
        self.is_fit = False

    def init_bias(self):
        total = sum(self.label_frec)
        output_bias = []
        for i in range(len(self.label_frec)):
            p_i = self.label_frec[i] / total
            b_i = -np.log((1 / p_i) - 1)
            output_bias.append(b_i)
        return output_bias

    def make_model(self, input_dim, output_bias=None) -> Model:
        if output_bias is not None:
            output_bias = Constant(output_bias)
        inputs = Input(shape=(input_dim,))

        x = Dense(units=self.h_dim, kernel_initializer="uniform", activation=self.act, name="hidden_layer")(inputs)

        outputs = Dense(
            units=len(self.unique_label),
            kernel_initializer="uniform",
            activation="softmax",
            name="last_layer",
            bias_initializer=output_bias,
        )(x)

        return Model(inputs=inputs, outputs=outputs)

    def initialize(self, input_dim: int, y: np.ndarray = None):
        self.logger.info(f"MLP input_dim: {input_dim}")
        self.unique_label, self.label_frec = np.unique(y, return_counts=True)
        output_bias = self.init_bias()

        self.model = self.make_model(input_dim, output_bias if self.use_output_bias else None)
        if self.optimizer == "adam":
            optimizer = Adam(learning_rate=self.lr)
        elif self.optimizer == "adamw":
            optimizer = AdamW(learning_rate=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = SGD(learning_rate=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.model.compile(
            # optimizer = 'rmsprop',
            optimizer=optimizer,
            loss="binary_crossentropy" if len(self.unique_label) == 2 else "categorical_crossentropy",
            metrics=["accuracy"],
        )

    def change_target_shape(self, y: np.ndarray):
        if len(self.unique_label) != 1:
            y = tf.one_hot(y, len(self.unique_label))
        return y

    def split_data(self, X: pd.DataFrame):
        if self.cate_onehoter is None:
            return pd.DataFrame(), X
        return X.select_dtypes(include=int), X.select_dtypes(include=float)

    def cate_transform(self, X: pd.DataFrame):
        _x = X.copy().reset_index(drop=True)
        fit_columns = list(self.cate_onehoter.feature_names_in_)
        dropped_cols = [col for col in fit_columns if col not in X.columns]
        onehot_columns = [f"{col}_{value}" for col in X.columns for value in self.categories[col]]
        fit_onehot_columns = sum(
            [[f"{col}_{value}" for value in category] for col, category in self.categories.items()], []
        )
        x_idx = [fit_onehot_columns.index(col) for col in onehot_columns]
        if len(dropped_cols) > 0:
            dummy_x = pd.DataFrame(np.zeros((_x.shape[0], len(dropped_cols))), columns=dropped_cols)
            _x = pd.concat([_x, dummy_x], axis=1)
        _x = self.cate_onehoter.transform(_x[fit_columns])[:, x_idx]
        return _x

    def cont_transform(self, X: pd.DataFrame):
        _x = X.copy().reset_index(drop=True)
        fit_columns = list(self.cont_scaler.feature_names_in_)
        dropped_cols = [col for col in fit_columns if col not in X.columns]
        x_idx = [fit_columns.index(col) for col in X.columns]
        if len(dropped_cols) > 0:
            dummy_x = pd.DataFrame(np.zeros((_x.shape[0], len(dropped_cols))), columns=dropped_cols)
            _x = pd.concat([_x, dummy_x], axis=1)
        _x = self.cont_scaler.transform(_x[fit_columns])[:, x_idx]
        return _x

    def scaler_transform(self, X: pd.DataFrame) -> np.ndarray:
        X_cate, X_cont = self.split_data(X)
        X = []
        if not X_cate.empty:
            if not hasattr(self.cate_onehoter, "feature_names_in_"):
                self.cate_onehoter.fit(X_cate)
                self.categories = {k: list(v) for k, v in zip(X_cate.columns, self.cate_onehoter.categories_)}
            X.append(self.cate_transform(X_cate))
        if not X_cont.empty:
            if not hasattr(self.cont_scaler, "feature_names_in_"):
                self.cont_scaler.fit(X_cont)
            X.append(self.cont_transform(X_cont))
        X = np.concatenate(X, axis=1)
        return X

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None) -> None:
        X = self.scaler_transform(X)

        self.initialize(X.shape[1], y)
        y = self.change_target_shape(y)

        callbacks = []
        if eval_set is not None:
            y_val = self.change_target_shape(eval_set[1])
            eval_set = (self.scaler_transform(eval_set[0]), y_val)
            if self.early_stopping is not None:
                callbacks.append(self.early_stopping)

        batch_size = self.batch_size
        if batch_size == "auto":
            batch_size = 2 ** int(np.log(X.shape[0]))
            self.logger.info(f"Batch size: {batch_size}")

        self.model.fit(
            X,
            y,
            validation_data=eval_set,
            batch_size=batch_size,
            epochs=self.epochs,
            verbose=0,
            shuffle=True,
            callbacks=callbacks,
        )

    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        results = {}
        y_pred = self.predict(X)

        results["ACC"] = accuracy_score(y, y_pred)
        if len(self.unique_label) == 2:
            results["AUC"] = roc_auc_score(y, y_pred)
            results["Precision"] = precision_score(y, y_pred)
            results["Recall"] = recall_score(y, y_pred)
            results["Specificity"] = recall_score(1 - y, 1 - y_pred)
            results["F1"] = f1_score(y, y_pred)
        return results

    def predict(self, X: pd.DataFrame):
        X = self.scaler_transform(X)
        return np.argmax(self.model.predict(X, batch_size=self.eval_batch_size, verbose=0), axis=1)

    def predict_proba(self, X: pd.DataFrame):
        X = self.scaler_transform(X)
        return self.model.predict(X, batch_size=self.eval_batch_size, verbose=0)

    def get_current_score(self, X: pd.DataFrame, y: np.ndarray):
        if self.pruning_metric == "acc":
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred)
        elif self.pruning_metric == "auc":
            y_pred = self.predict_proba(X)
            if len(self.unique_label) == 2:
                return roc_auc_score(y, y_pred[:, 1])
            else:
                return roc_auc_score(y, y_pred, multi_class="ovr")
        else:
            raise ValueError(self.pruning_metric)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        return self.model.set_weights(weights)

    def pruning(self, ratio: float):
        weights = self.get_weights()
        weight_i2h = weights[0]
        weight_i2h = weight_i2h.flatten()
        weight_i2h = list(map(abs, weight_i2h))

        weight_i2h_index = np.argsort(weight_i2h)

        for i in range(int(len(weight_i2h_index) * ratio)):
            change_zero = weight_i2h_index[i]
            a = change_zero // self.h_dim
            b = change_zero % self.h_dim
            weights[0][a, b] = 0

        self.set_weights(weights)

    def get_droped_columns(self, X: pd.DataFrame):
        X_cate, X_cont = self.split_data(X)
        drop_columns = []
        weights = self.get_weights()
        weight_i2h = weights[0]

        if not X_cate.empty:
            cate_name_onehot = [f"{col}_{value}" for col in X_cate.columns for value in self.categories[col]]
            categories = {k: self.categories[k] for k in X_cate.columns}

            cate_drop_columns = []
            for column, attribute in zip(cate_name_onehot, weight_i2h[:, : len(cate_name_onehot)]):
                if np.all(attribute == 0):
                    cate_drop_columns.append(column)

            for column, category in categories.items():
                if all([f"{column}_{value}" in cate_drop_columns for value in category]):
                    drop_columns.append(column)

        if not X_cont.empty:
            for column, attribute in zip(X_cont.columns, weight_i2h[:, -len(X_cont.columns) :]):
                if np.all(attribute == 0):
                    drop_columns.append(column)

        return drop_columns
