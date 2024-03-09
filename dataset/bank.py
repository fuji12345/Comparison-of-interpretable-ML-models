import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame
from .utils import get_categorical_features


class Bank(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = pd.read_csv(to_absolute_path("datasets/bank.txt"), sep=";")

        self.target_column = "y"
        self.categorical_columns = get_categorical_features(self.data, self.target_column)
        self.continuous_columns = [x for x in self.data.columns[:-1] if x not in self.categorical_columns]
