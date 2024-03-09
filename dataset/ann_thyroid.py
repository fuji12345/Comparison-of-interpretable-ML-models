import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame


class AnnThyroid(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        columns = ["x" + str(i) for i in range(21)]
        self.target_column = "class"
        dummy_cols = ["x22", "x23"]

        train = pd.read_csv(
            to_absolute_path("datasets/ann-train.data"),
            sep=" ",
            names=columns + [self.target_column] + dummy_cols,
            na_values="?",
            engine="python",
        )

        test = pd.read_csv(
            to_absolute_path("datasets/ann-test.data"),
            sep=" ",
            names=columns + [self.target_column] + dummy_cols,
            na_values="?",
            engine="python",
        )

        self.data = pd.concat([train, test], axis=0)
        self.data.drop(dummy_cols, axis=1, inplace=True)

        self.continuous_columns = [x for x in columns if x != self.target_column]
        self.categorical_columns = []
