import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame
from .utils import get_categorical_features


class German(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        columns = ["x" + str(i) for i in range(20)]
        self.target_column = "class"

        self.data = pd.read_csv(to_absolute_path("datasets/german.data"), sep=" ", names=columns + [self.target_column])

        self.categorical_columns = get_categorical_features(self.data, self.target_column)
        self.continuous_columns = [x for x in columns if x not in self.categorical_columns]
