import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame


class Occupancy(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data = pd.read_csv(to_absolute_path("datasets/occupancy/datatraining.txt"), sep=",")
        self.data.drop(columns=["date"], inplace=True)
        self.target_column = "Occupancy"
        self.categorical_columns = []
        self.continuous_columns = [x for x in self.data.columns if x not in self.target_column]
