import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path

from .tabular_dataframe import TabularDataFrame


def get_thyroid(file_path):
    with open(file_path, "r") as f:
        records = []
        for line in f:
            line = line.split("|")[0].split(",")
            records.append({i: val for i, val in enumerate(line)})
    return pd.DataFrame(records)


class Thyroid(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        train = get_thyroid(to_absolute_path("datasets/allbp.data"))
        test = get_thyroid(to_absolute_path("datasets/allbp.test"))
        numeric_idices = [0, 17, 19, 21, 23, 25]
        for indx in numeric_idices:
            train[indx] = train[indx].replace("?", np.nan).astype(float)
            test[indx] = test[indx].replace("?", np.nan).astype(float)
            mean = train[indx].mean()
            train[indx] = train[indx].fillna(mean)
            test[indx] = test[indx].fillna(mean)

        columns = ["x" + str(col) for col in train.columns[:-1]] + ["class"]
        train.columns = columns
        test.columns = columns

        self.continuous_columns = ["x" + str(i) for i in numeric_idices]
        self.categorical_columns = [x for x in columns[:-1] if x not in self.continuous_columns]
        self.target_column = columns[-1]
        self.train = train
        self.test = test
