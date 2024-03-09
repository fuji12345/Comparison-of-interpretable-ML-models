from sklearn.datasets import load_breast_cancer

from .tabular_dataframe import TabularDataFrame
from .utils import get_sklearn_ds


class BreastCancer(TabularDataFrame):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data, columns, self.target_column = get_sklearn_ds(
            load_breast_cancer(),
        )
        self.continuous_columns = list(columns)
        self.categorical_columns = []
