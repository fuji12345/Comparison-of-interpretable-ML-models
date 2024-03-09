import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils import Logger


class BaseTree:
    # def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    #     NotImplementedError()

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None) -> None:
        NotImplementedError()


class J48graft(BaseTree):
    def __init__(
        self,
        *,
        tree="j48graft",
        mode="ori",
        min_instance: Union[int, float] = 2,
        pruning_conf=0.25,
        out_dir: Union[Path, str] = Path("./j48graft/"),
        verbose=1,
        log_func="print",
    ) -> None:
        super().__init__()
        self.logger = Logger("J48graft", verbose, log_func=log_func)

        self.tree = tree
        self.mode = mode
        self.min_instance = min_instance
        self.pruning_conf = pruning_conf

        if isinstance(out_dir, str):
            out_dir = Path(out_dir)
        out_dir = out_dir.resolve()
        self.out_dir = out_dir
        self.data_path = out_dir / "datas.csv"
        self.result_txt_name = out_dir / "weka_rules.txt"
        self.weka_dir = Path(__file__).parent.resolve() / "weka"

        self.data_path.parent.mkdir(parents=True, exist_ok=True)

    def data2csv(self, X: pd.DataFrame, y: np.ndarray) -> None:
        data = X.copy()
        # print(np.unique(y, return_counts=True))
        data["class"] = y
        data.to_csv(self.data_path, index=False)
        # assert len(data["class"].unique()) >= 2, "The training data for weka J48graft consists of only a single class."

    def fit(self, X: pd.DataFrame, y: np.ndarray, eval_set: Optional[Tuple[pd.DataFrame, np.ndarray]] = None) -> None:
        min_instance = self.min_instance
        if isinstance(min_instance, float):
            min_instance = int(min_instance * len(y))
            self.logger(f"min_instance is changed to {min_instance}.")

        size_minor_class = np.unique(y, return_counts=True)[1].min()
        if size_minor_class < min_instance:
            self.logger(f"Warning : The number of minor class is less than {min_instance}.")
            min_instance = size_minor_class
            self.logger(f"min_instance is changed to {min_instance}.")

        self.data2csv(X, y)

        # choose categorical colmns index (java index start from 1, so + 1)
        cate_cols_index = [str(X.columns.get_loc(cate_col) + 1) for cate_col in X.select_dtypes("int").columns]
        cate_cols_index = ",".join(cate_cols_index) if cate_cols_index != [] else '""'

        success = subprocess.call(
            [
                "sh",
                self.weka_dir / "my_weka.sh",
                self.weka_dir,
                cate_cols_index,
                self.out_dir / "weka_datas.arff",
                self.result_txt_name,
                self.mode,
                self.tree,
                str(min_instance),
                str(self.pruning_conf),
                self.data_path,
            ]
        )
        assert success == 0, "weka is failed"
        self.logger("Success : weka")

    def read_tree_from_text(self):
        pattern = r"J48graft pruned tree\n-+\n([\s\S]+?)(?=\n\n|\Z)"

        with open(self.result_txt_name, "r") as f:
            match = re.search(pattern, f.read())

        if match:
            decision_tree_text = match.group(1).split("\n")
        else:
            raise KeyError("No decision tree was found.")

        if len(decision_tree_text) == 1:
            self.logger("All prediction by weka J48graft are in the same class.")
        else:
            decision_tree_text = decision_tree_text[1:]
        return decision_tree_text
