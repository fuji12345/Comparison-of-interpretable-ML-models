import logging
from typing import Dict, Union


class Logger:
    def __init__(self, name: str, verbose=1, log_func="print") -> None:
        self.verbose = verbose
        if log_func == "logging":
            self.__log_func = logging.getLogger(name).info
        elif log_func == "print":
            self.__log_func = print
        else:
            raise ValueError()

    def info(self, msg: object, verbose=None):
        if verbose is None:
            verbose = self.verbose
        if verbose > 0:
            self.__log_func(msg)

    def __call__(self, msg: object, verbose=None):
        self.info(msg, verbose)

    def log_score(self, score: Dict[str, Union[float, str, int]], flag=""):
        if flag != "":
            self.info(f"[{flag}]")
        for k, v in score.items():
            if isinstance(v, float):
                v = f"{v: .4f}"
            if isinstance(v, int):
                v = f"{str(v): >2}"
            self.info(f"\t{k: <15}: {v: >1}")
