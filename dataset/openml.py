import logging

import openml

from .tabular_dataframe import TabularDataFrame

logger = logging.getLogger(__name__)

exceptions_binary = [45062]
exceptions_multiclass = [44135]
correct_cate_dict = {
    53: {
        "age": False,
        "sex": True,
        "chest": True,
        "resting_blood_pressure": False,
        "serum_cholestoral": False,
        "fasting_blood_sugar": True,
        "resting_electrocardiographic_results": True,
        "maximum_heart_rate_achieved": False,
        "exercise_induced_angina": True,
        "oldpeak": False,
        "slope": True,
        "number_of_major_vessels": False,
        "thal": True,
        "class": True,
    },
}


def get_task_and_dim_out(data_id, df, columns, cate_indicator, target_col):
    target_idx = columns.index(target_col)

    if data_id in exceptions_binary:
        task = "binary"
        dim_out = 1
    elif data_id in exceptions_multiclass:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    elif cont_checker(df, target_col, cate_indicator[target_idx]):
        task = "regression"
        dim_out = 1
    elif int(df[target_col].nunique()) == 2:
        task = "binary"
        dim_out = 1
    else:
        task = "multiclass"
        dim_out = int(df[target_col].nunique())
    return task, dim_out


def update_cate_indicator(data_id, cate_indicator, columns):
    if data_id not in correct_cate_dict:
        return cate_indicator

    for col, is_cate in zip(columns, cate_indicator):
        assert col in correct_cate_dict[data_id]
        if is_cate != correct_cate_dict[data_id][col]:
            logger.info(f"Update {col} from {is_cate} to {correct_cate_dict[data_id][col]}")
        cate_indicator[columns.index(col)] = correct_cate_dict[data_id][col]
    return cate_indicator


def cont_checker(df, col, is_cate):
    return not is_cate and df[col].dtype != bool and df[col].dtype != object


def cate_checker(df, col, is_cate):
    return is_cate or df[col].dtype == bool or df[col].dtype == object


def get_columns_list(df, columns, cate_indicator, target_col, checker):
    return [col for col, is_cate in zip(columns, cate_indicator) if col != target_col and checker(df, col, is_cate)]


def print_dataset_details(dataset: openml.datasets.OpenMLDataset):
    df, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")
    print(dataset.name)
    print(dataset.openml_url)
    print(df)

    target_col = dataset.default_target_attribute
    print("Nan count", df.isna().sum().sum())
    print("cont", get_columns_list(df, columns, cate_indicator, target_col, cont_checker))
    print("cate", get_columns_list(df, columns, cate_indicator, target_col, cate_checker))
    print("target", target_col)

    task, dim_out = get_task_and_dim_out(dataset.id, df, columns, cate_indicator, target_col)
    print(f"task: {task}")
    print(f"dim_out: {dim_out}")
    print(df[target_col].value_counts())
    exit()


class OpenMLDataFrame(TabularDataFrame):
    def __init__(self, id: str, show_details: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)

        dataset = openml.datasets.get_dataset(id)
        if show_details:
            print_dataset_details(dataset)

        data, _, cate_indicator, columns = dataset.get_data(dataset_format="dataframe")

        cate_indicator = update_cate_indicator(id, cate_indicator, columns)

        target_col = dataset.default_target_attribute
        self.continuous_columns = get_columns_list(data, columns, cate_indicator, target_col, cont_checker)
        self.categorical_columns = get_columns_list(data, columns, cate_indicator, target_col, cate_checker)

        self.task, self.dim_out = get_task_and_dim_out(dataset.id, data, columns, cate_indicator, target_col)

        assert self.task != "regression"

        self.target_column = target_col
        self.data = data.dropna(axis=0)
