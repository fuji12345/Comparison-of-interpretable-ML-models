import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from rerx import MLP, J48graft, ReRx


def get_X_y(data, feature_cols, label_col):
    X, y = data[feature_cols], data[label_col].values.squeeze()
    return X, y


def main():
    os.makedirs("outputs", exist_ok=True)

    data = load_breast_cancer()
    feature_cols, label_col = data.feature_names, "class"
    X, y = data.data, data.target
    data = pd.DataFrame(X, columns=data.feature_names)
    data["class"] = y

    train_data, test_data = train_test_split(data, test_size=0.2)
    train_data, val_data = train_test_split(train_data, test_size=0.1)

    mlp = MLP(len(feature_cols), 2, h_dim=10)
    tree = J48graft(out_dir="outputs/")
    model = ReRx(base_model=mlp, tree=tree, output_dim=2, is_eval=True)

    X_train, y_train = get_X_y(train_data, feature_cols, label_col)
    X_val, y_val = get_X_y(val_data, feature_cols, label_col)
    X_test, y_test = get_X_y(test_data, feature_cols, label_col)

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    scores = model.evaluate(X_test, y_test)
    for metrics, score in scores.items():
        print(f"{metrics:<15}: {score}")


if __name__ == "__main__":
    main()
