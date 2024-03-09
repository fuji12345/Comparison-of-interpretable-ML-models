# Recursive-Rule eXtraction (Re-RX) algorithm

## Environment
- python>=3.7.11
- If you want to use `J48graft`, you need to install `Java`.
  - See `Custom Run -> BaseTree -> Setting for using J48graft`

## How to run
### 1. Install Re-RX
Run the following command in the directory where `setup.py` is located

```bash
pip install .
```

### 2. Run example
```bash
python example.py
```

## Custom run
### ReRX
You can use `ReRX` by `from rerx import ReRX`.
The arguments of `ReRX` are as follows:

- base_model
  - A model that satisfies the requirements of `BaseModel`.
- tree
  - A model that satisfies the requirements of `BaseTree`.

#### Options
- output_dim [default 2].
  - Output dimension of class
- pruning_lamda [default 0.01].
  - Threshold of drop in classification performance to stop pruning
- pruning_step [default 0.01]
  - Step size for the pruning phase
- is_increasing_decision_score [default True]
  - High score is better score or not
- delta_1
  - Threshold of the support (coverage) rate
- delta_2
  - threshold for the error rate
- min_instance [default 10]
  - Minimum number of instances to fit the tree.
- is_pruning [default True]
  - Whether to prune or not in the fitting phase.
- is_eval [default False]
  - Whether to evaluate the model or not in the fitting phase.
- verbose [default 1]
  - 0: no log
  - 1: log
- log_func [default `print`]

#### Setting for using J48graft
J48graft operates based on weka and requires Java to be installed.

```bash
sudo apt update
sudo apt upgrade
sudo apt autoremove
sudo apt install net-tools
sudo apt install build-essential
sudo apt install make
sudo apt install cmake
sudo apt install screen
sudo apt install curl
sudo apt install vim-nox
```

Check the version of Java that can be installed.
```bash
sudo apt search openjdk-\(\.\)\+-jdk$
```

Install the installable version.
```bash
sudo apt install openjdk-{version}-jdk
```

Check installed version.
```bash
java -version
```


### Examples
#### Use J48graft
```python
from rerx import MLP, J48graft, ReRX

mlp = MLP()
j48graft = J48graft()
rerx = ReRX(base_model=mlp, tree=j48graft)
rerx.fit(X, y)
score = rerx.evaluate(X_test, y_test)
```

#### Use CART (DecisionTreeClassifier)
```python
from rerx import MLP, ReRX
from sklearn.tree import DecisionTreeClassifier


mlp = MLP()
cart = DecisionTreeClassifier()
rerx = ReRX(base_model=mlp, tree=cart)
rerx.fit(X, y)
score = rerx.evaluate(X_test, y_test)
```
