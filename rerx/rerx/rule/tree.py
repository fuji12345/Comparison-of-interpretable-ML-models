import operator as op
import re
from typing import List, Union

operator_dict = {
    "=": op.eq,
    "!=": op.ne,
    "<=": op.le,
    ">=": op.ge,
    "<": op.lt,
    ">": op.gt,
}


class Leaf:
    def __init__(self, value, class_distribution):
        self.value = int(value)
        self.class_distribution = class_distribution


class Node:
    def __init__(self, feature, children, operator, threshold):
        self.feature = feature
        self.childrens = [children]
        self.operators = [operator]
        self.thresholds = [threshold]

    def append(self, children, operator, threshold):
        self.childrens.append(children)
        self.operators.append(operator)
        self.thresholds.append(threshold)

    def feat_to_idx(self, columns):
        self.feature = columns.index(self.feature.replace("|   ", ""))


class Tree:
    def __init__(self) -> None:
        self.tree: List[Union[Node, Leaf]] = []

    def __getitem__(self, idx):
        return self.tree[idx]

    def make_tree(self):
        raise NotImplementedError()

    def append(self, node_or_leaf: Union[Node, Leaf]):
        self.tree.append(node_or_leaf)


class J48graftTree(Tree):
    def __init__(self, columns: List[str], tree_text: List[str]) -> None:
        super().__init__()
        self.tree_text = tree_text
        self.columns = columns
        self.make_tree()

    def get_observed_node(self, feature):
        for node in self.tree:
            if isinstance(node, Node) and node.feature == feature:
                return node
        return None

    def _append(self, feature, operator, threshold):
        current_tree_size = len(self.tree)
        node = self.get_observed_node(feature)
        if node is None:
            self.append(Node(feature, current_tree_size + 1, operator, threshold))
        else:
            # node is not added. children is current_tree_size
            node.append(current_tree_size, operator, threshold)

    def feat_to_idx(self):
        for node_or_leaf in self.tree:
            if isinstance(node_or_leaf, Node):
                node_or_leaf.feat_to_idx(self.columns)

    def make_tree(self):
        if len(self.tree_text) == 1:
            node = self.tree_text[0]
            node = node.replace(": ", "")
            value, class_distribution = node.split(" ")
            self.append(Leaf(value, class_distribution))
            return

        # 行ごとに決定木のテキストを処理
        for line in self.tree_text:
            # 分割
            parts = re.split(r" (=|!=|<=|>=|<|>) ", line)

            # 葉ノードを判定する正規表現パターン
            leaf_pattern = r"(\d+) \("
            leaf_match = re.search(leaf_pattern, parts[-1])

            if leaf_match:
                threshold_value, class_value = re.split(r": ", parts[-1])
                parts[-1] = threshold_value

            feat = parts[0].strip()
            operator = operator_dict[parts[1].strip()]
            threshold = parts[2].strip()
            if threshold.isdigit():
                threshold = int(threshold)
            else:
                threshold = float(threshold)

            self._append(feat, operator, threshold)

            if leaf_match:
                value, class_distribution = class_value.split(" ")
                self.append(Leaf(value, class_distribution))

        self.feat_to_idx()
