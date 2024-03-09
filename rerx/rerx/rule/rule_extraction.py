# This file is copied from https://github.com/jobregon1212/rulecosi and modified by somaonishi.
""" This module contains the functions used for extracting the rules for
different type of base ensembles.

The module structure is the following:

- The `BaseRuleExtractor` base class implements a common ``get_base_ruleset``
  and ``recursive_extraction``  method for all the extractors in the module.

    - :class:`rule_extraction.DecisionTreeRuleExtractor` implements rule
        extraction from a single decision tree

    - :class:`rule_extraction.ClassifierRuleExtractor` implements rule
        extraction from a classifier Ensembles such as Bagging and
        Random Forests

    - :class:`rule_extraction.GBMClassifierRuleExtractor` implements rule
        extraction from sklearn GBM classifier and works as base class for the
        other GBM implementations

        - :class:`rule_extraction.XGBClassifierExtractor` implements rule
            extraction from XGBoost classifiers

        - :class:`rule_extraction.LGBMClassifierExtractor` implements rule
            extraction from Light GBM classifiers

        - :class:`rule_extraction.CatBoostClassifierExtractor` implements rule
            extraction from CatBoost classifiers


"""

import copy
import operator as op
from abc import ABCMeta, abstractmethod
from math import copysign
from typing import Any, Dict, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from ..tree import J48graft
from .rules import Condition, Rule, RuleSet
from .tree import J48graftTree, Leaf, Node, Tree


class BaseRuleExtractor(metaclass=ABCMeta):
    """Base abstract class for a rule extractor from tree ensembles"""

    def __init__(self, _tree, _column_names, classes_, X, y, float_threshold):
        self._column_names = _column_names
        self.classes_ = classes_
        self._tree = _tree
        self.X = X
        self.y = y
        self.float_threshold = float_threshold
        _, counts = np.unique(self.y, return_counts=True)
        self.class_ratio = counts.min() / counts.max()

    def get_tree_dict(self, base_tree, n_nodes=0):
        """Create a dictionary with the information inside the base_tree

        :param base_tree: :class: `sklearn.tree.Tree` object which is an array
            representation of a tree

        :param n_nodes: number of nodes in the tree

        :return: a dictionary containing the information of the base_tree
        """
        return {
            "children_left": base_tree.tree_.children_left,
            "children_right": base_tree.tree_.children_right,
            "feature": base_tree.tree_.feature,
            "threshold": base_tree.tree_.threshold,
            "value": base_tree.tree_.value,
            "n_samples": base_tree.tree_.weighted_n_node_samples,
            "n_nodes": base_tree.tree_.node_count,
        }

    @abstractmethod
    def create_new_rule(
        self, node_index, tree_dict, condition_set=None, logit_score=None, weights=None, tree_index=None
    ):
        """Creates a new rule with all the information in the parameters

        :param node_index: the index of the leaf node

        :param tree_dict: a dictionary containing  the information of the
            base_tree (arrays on:class: `sklearn.tree.Tree` class

        :param condition_set: set of :class:`rulecosi.rule.Condition` objects
            of the new rule

        :param logit_score: logit_score of the rule (only applies for Gradient
            Boosting Trees)

        :param weights: weight of the new rule

        :param tree_index: index of the tree inside the ensemble

        :return: a :class:`rulecosi.rules.Rule` object
        """

    @abstractmethod
    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """

    @abstractmethod
    def recursive_extraction(
        self, tree: Union[Tree, Dict[str, Any]], tree_index=0, node_index=0, condition_map=None, condition_set=None
    ):
        """
        TODO: update this context.
        Recursive function for extracting a ruleset from a tree

        :param tree: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param tree_index: index of the tree in the ensemble

        :param node_index: the index of the leaf node

        :param condition_map: condition_map: dictionary of <condition_id,
        Condition>, default=None Dictionary of Conditions extracted from all
        the ensembles. condition_id is an integer uniquely identifying the
        Condition.

        :param condition_set:  set of :class:`rulecosi.rule.Condition` objects

        :return: array of :class:`rulecosi.rules.Rule` objects
        """

    def get_base_ruleset(self, tree_dict, class_index=None, condition_map=None, tree_index=None):
        """

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param class_index: Right now is not used but it will be used
        when multiclass is supported

        :param condition_map: dictionary of <condition_id, Condition>,
        default=None. Dictionary of Conditions extracted from all the
        ensembles.condition_id is an integer uniquely identifying the Condition.

        :param tree_index: index of the tree in the ensemble

        :return:   a :class:`rulecosi.rules.RuleSet' object
        """

        if condition_map is None:
            condition_map = dict()  # dictionary of conditions A

        extracted_rules = self.recursive_extraction(
            tree_dict, tree_index, node_index=0, condition_map=condition_map, condition_set=set()
        )
        return RuleSet(extracted_rules, condition_map, classes=self.classes_)

    def get_split_operators(self):
        """Return the operator applied for the left and right branches of
        the tree. This function is needed because different implementations
        of trees use different operators for the children nodes.

        :return: a tuple containing the left and right operator used for
        creating conditions
        """
        op_left = op.le  # Operator.LESS_OR_EQUAL_THAN
        op_right = op.gt  # Operator.GREATER_THAN
        return op_left, op_right


class J48graftRuleExtractor(BaseRuleExtractor):
    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """
        self._tree: J48graft

        global_condition_map = dict()
        decision_tree_text = self._tree.read_tree_from_text()
        tree = J48graftTree(self._column_names, decision_tree_text)
        original_ruleset = self.get_base_ruleset(tree)
        global_condition_map.update(original_ruleset.condition_map)
        return original_ruleset, global_condition_map

    def create_new_rule(self, value, condition_set=None, logit_score=None, weights=None, tree_index=None):
        """Creates a new rule with all the information in the parameters

        :param node_index: the index of the leaf node

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param condition_set: set of :class:`rulecosi.rule.Condition` objects
        of the new rule

        :param logit_score: logit_score of the rule (only applies for
        Gradient Boosting Trees)

        :param weights: weight of the new rule

        :param tree_index: index of the tree inside the ensemble

        :return: a :class:`rulecosi.rules.Rule` object
        """
        if condition_set is None:
            condition_set = {}

        if weights is not None:
            weight = weights[tree_index]
        else:
            weight = None

        class_dist = np.zeros(len(self.classes_)).reshape((len(self.classes_),))
        y_class_index = self.classes_.tolist().index(value)
        class_dist[y_class_index] = 1

        # predict y_class_index = np.argmax(class_dist).item()
        y = np.array([self.classes_[y_class_index]])

        return Rule(
            set(condition_set),
            class_dist=class_dist,
            ens_class_dist=class_dist,
            logit_score=logit_score,
            y=y,
            y_class_index=y_class_index,
            n_samples=None,
            classes=self.classes_,
            weight=weight,
        )

    def recursive_extraction(self, tree: Tree, tree_index=0, node_index=0, condition_map=None, condition_set=None):
        """Recursive function for extracting a ruleset from a tree

        :param tree: a Tree object containing  the information of the
        J48graft

        :param tree_index: index of the tree in the ensemble

        :param node_index: the index of the leaf node

        :param condition_map: condition_map: dictionary of <condition_id,
        Condition>, default=None Dictionary of Conditions extracted from all
        the ensembles. condition_id is an integer uniquely identifying the
        Condition.

        :param condition_set:  set of :class:`rulecosi.rule.Condition` objects

        :return: array of :class:`rulecosi.rules.Rule` objects
        """
        if condition_map is None:
            condition_map = dict()
        if condition_set is None:
            condition_set = set()
        rules = []
        node = tree[node_index]

        # leaf node so a rule is created
        if isinstance(node, Leaf):
            weights = None
            logit_score = None
            new_rule = self.create_new_rule(node.value, condition_set, logit_score, weights, tree_index)
            rules.append(new_rule)
            return rules

        node: Node
        feature = node.feature
        childrens = node.childrens
        operators = node.operators
        thresholds = node.thresholds

        # create condition, add it to the condition_set and get conditions from left and right child
        att_name = None
        if self._column_names is not None:
            att_name = self._column_names[feature]

        for children, operator, threshold in zip(childrens, operators, thresholds):
            condition_set_i = copy.copy(condition_set)
            new_condition = Condition(
                feature,
                operator,
                threshold,
                att_name,
            )
            condition_map[hash(new_condition)] = new_condition
            condition_set_i.add((hash(new_condition), new_condition))
            i_rules = self.recursive_extraction(
                tree,
                tree_index,
                node_index=children,
                condition_set=condition_set_i,
                condition_map=condition_map,
            )
            rules += i_rules
        return rules


class DecisionTreeRuleExtractor(BaseRuleExtractor):
    """Rule extraction of a single decision tree classifier

    Parameters
    ----------
    base_tree: Parameter kept just for compatibility with the other classes

    column_names: array of string, default=None Array of strings with the
    name of the columns in the data. This is useful for displaying the name
    of the features in the generated rules.

    classes: ndarray, shape (n_classes,)
        The classes seen when fitting the ensemble.

    X: array-like, shape (n_samples, n_features)
        The training input samples.

    """

    def extract_rules(self):
        """Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulecosi.rules.RuleSet'
        """

        global_condition_map = dict()
        original_ruleset = self.get_base_ruleset(self.get_tree_dict(self._tree))
        global_condition_map.update(original_ruleset.condition_map)
        return original_ruleset, global_condition_map

    def create_new_rule(
        self, node_index, tree_dict, condition_set=None, logit_score=None, weights=None, tree_index=None
    ):
        """Creates a new rule with all the information in the parameters

        :param node_index: the index of the leaf node

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param condition_set: set of :class:`rulecosi.rule.Condition` objects
        of the new rule

        :param logit_score: logit_score of the rule (only applies for
        Gradient Boosting Trees)

        :param weights: weight of the new rule

        :param tree_index: index of the tree inside the ensemble

        :return: a :class:`rulecosi.rules.Rule` object
        """
        if condition_set is None:
            condition_set = {}
        value = tree_dict["value"]
        n_samples = tree_dict["n_samples"]

        if weights is not None:
            weight = weights[tree_index]
        else:
            weight = None
        class_dist = (value[node_index] / value[node_index].sum()).reshape((len(self.classes_),))
        # predict y_class_index = np.argmax(class_dist).item()
        y_class_index = np.argmax(class_dist)
        y = np.array([self.classes_[y_class_index]])

        return Rule(
            set(condition_set),
            class_dist=class_dist,
            ens_class_dist=class_dist,
            logit_score=logit_score,
            y=y,
            y_class_index=y_class_index,
            n_samples=n_samples[node_index],
            classes=self.classes_,
            weight=weight,
        )

    def recursive_extraction(self, tree_dict, tree_index=0, node_index=0, condition_map=None, condition_set=None):
        """Recursive function for extracting a ruleset from a tree

        :param tree_dict: a dictionary containing  the information of the
        base_tree (arrays on :class: `sklearn.tree.Tree` class

        :param tree_index: index of the tree in the ensemble

        :param node_index: the index of the leaf node

        :param condition_map: condition_map: dictionary of <condition_id,
        Condition>, default=None Dictionary of Conditions extracted from all
        the ensembles. condition_id is an integer uniquely identifying the
        Condition.

        :param condition_set:  set of :class:`rulecosi.rule.Condition` objects

        :return: array of :class:`rulecosi.rules.Rule` objects
        """
        if condition_map is None:
            condition_map = dict()
        if condition_set is None:
            condition_set = set()
        rules = []
        children_left = tree_dict["children_left"]
        children_right = tree_dict["children_right"]
        feature = tree_dict["feature"]
        threshold = tree_dict["threshold"]

        # leaf node so a rule is created
        if children_left[node_index] == children_right[node_index]:
            weights = None
            logit_score = None
            new_rule = self.create_new_rule(node_index, tree_dict, condition_set, logit_score, weights, tree_index)
            rules.append(new_rule)
        else:
            # create condition, add it to the condition_set and get conditions from left and right child
            att_name = self._tree.feature_names_in_[feature[node_index]]
            att_index = self._column_names.index(att_name)
            condition_set_left = copy.copy(condition_set)
            # condition_set_left = copy.copy(condition_set)
            # determine operators

            if "operator_left" and "operator_right" in tree_dict:
                op_left = tree_dict["operator_left"][node_index]
                op_right = tree_dict["operator_right"][node_index]
            else:
                op_left, op_right = self.get_split_operators()

            # -0 problem solution
            split_value = threshold[node_index]
            if abs(split_value) < self.float_threshold:
                split_value = copysign(self.float_threshold, split_value)
                # split_value=0
                # print(split_value)
            new_condition_left = Condition(
                att_index,
                op_left,
                # threshold[node_index],
                split_value,
                att_name,
            )
            condition_map[hash(new_condition_left)] = new_condition_left
            # condition_set_left.add(hash(new_condition_left))
            condition_set_left.add((hash(new_condition_left), new_condition_left))
            left_rules = self.recursive_extraction(
                tree_dict,
                tree_index,
                node_index=children_left[node_index],
                condition_set=condition_set_left,
                condition_map=condition_map,
            )
            rules = rules + left_rules

            condition_set_right = copy.copy(condition_set)
            # condition_set_right = copy.copy(condition_set)
            new_condition_right = Condition(
                feature[node_index],
                op_right,
                # threshold[node_index],
                split_value,
                att_name,
            )
            condition_map[hash(new_condition_right)] = new_condition_right
            # condition_set_right.add(hash(new_condition_right))
            condition_set_right.add((hash(new_condition_right), new_condition_right))
            right_rules = self.recursive_extraction(
                tree_dict,
                tree_index,
                node_index=children_right[node_index],
                condition_set=condition_set_right,
                condition_map=condition_map,
            )
            rules = rules + right_rules
        return rules


class RuleExtractorFactory:
    """Factory class for getting an implementation of a BaseRuleExtractor"""

    def get_rule_extractor(tree, column_names, classes, X, y, float_threshold):
        """

        :param tree: Tree object, default = None

        :param column_names: array of string, default=None Array of strings
        with the name of the columns in the data. This is useful for
        displaying the name of the features in the generated rules.

        :param classes: ndarray, shape (n_classes,)
            The classes seen when fitting the ensemble.

        :param X: array-like, shape (n_samples, n_features)
         The training input samples.

        :return: A BaseRuleExtractor class implementation instantiated object
        to be used for extracting rules from trees
        """
        if isinstance(tree, DecisionTreeClassifier):
            return DecisionTreeRuleExtractor(tree, column_names, classes, X, y, float_threshold)
        elif isinstance(tree, J48graft):
            return J48graftRuleExtractor(tree, column_names, classes, X, y, float_threshold)
