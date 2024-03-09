import argparse
import pickle
import re
from collections import Counter

from rerx.rule import Condition, RuleSet


def load_object(input_path: str):
    with open(input_path, "rb") as f:
        return pickle.load(f)


def convert_one_hot_rules(rule_str):
    # Regular expression to find one-hot encoded attributes
    one_hot_pattern = re.compile(r'(\w+=".+?")')

    # Find all one-hot encoded attributes
    one_hot_attrs = one_hot_pattern.findall(rule_str)

    print(one_hot_attrs)
    for attr in one_hot_attrs:
        # Check if the attribute is less than a certain value
        less_than_match = re.search(attr + r"} \\le ([0-9\.]+)", rule_str)
        if less_than_match:
            # Convert to the format x != a
            a = attr.split("=")[1].strip('"')
            rule_str = rule_str.replace(
                attr + r"} \le " + less_than_match.group(1), attr.split("=")[0] + "} \\neq \\textit{" + a + "}"
            )
            continue
        less_than_match = re.search(attr + "} < ([0-9\.]+)", rule_str)
        if less_than_match:
            # Convert to the format x != a
            a = attr.split("=")[1].strip('"')
            rule_str = rule_str.replace(
                attr + r"} < " + less_than_match.group(1), attr.split("=")[0] + "} \\neq \\textit{" + a + "}"
            )
            continue

        # Check if the attribute is greater than or equal to a certain value
        greater_than_match = re.search(attr + r"} \\ge ([0-9\.]+)", rule_str)
        if greater_than_match:
            # Convert to the format x = a
            a = attr.split("=")[1].strip('"')
            rule_str = rule_str.replace(
                attr + r"} \ge " + greater_than_match.group(1), attr.split("=")[0] + "} = \\textit{" + a + "}"
            )
            continue
        greater_than_match = re.search(attr + "} > ([0-9\.]+)", rule_str)
        if greater_than_match:
            # Convert to the format x = a
            a = attr.split("=")[1].strip('"')
            rule_str = rule_str.replace(
                attr + r"} > " + greater_than_match.group(1), attr.split("=")[0] + "} = \\textit{" + a + "}"
            )
            continue

    return rule_str


def main(args):
    ruleset_path = args.ruleset_path
    ruleset: RuleSet = load_object(ruleset_path)
    if not isinstance(ruleset, RuleSet):
        for r in ruleset:
            r.A = [(c[0], Condition(c[1].att_index, c[1].op, c[1].value, c[1].att_name)) for c in r.A]
        ruleset = RuleSet(ruleset=ruleset)

    conditions = [cond for rule in ruleset for cond in rule.A]
    condition_counts = {}
    for cond in conditions:
        condition_counts[cond] = condition_counts.get(cond, 0) + 1
    sorted_conditions = sorted(condition_counts.keys(), key=lambda x: condition_counts[x], reverse=True)
    for rule in ruleset:
        rule.A = list(rule.A)
        rule.A.sort(key=lambda x: condition_counts[x], reverse=True)

    latex_style = ruleset.to_latex()
    print(latex_style)
    converted_rules = [convert_one_hot_rules(rule) for rule in latex_style.split("\n")]

    # print each converted rule
    for rule in converted_rules:
        print(rule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ruleset_path")
    args = parser.parse_args()
    main(args)
