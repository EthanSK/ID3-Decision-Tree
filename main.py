import pandas as pd
import math


df = pd.read_csv('table.csv', sep=r'\s*,\s*', header=0,
                 encoding='ascii', engine='python')
label_col_name = df.columns[-1]


def ctg_counts(column, data):
    return data[column].value_counts()


def calc_label_uncertainty(label_col):
    counts = label_col.value_counts()
    # print(counts.to_numpy(), len(label_col))
    res = 0
    for ctg in counts.to_numpy():
        # row[-1] has the count of that category for one label category
        x = ctg / len(label_col)
        term = x * math.log2(x)
        res -= term
    return res


tree_print_sym = "    "


def split_node(data, print_tree_mode=False, depth=0):
    # end condition check: if all rows have same label, return label
    if data[label_col_name].nunique() == 1:
        leaf = data[label_col_name].iloc[0]
        if not print_tree_mode:
            print("LEAF NODE: ", leaf)
        else:
            print(tree_print_sym * depth, leaf)
        return leaf
    # first calc the col that gives the highest IG
    H_data = calc_label_uncertainty(data[label_col_name])
    max_IG = -math.inf
    col_max_IG = ''
    if not print_tree_mode:
        print("\n\n======NEW NODE=======")
    for col in data.drop(columns=[label_col_name]):
        if not print_tree_mode:
            print(f"\n----------- IG calculations for {col} --------------")
        counts = data.groupby([col, label_col_name]
                              ).size().reset_index(name="Counts")
        uncertainties = {}
        for row in counts.to_numpy():
            # row[-1] has the count of that category for one label category
            x = row[-1] / ctg_counts(col, data)[row[0]]
            term = x * math.log2(x)
            uncertainties[row[0]] = uncertainties.get(row[0], 0) - term
        H_col = 0
        for key in uncertainties:
            H = uncertainties[key]
            if not print_tree_mode:
                print(f"H_{key} = {H}")
            H_col += H * ctg_counts(col, data)[key] / len(data)
        IG = H_data - H_col
        if not print_tree_mode:
            print(f"\nIG_{col} = {IG}")
        if IG > max_IG:
            max_IG = IG
            col_max_IG = col
    # now split it based on the col with highest IG
    if print_tree_mode:
        print(tree_print_sym * depth, f"-{col_max_IG}-")
    else:
        print("\n\ABOUT TO TRY SPLIT ON:", col_max_IG, "\n")
    for ctg in data[col_max_IG].drop_duplicates():
        new_data = data[data[col_max_IG] == ctg]
        if print_tree_mode:
            print(tree_print_sym * depth, f"{ctg}")
        split_node(new_data, print_tree_mode, depth + 1)


split_node(df)  # prints intermediate steps
# too lazy to implement a proper way to print the tree
split_node(df, print_tree_mode=True)
