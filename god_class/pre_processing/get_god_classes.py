import os
import sys

import javalang
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def get_god_classes(path):
    import os
    all_files = []
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if name.endswith(".java"):
                with open(root + "/" + name) as f:
                    tree = javalang.parse.parse(f.read())
                    for path, node in tree.filter(javalang.parser.tree.ClassDeclaration):
                        all_files.append((node.name, len(node.methods), root + "/" + name))

    all_methods = [x[1] for x in all_files]

    mean = np.mean(all_methods)
    std = 6 * np.std(all_methods)

    all_files = filter(lambda x: x[1] > mean + std, all_files)

    df = pd.DataFrame(all_files, columns=["class_name", "method_num", "path"])

    df_to_csv(df)

    return df


def df_to_csv(god_classes):
    god_classes.to_csv('god_class.csv')


if __name__ == '__main__':
    get_god_classes("res/xerces2-j")
