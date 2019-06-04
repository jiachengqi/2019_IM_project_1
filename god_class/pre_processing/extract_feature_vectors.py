import os
import sys

import javalang
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pre_processing.get_god_classes import get_god_classes

fv_dir = "/Users/jq/Documents/Assignments/IM/god_class/"


def extract_feature_vectors(god_classes):
    class_names = god_classes.class_name.tolist()
    feature_vectors = {}
    for source_path in god_classes.path.tolist():
        with(open(source_path, 'r')) as javasrc:
            tree = javalang.parse.parse(javasrc.read())

            for path, node in tree.filter(javalang.tree.ClassDeclaration):

                if node.name in class_names:
                    feature_vectors[node.name] = get_all(node)
                    df_to_csv_all(fv_dir, feature_vectors[node.name], node.name)

    fv_dir_n = os.path.abspath(fv_dir)

    return fv_dir_n


def get_all(node):
    fields = get_fields(node)
    methods = get_methods(node)

    fv_dict = {}
    for method in node.methods:
        fv = get_feature_vectors(method, fields, methods)
        add_v(fv, fv_dict)

    df = dict_df(fv_dict)
    return df


def get_fields(node):
    return [field.declarators[len(field.declarators) - 1].name for field in node.fields]


def get_methods(node):
    return np.unique([method.name for method in node.methods])


def get_feature_vectors(method, fields, methods):
    row = {'method_name': method.name}
    a_f = get_fbym(method, fields)
    a_m = get_fbym(method, methods)

    for field in list(a_f) + list(a_m):
        row[field] = 1

    return row


def add_v(fv, fv_dict):
    if fv['method_name'] in fv_dict:
        fv_dict[fv['method_name']].update(fv)
    else:
        fv_dict[fv['method_name']] = fv


def get_fbym(method, fields):
    return get_m(method, fields, javalang.parser.tree.MemberReference)


def get_m(method, arr, tree_filter):
    return np.unique([node.member for path, node in method.filter(tree_filter) if node.member in arr])


def get_mbym(method, methods):
    return get_m(method, methods, javalang.parser.tree.MethodInvocation)


def dict_df(vec_dict):
    df = pd.DataFrame([vec_dict.get(x) for x in vec_dict.keys()])
    df = df.fillna(0)
    df[[x for x in df.columns if x != 'method_name']] = df[
        [x for x in df.columns if x != 'method_name']].astype('int')

    return df


def df_to_csv_all(directory, df, classname):
    print(directory, df, classname)
    df.to_csv(directory + '/' + classname + ".csv")


extract_feature_vectors(get_god_classes("res/xerces2-j/src"))
