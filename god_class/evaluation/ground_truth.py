import os

import pandas as pd

from clustering.Silhouette import get_paths_and_names


def match(n, k):
    for a in k:
        if a in n.lower():
            return a
    return "none"


def fv_m(csv):
    return pd.read_csv(csv, index_col=0).sort_values('method_name')['method_name'].tolist()


def imp_gt(methods, keywords):
    gt = {}
    for m in methods:
        k = match(m, keywords)

        if k in gt:
            gt[k] = gt[k] + [m]
        else:
            gt[k] = [m]

    return gt


def df_csv(dir, df, name):
    if not os.path.exists(dir):
        os.makedirs(dir)

    df.to_csv(dir + '/' + name + ".csv")


def sort_column_labels(labels):
    first = [labels.pop(labels.index('method_name'))]
    labels.sort()

    return first + labels


def kkk(f):
    c = open(f).readlines()
    return list(map(lambda x: str(x).replace("\n", ""), c))


# pass keyword list file and get the list of keywords


def gt_to_df(gt, kws):
    df = pd.DataFrame(columns=['method_name'] + kws)
    for key, value in gt.items():
        for v in value:
            df = df.append({'method_name': v, key: 1}, ignore_index=True, sort=-1)

    df = df.reindex(columns=sort_column_labels(df.columns.tolist()))
    df = df.fillna(0)
    df[[col for col in df.columns if col != 'method_name']] = df[
        [col for col in df.columns if col != 'method_name']].astype('int')

    return df


def gt_m(
        files=None,
        kws=None):
    [print("\t- " + os.path.relpath(file[0])) for file in files]

    for p, n in files:
        methods = fv_m(p)
        ground_truth = imp_gt(methods, kws)

        df_csv(gt_dir, gt_to_df(ground_truth, kws), n)

    return gt_dir


gt_dir = "/Users/jq/Documents/Assignments/IM/god_class/evaluation/res/ground_truth"

gt_m(get_paths_and_names("/Users/jq/Documents/Assignments/IM/god_class/feature_vectors"),
     kkk("/Users/jq/Documents/Assignments/IM/god_class/res/keywords.txt"))
