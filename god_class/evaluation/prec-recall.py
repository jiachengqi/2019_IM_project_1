import pandas as pd

from clustering.Silhouette import get_paths_and_names


def get_gt(g):
    df = pd.read_csv(g, index_col=0)
    cols = [el for el in df.columns.tolist() if el != 'method_name']
    gt = []

    for c in cols:
        v = df[df[c] == 1].method_name.tolist()
        if v:
            gt.append(v)
    return gt


def cl_dict(cl):
    df = pd.read_csv(cl, index_col=0)

    clusters = []
    for c in df.cluster_id.unique():
        v = df[df.cluster_id == c]
        clusters.append(v.index.values.tolist())
    return clusters


def intra_pair_m(l):
    return [j for i in l for j in intra_pair(i)]


def intra_pair(l):
    return [(i, j) for i in l for j in l if i != j]


def merge(x, y):
    cc = {}
    for a in x:
        for b in y:
            if a[1] == b[1]:
                cc[a[1]] = [a[0], b[0]]
    return cc


def p_n_r(c, gts):
    for k, v in merge(c, gts).items():
        cl = cl_dict(v[0])
        gt = get_gt(v[1])

        a = set(intra_pair_m(cl))
        b = set(intra_pair_m(gt))

        p, r = computation(a, b)
        print('\t- ' + k + ': p=' + str(p) + ', r=' + str(r))


def computation(ip_dk, ip_g):
    interception = ip_dk.intersection(ip_g)
    return round(len(interception) / len(ip_dk), 2), round(len(interception) / len(ip_g), 2)


p_n_r(get_paths_and_names("/Users/jq/Documents/Assignments/IM/god_class/clusters"),
      get_paths_and_names("/Users/jq/Documents/Assignments/IM/god_class/evaluation/res/ground_truth"))
