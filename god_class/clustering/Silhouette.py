import os

import pandas as pd
from sklearn.cluster import KMeans, hierarchical
from sklearn.metrics import silhouette_score


def fv_n_m(csv):
    df = pd.read_csv(csv, index_col=0)

    return df['method_name'].values, df.drop(['method_name'], axis=1).values


# read the csv and drop the column method name
# return all values

def cluster_df(clids, methods):
    data = [{'cluster_id': k, 'method_name': v} for k, v in zip(clids, methods)]

    df = pd.DataFrame(data, columns=['cluster_id', 'method_name'])
    # df.sort_values(['cluster_id', 'method_name'], inplace=True, ascending=True)
    df.set_index('method_name', inplace=True)

    return df


# create a df for all the cluster with cluster id and method names


def k_algo(v, n):
    return KMeans(n_clusters=n, random_state=10).fit_predict(v)


# kmeas algo


def imp_k(csv, n):
    m, v = fv_n_m(csv)
    cluster_ids = k_algo(v, n)
    return cluster_df(cluster_ids, m)


# read a csv and give back the cluster name for each method

def a_algo(v, n):
    return hierarchical.AgglomerativeClustering(n_clusters=n).fit_predict(v)


# agglomerative algo

def imp_a(csv, n):
    m, v = fv_n_m(csv)
    cluster_ids = a_algo(v, n)
    return cluster_df(cluster_ids, m)


# get the method name = m and value = v and then
# read a csv and give back the cluster name for each method

def fv_v(csv):
    return pd.read_csv(csv, index_col=0).sort_values('method_name').drop(['method_name'], axis=1).values


# get a df value with no method name but sort by method name


def c_name(c, name):
    for p, n in c:
        if n == name:
            return p

    return None


def cl_to_df(c, sort_field):
    return pd.read_csv(c, index_col=0).sort_values(sort_field).cluster_id.values


# read cluster file and give back the dict of cluster values and id


def silhouette(f, c):
    for path, name in f:
        X = fv_v(path)
        cl = c_name(c, name)
        if cl:
            labels = cl_to_df(cl, 'method_name')
            print(name + ' ', silhouette_score(X, labels))


def find_k_m(f, n):
    for path, name in f:
        print(name)
        print('cluster\tK-Means\t Agglomerative')
        [print('\t\t'.join(m)) for m in find_k_s(fv_v(path), n + 1)]


def find_k_s(f, n):
    metrics = []
    for i in range(2, n):
        metrics.append((
            str(i),
            str(round(silhouette_score(f, k_algo(f, i)), 2)),
            str(round(silhouette_score(f, a_algo(f, i)), 2))
        ))
    return metrics


def d(p):
    if not p.endswith('/'):
        return '/'
    return ''


def get_paths_and_names(path):
    return list(zip(
        [path + d(path) + file for file in os.listdir(path)],
        [file.replace('.csv', '') for file in os.listdir(path)]))


#
print(imp_k("/Users/jq/Documents/Assignments/IM/god_class/CoreDocumentImpl.csv", 5))
print(imp_a("/Users/jq/Documents/Assignments/IM/god_class/feature_vectors/CoreDocumentImpl.csv", 5))

print(find_k_s(fv_v("CoreDocumentImpl.csv"), 5))

find_k_m(get_paths_and_names("/Users/jq/Documents/Assignments/IM/god_class/feature_vectors"), 80)

print(silhouette(get_paths_and_names("/Users/jq/Documents/Assignments/IM/god_class/feature_vectors"),
                 get_paths_and_names("/Users/jq/Documents/Assignments/IM/god_class/clusters")))

#
# #
# imp_k("feature_vectors/CoreDocumentImpl.csv",5).to_csv("/Users/jq/Documents/Assignments/IM/god_class/CoreDocumentImpl")
# imp_k("feature_vectors/DTDGrammar.csv",5).to_csv("/Users/jq/Documents/Assignments/IM/god_class/DTDGrammar")
# imp_k("feature_vectors/XIncludeHandler.csv",5).to_csv("/Users/jq/Documents/Assignments/IM/god_class/XIncludeHandler")
# imp_k("feature_vectors/XSDHandler.csv",5).to_csv("/Users/jq/Documents/Assignments/IM/god_class/XSDHandler")
