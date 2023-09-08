import sklearn.cluster
import sklearn.metrics
import numpy as np
import pandas as pd
import time
import json

def dbscan(eps, min_samples, data_path, output_name, distance_fn):
    #Load pose data
    start_time = time.time()
    df = pd.read_csv(data_path)
    data_array = df['keypoints']

    #Clustering
    print(f"Started DBSCAN(eps={eps}, min_samples={min_samples})...", end = '')
    dbscan = sklearn.cluster.DBSCAN(eps=eps, min_samples=min_samples, metric = distance_fn)
    labels = dbscan.fit_predict(data_array)
    print("Finished!")
    #save results
    
    silhouette_score = sklearn.metrics.silhouette_score(data_array, labels)

    groups = {}
    for i,x in enumerate(labels):
        if x in groups:
            groups[int(x)].append(df[i]['name'])
        else:
            groups[int(x)] = [df[i]['name']]

    results = {}
    results['duration'] = start_time - time.time()
    results['params'] = {'eps' : eps, 'min_samples' : min_samples}
    results['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
    results['n_noise'] = list(labels).count(-1)
    results['groups'] = groups
    results['silhouette_score'] = silhouette_score

    json_path = output_name + '.json'
    with open(json_path, 'w') as json_file:
        json.dump(results, json_file)

    csv_path = output_name + '.csv'
    result_df = df[['name']]
    result_df['labels'] = labels
    result_df.to_csv(csv_path)
    
    print(f"Saved results in {json_path} and {csv_path}.")
    print(f" Silhouette score : {results['silhouette_score']}")
    print(f" Duration : {results['duration']}")
    print(f" Cluster count : {results['n_clusters']}")
    print(f" Noise count : {results['n_noise']}")