from dbscan import dbscan

import datetime

DATA_PATH = '../../datas/pose_data.csv'
OUTPUT_NAME = f'../../datas/cluster_results/{datetime.datetime.today().strftime("%m%d%H%M")}'
dbscan(eps = 0.7, min_samples = 3, 
       data_path = DATA_PATH, output_name = '../../da')

