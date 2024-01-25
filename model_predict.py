import gzip
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time

IMDB_PATH = "data/IMDB.csv"


def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    df = df.iloc[100:]
    texts = df['review'].tolist()
    labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]
    return texts, labels


def gzip_predict(request, k):
    texts, labels = load_imdb_data(IMDB_PATH)

    x1 = request[0]
    distances_from_x1 = []
    start = time.time()
    for x2 in texts:
        Cx1 = len(gzip.compress(x1.encode()))
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = "".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))

        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distances_from_x1.append(ncd)
    sorted_idx = np.argsort(np.array(distances_from_x1))
    top_k_class = np.array(labels)[sorted_idx[:k]].tolist()
    predict_class = max(set(top_k_class), key=top_k_class.count)
    end = time.time()

    return predict_class, end-start


def bert_predict(request):
    pass