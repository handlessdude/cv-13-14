import numpy as np
import cv2
import pandas as pd
import glob
from sklearn.cluster import KMeans
import tqdm
import pickle
from sklearn.neighbors import NearestNeighbors


TRAIN_SET_PATH = './train/*.jpg'
DB_SET_PATH = './db/*'
DESCRIPTORS_LIMIT = int(2e5)
N_CLUSTERS = 1024
N_NEIGHBOURS = 5


KMEANS_MODEL_PATH = 'kmeans_dump_{n_clusters}.sav'.format(n_clusters=N_CLUSTERS)
KNN_MODEL_PATH = 'knn_dump_{n_clusters}.sav'.format(n_clusters=N_CLUSTERS)
DB_PATH = 'db_{n_clusters}.csv'.format(n_clusters=N_CLUSTERS)

# создаем базу дескрипторов (с ключевыми точками) для обучения

descriptors = None

sift = cv2.SIFT_create()

for i in tqdm.tqdm(glob.glob(TRAIN_SET_PATH)):
    gs = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
    kp, des = sift.detectAndCompute(gs, None)
    if des is None:
        continue
    if descriptors is None:
        descriptors = des
    else:
        descriptors = np.append(descriptors, des, axis=0)

print(descriptors.shape)  # (186062, 128)

# кластеризация
model = KMeans(n_clusters=N_CLUSTERS, n_init='auto')
model.fit(descriptors[np.random.choice(descriptors.shape[0], DESCRIPTORS_LIMIT, replace=False), :])

# сохранение модели
open(KMEANS_MODEL_PATH, 'wb').write(pickle.dumps(model))


# 5. индексация базы
def get_embedding(img_gs: np.ndarray):
    kp, des = sift.detectAndCompute(img_gs, None)
    if des is None:
        return None
    hist = np.zeros(N_CLUSTERS, dtype='float')
    for i in model.predict(des):
        hist[i] += 1

    hist /= N_CLUSTERS
    return hist

ids = []
imgsPath = []
embeddings = []
for img_path in tqdm.tqdm(glob.glob(DB_SET_PATH)):
    idx = img_path.split('/')[-1][0:12]
    img_gs = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    embedding = get_embedding(img_gs)
    if embedding is not None:
        ids.append(idx)
        imgsPath.append(img_path)
        embeddings.append(embedding)

# сохраняем db в .csv
db = pd.DataFrame(data={
    'id': ids,
    'img_path': imgsPath,
    'embedding': embeddings,
})
open(DB_PATH, 'wb').write(pickle.dumps(db))

# сохраняем db в .csv
X = np.empty((len(db['embedding']), N_CLUSTERS))
for i in range(len(db['embedding'])):
    X[i] = np.array(db['embedding'][i], dtype='float').flatten()
knn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric='cosine', algorithm='brute', n_jobs=-1).fit(X)

open(KNN_MODEL_PATH, 'wb').write(pickle.dumps(knn))
