import pickle
import cv2

import numpy as np
import streamlit as st
from PIL import Image

TRAIN_SET_PATH = './train/*.jpg'
DB_SET_PATH = './db/*'
DESCRIPTORS_LIMIT = int(2e5)
N_CLUSTERS = 1024
N_NEIGHBOURS = 5


KMEANS_MODEL_PATH = 'kmeans_dump_{n_clusters}.sav'.format(n_clusters=N_CLUSTERS)
KNN_MODEL_PATH = 'knn_dump_{n_clusters}.sav'.format(n_clusters=N_CLUSTERS)
DB_PATH = 'db_{n_clusters}.csv'.format(n_clusters=N_CLUSTERS)


sift = cv2.SIFT_create()

kmeans_model = pickle.loads(open(KMEANS_MODEL_PATH, 'rb').read())
knn_model = pickle.loads(open(KNN_MODEL_PATH, 'rb').read())
db = pickle.loads(open(DB_PATH, 'rb').read())


def get_embedding(img_gs: np.ndarray):
    # img_gs = cv2.cvtColor(img_gs, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(img_gs, None)
    if des is None:
      return None
    hist = np.zeros(N_CLUSTERS, dtype='float')
    for i in kmeans_model.predict(des):
      hist[i] += 1

    hist /= N_CLUSTERS
    return hist


def main():
    st.title('Image Search')
    up_file = st.file_uploader('Choose file...', type=['jpeg', 'jpg', 'webp', 'png', 'tiff'])
    if up_file is not None:
        st.image(up_file)
        img = Image.open(up_file).convert("RGB")

        predictions = knn_model.kneighbors([get_embedding(np.array(img))])
        # print(predictions)  # ([array_of_distances], [array_of_img_indices])
        paths = db.iloc[predictions[1][0]]['img_path'].head(N_NEIGHBOURS).values

        for i in range(N_NEIGHBOURS):
            st.image(Image.open(paths[i]), width=580)


if __name__ == '__main__':
    main()
