import pickle

import numpy as np
import streamlit as st
from PIL import Image
import torch
import clip

TRAIN_SET_PATH = './train/*.jpg'
DB_SET_PATH = './db/*'
DESCRIPTORS_LIMIT = int(2e5)
N_NEIGHBOURS = 5


KNN_MODEL_PATH = 'knn_dump_14.sav'
DB_PATH = 'db_14.csv'


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


db = pickle.loads(open(DB_PATH, 'rb').read())
knn_model = pickle.loads(open(KNN_MODEL_PATH, 'rb').read())


def get_embedding(img_rgb: np.ndarray):
    image = preprocess(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image)
    return features[0].cpu().numpy()


def main():
    st.title('Image Search')
    up_file = st.file_uploader('Choose file...', type=['jpeg', 'jpg', 'webp', 'png', 'tiff'])
    if up_file is not None:
        st.image(up_file)
        img = Image.open(up_file).convert("RGB")

        predictions = knn_model.kneighbors([get_embedding(img)])
        # print(predictions)  # ([array_of_distances], [array_of_img_indices])
        paths = db.iloc[predictions[1][0]]['img_path'].head(N_NEIGHBOURS).values

        for i in range(N_NEIGHBOURS):
            st.image(Image.open(paths[i]), width=580)


if __name__ == '__main__':
    main()
