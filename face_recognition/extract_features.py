import tensorflow as tf
import os
from cv2 import cv2
import glob
import numpy as np
from retinaface import RetinaFace

from tqdm import trange, tqdm

model_path = "cosface.h5"
sizes=112
test_gallery_path = "../test_data_mod/demo_gallery"


### LOAD TRAINED MODELS
model = tf.keras.models.load_model(model_path)


def decode_img(img, img_height=sizes, img_width=sizes):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shapes = np.array(img.shape, dtype=np.float)[:-1]
    big_side = max(shapes)
    new_side = np.ceil(big_side / 2) * 2
    diff = new_side - shapes
    half_diff = diff // 2
    top, left = diff - half_diff
    bottom, right = half_diff
    img = cv2.copyMakeBorder(img, int(top), int(bottom), int(left), int(right), cv2.BORDER_CONSTANT)
    return tf.image.resize(img, [img_height, img_width])

def normalize_image(img):
    return (img - 127.5) / 128

@tf.function
def extract_features(images):
    features = model(images, training=False)
    return features

def extract_features_from_file_list(file_list):
    features = []
    for i in trange(0, len(file_list), 128):
        images = []
        tmp_img_list = file_list[i:i+128]
        #print(tmp_img_list)
        for img_path in tqdm(tmp_img_list):
            tmp_img = normalize_image(decode_img(img_path))
            images += [tmp_img]
        features += [extract_features(tf.stack(images))]
    print(len(features))
    features = tf.concat(features, axis=0)
    return features



gallery_file_list = glob.glob(os.path.join(test_gallery_path, "*.jpg"))
features = extract_features_from_file_list(gallery_file_list)

np.save('features.npy', features)
np.save('file_names.npy', np.array(gallery_file_list))

