import tensorflow as tf
import os
import numpy as np
from cv2 import cv2
import glob
#from PIL import Image
from tqdm import trange, tqdm

model_path = "cosface.h5"
test_query_path = "../test_data/query_mod"
test_gallery_path = "../test_data/gallery_mod"
sizes=112

### LOAD TRAINED MODELS
model = tf.keras.models.load_model(model_path)

### PREFORM PREDICTIONS OVER GRANT/FIRM FEATURES
def decode_img(img, img_height=sizes, img_width=sizes):
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


def extract_feature_image(image):
    tmp_img = normalize_image(decode_img(image))
    return extract_features(tf.expand_dims(tmp_img,axis=0))


gallery_features = np.load('features.npy')
gallery_file_list = np.load('file_names.npy')


### EVALUATION
max = 0
min = 2000000

image = np.zeros((222,222,3))
query_sample = extract_feature_image(image)
query_sample = tf.reshape(query_sample, [1, len(query_sample)])
query_sample_tiles = tf.tile(query_sample, [len(gallery_features), 1])
dists = tf.math.sqrt(tf.reduce_sum(tf.math.square(query_sample_tiles - gallery_features), axis=-1))
if tf.reduce_max(dists) > tf.cast(max, tf.float32):
    max = tf.reduce_max(dists).numpy()
if tf.reduce_min(dists) < tf.cast(min, tf.float32):
    min = tf.reduce_min(dists).numpy()
rank = tf.argsort(
    dists, axis=-1, direction='ASCENDING', stable=False, name=None
)
gallery_list = []
for j, r in enumerate(rank):
    gallery_list.append(os.path.basename(gallery_file_list[r]))
results = gallery_list[:10]


