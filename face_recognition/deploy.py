import tensorflow as tf
import os
import requests
import json
import numpy as np
from cv2 import cv2
import glob
#from PIL import Image
import pickle as pk
from tqdm import trange, tqdm

model_path = "cosface.h5"
model_path = "../tmp_models/cos_best_5"
test_query_path = "../test_data/query_mod"
test_gallery_path = "../test_data/gallery_mod"
sizes=112
pickle_name = 'to_submit_2'

#### CHECK IF DATA IS PRESENT
#assert os.path.isdir(model_path)

to_submit = dict()
to_submit['groupname'] = "chachacha"

def submit(results, url="http://"):
    res = json.dumps(results)
    response = requests.post(url, res)
    try:
        result = json.loads(response.text)
        print(f"accuracy is {result['results']}")
        return result
    except json.JSONDecodeError:
        print(f"ERROR: {response.text}")
        return None



### LOAD TRAINED MODELS
model = tf.keras.models.load_model(model_path)

### PREFORM PREDICTIONS OVER GRANT/FIRM FEATURES
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


query_file_list = glob.glob(os.path.join(test_query_path, "*.jpg"))
gallery_file_list = glob.glob(os.path.join(test_gallery_path, "*.jpg"))
query_features = extract_features_from_file_list(query_file_list)
gallery_features = extract_features_from_file_list(gallery_file_list)


### EVALUATION
max = 0
min = 2000000
results = dict()
for i, query_sample in enumerate(query_features):
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
    results[os.path.basename(query_file_list[i])] = gallery_list[:10]



to_submit['images'] = results

with open(rf"{pickle_name}.pickle", "wb") as output_file:
    pk.dump(to_submit, output_file)
#scores = submit(to_submit)


