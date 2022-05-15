import tensorflow as tf
import os
from cv2 import cv2
import glob
import numpy as np
from retinaface import RetinaFace

from tqdm import trange, tqdm

test_query_path = "../test_data_mod/query"
test_gallery_path = "../test_data_mod/gallery"

train_data_path = '../lfw_mod'


model = RetinaFace.build_model()

def find_faces(img):
    faces = RetinaFace.extract_faces(img_path=img, model=model, align=True)
    if len(faces) == 0:
        tmp_img = cv2.imread(img)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        print('no face found')
    else:
        print('found face count', len(faces))
        tmp_img = faces[0]
    return tmp_img

def image_convert(img):
    tmp_img = cv2.imread(img)
    tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
    shapes = np.array(tmp_img.shape, dtype=np.float)[:-1]
    big_side = max(shapes)
    new_side = np.ceil(big_side / 2) * 2
    diff = new_side - shapes
    half_diff = diff // 2
    top, left = diff - half_diff
    bottom, right = half_diff
    tmp_img = cv2.copyMakeBorder(tmp_img,
                                 int(top),
                                 int(bottom),
                                 int(left),
                                 int(right),
                                 cv2.BORDER_CONSTANT) 
    # cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE
    return tmp_img


def extract_faces_from_file_list(file_list):
    for img_path in tqdm(file_list):
        tmp_img = find_faces(img_path)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, tmp_img)

def convert_from_file_list(file_list):
    for img_path in tqdm(file_list):
        tmp_img = image_convert(img_path)
        tmp_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path, tmp_img)

query_file_list = glob.glob(os.path.join(test_query_path, "*.jpg"))
gallery_file_list = glob.glob(os.path.join(test_gallery_path, "*.jpg"))
# extract_faces_from_file_list(query_file_list)
# extract_faces_from_file_list(gallery_file_list)
convert_from_file_list(query_file_list)
convert_from_file_list(gallery_file_list)

gallery_file_list = glob.glob(os.path.join(train_data_path, "*/*.jpg"))
convert_from_file_list(gallery_file_list)


