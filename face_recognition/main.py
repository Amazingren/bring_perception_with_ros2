import time
import numpy as np
import cv2
import tqdm
import os
import glob
import tensorflow_addons as tfa
import tensorflow as tf
import random as rnd
# import wandb
from scipy import spatial

# Suppress warnings
tf.get_logger().setLevel('ERROR')

# allow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print('Gpus:', gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


datafolder = "../lfw_mod"
checkpoint_folder = "../tmp_models"
save_dir = 'cos_best_5'
model_path = "cosface.h5"

sizes = 112

LEARNING_RATE = 0.00001
N_EPOCHS = 100
BATCH_SIZE = 32
DEBUG = True
PROJECT = "Face Identification"
ENTITY = 'unitn-mhug'
GROUP = "Triplet"
NAME = "final-model-longer"

config = {
    "learning rate": LEARNING_RATE,
    "epochs": N_EPOCHS,
    "batch size": BATCH_SIZE,
}


#wandb.init(project=PROJECT, entity=ENTITY, sync_tensorboard=False, group=GROUP,
#           name=NAME, config=config, dir="../wandb")


'''
Create the list of paths (only folders with more than one image)
'''
dir_list = os.listdir(datafolder)
class_dict = dict()
class_dict_positives = dict()
class_dict_negatives = dict()
count = 0
for d in dir_list:
    file = glob.glob(os.path.join(datafolder, d, "*"))
    class_dict[d] = file
    count += len(file)
    if (len(file) > 1):
        class_dict_positives[d] = file
    else:
        class_dict_negatives[d] = file

# train/test

def reverse_dict(img_dict):
    """
    reverse the dictionary keys/values -> values/keys
    Args:
        img_dict: some dictionary

    Returns:
        reversed dictionary

    """
    d = dict()
    for k, v in img_dict.items():
        for i in v:
            d[i] = k
    return d

class_list = list(class_dict_positives.keys())

class_ids = dict()
for i, n in enumerate(class_list):
    class_ids[n] = i

"""
Generate query and gallery in test data
gallery:
    key: [path to file]
    value: [class name]
query:
    key: [path to file]
    value: [class name]
"""

rnd.shuffle(class_list)
training_list = class_list[0:int(len(class_list)*0.8)]
test_list = class_list[int(len(class_list)*0.8):]
query = dict()
gallery = dict()

my_test_dict = {k: v for k, v in class_dict_positives.items() if k in test_list}
my_test_dict_inverse = reverse_dict(my_test_dict)
test_image_list = list(my_test_dict_inverse.keys())
rnd.shuffle(test_image_list)

n_query = 100
count = 0
for test_img in test_image_list:
    if count < n_query:
        query[test_img] = my_test_dict_inverse[test_img]
        count += 1
    else:
        gallery[test_img] = my_test_dict_inverse[test_img]
negative_inverse = reverse_dict(class_dict_negatives)
for k, v in negative_inverse.items():
    gallery[k] = v



# Data loader
def decode_img(img, img_height=sizes, img_width=sizes):
    """
    Decodes an image starting from the file path
    Args:
        img:
        img_height:
        img_width:

    Returns:

    """
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
    """
    Normalize an image
    Args:
        img:

    Returns:

    """
    return img / 255

def get_images(batch_names, positive_dict, class_dict):
    """
    Generate the batch of images
    Args:
        batch_names:
        positive_dict:
        class_dict:

    Returns:

    """
    images = []
    labels = []
    for i, n in enumerate(batch_names):
        rnd.shuffle(positive_dict[n])
        pos_files = positive_dict[n][:2]
        img1 = normalize_image(decode_img(pos_files[0]))
        img2 = normalize_image(decode_img(pos_files[1]))
        images += [img1]
        images += [img2]
        labels += [class_dict[n], class_dict[n]]
    images = tf.stack(images)
    labels = tf.stack(labels)
    return images, labels


model = tf.keras.models.load_model(model_path)
model.summary()




optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
triplet_loss = tfa.losses.triplet_hard_loss


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        features = model(images, training=True)
        loss = triplet_loss(labels, features)
    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    return loss

@tf.function
def extract_features(images):
    features = model(images, training=False)
    return features


def extract_features_from_file_list(file_list):
    features = []
    for i in range(0, len(file_list), BATCH_SIZE):
        images = []
        tmp_img_list = file_list[i:i+BATCH_SIZE]
        for img_path in tmp_img_list:
            tmp_img = normalize_image(decode_img(img_path))
            images += [tmp_img]
        features += [extract_features(tf.stack(images))]
    features = tf.concat(features, axis=0)
    return features


query_image_list = list(query.keys())
gallery_image_list = list(gallery.keys())
step = 0
best_top5 = 0
for epoch in range(N_EPOCHS):

    """
    Training Step
    """

    rnd.shuffle(training_list)
    for i in tqdm.tqdm(range(0, len(training_list), BATCH_SIZE), desc=f"iterating epoch {epoch+1}"):
        batch_names = training_list[i:i+BATCH_SIZE]
        images, labels = get_images(batch_names, class_dict_positives, class_ids)
        loss = train_step(images, labels)
        step += 1
        #wandb.log({"train/triplet_loss": loss}, step=step)
        # print(f"iter {step}")



    """
    Test Step
    """

    t0 = time.time()
    features_gallery = extract_features_from_file_list(list(gallery.keys())).numpy()
    t_gal = time.time() - t0
    features_query = extract_features_from_file_list(list(query.keys())).numpy()
    t_que = time.time() - t0 - t_gal

    # compute pairwise distance [n_queries x n_galleries]
    pairwise_dist = spatial.distance.cdist(features_query, features_gallery, 'minkowski', p=2.)
    t_pairwise = time.time() - t0 - t_gal - t_que

    # get gallery indexes
    indices = np.argsort(pairwise_dist, axis=-1)
    top1 = []
    top5 = []
    top10 = []
    for i, index in enumerate(query.keys()):
        query_class = query[index]
        gallery_indexes = [gallery_image_list[j] for j in indices[i, :10]]
        gallery_classes = [gallery[j] for j in gallery_indexes]
        if query_class == gallery_classes[0]:
            top1.append(1)
        else:
            top1.append(0)
        if query_class in gallery_classes[:5]:
            top5.append(1)
        else:
            top5.append(0)
        if query_class in gallery_classes[:10]:
            top10.append(1)
        else:
            top10.append(0)
    top1_acc = np.mean(np.asarray(top1))
    top5_acc = np.mean(np.asarray(top5))
    top10_acc = np.mean(np.asarray(top10))
    #wandb.log({"test/top1": top1_acc}, step=step)
    #wandb.log({"test/top5": top5_acc}, step=step)
    #wandb.log({"test/top10": top10_acc}, step=step)
    print(f"epoch {epoch + 1}: top1: {top1_acc} - top5: {top5_acc} - top10: {top10_acc}")


    # Save the model if is good enough
    if top5_acc > best_top5:
        best_top5 = top5_acc
        model.save(os.path.join(checkpoint_folder, save_dir))
        print("Model saved!")


# a = decode_img(class_dict_positives['Kimi_Raikkonen'][0])
# plt.imshow(a.numpy().astype("uint8"))
# plt.show()
