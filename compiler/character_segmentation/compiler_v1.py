import os
import random
import shutil

import xmltodict
from string import ascii_uppercase, digits

"""Constants"""

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET = os.path.join(PROJECT_ROOT, 'datasets', 'character_segmentation', 'raw')

LABELS_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'character_segmentation', 'labels')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'datasets', 'character_segmentation', 'images')

LABELS_DIR_TRAIN = os.path.join(LABELS_DIR, 'train')
LABELS_DIR_VAL = os.path.join(LABELS_DIR, 'val')
IMAGES_DIR_TRAIN = os.path.join(IMAGES_DIR, 'train')
IMAGES_DIR_VAL = os.path.join(IMAGES_DIR, 'val')

chars = digits + ascii_uppercase

char2idx = {c:i for i,c in enumerate(chars)}
idx2char = {i:c for i,c in enumerate(chars)}

"""Create File Structure"""

if os.path.exists(LABELS_DIR):
    shutil.rmtree(LABELS_DIR)
if os.path.exists(IMAGES_DIR):
    shutil.rmtree(IMAGES_DIR)

if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

if not os.path.exists(LABELS_DIR_TRAIN):
    os.makedirs(LABELS_DIR_TRAIN)
if not os.path.exists(LABELS_DIR_VAL):
    os.makedirs(LABELS_DIR_VAL)

if not os.path.exists(IMAGES_DIR_TRAIN):
    os.makedirs(IMAGES_DIR_TRAIN)
if not os.path.exists(IMAGES_DIR_VAL):
    os.makedirs(IMAGES_DIR_VAL)

"""Methods"""

def clean_data(labels_dir, images_dir):
    label_ids = {os.path.basename(f).split('.')[0] for f in os.listdir(labels_dir) if f.endswith('.xml')}
    image_ids = {os.path.basename(f).split('.')[0] for f in os.listdir(images_dir) if f.endswith('.png')}

    for img_id in label_ids - image_ids:  # labels with no image
        os.remove(os.path.join(labels_dir, img_id + '.xml'))

    for img_id in image_ids - label_ids:  # images with no label
        os.remove(os.path.join(images_dir, img_id + '.png'))


def convert_file(file_path):
    """
    Converts an XML file into a dictionary

    Args:
        file_path: path to the XML file

    Returns:
        Dictionary composed of key:values from XML file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        xml = f.read()
        annotation = xmltodict.parse(xml)
    return annotation['annotation']

def parse_annotations(annotations, size):
    """
    Takes dictionary of annotations and converts it into a format readable by YOLO8

    Args:
        annotations: list containing the annotation of each character
        size: dictionary describing the size of the image

    Returns:
        String representation of the annotations
    """
    string_rep = ''
    for item in annotations:
        character = item['name']
        character = char2idx[character]
        labels = item['bndbox']

        img_height = int(size['height'])
        img_width = int(size['width'])

        xmin = int(labels['xmin'])
        ymin = int(labels['ymin'])
        xmax = int(labels['xmax'])
        ymax = int(labels['ymax'])

        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height

        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        string_rep += f'{character} {x_center:2f} {y_center:2f} {width:2f} {height:2f}\n'
    return string_rep

def move_file_pair(label_file: str, dest_labels: str, dest_images: str):
    """
    Moves a pair of files from raw data directory to sorted directories

    Precondition: Label file has a matching image file that both exist

    Args:
        label_file: the target label file
        dest_labels: the directory to move the label file to
        dest_images: the directory to move the image file to
    """
    img_file = label_file.split('.')[0] + '.png'
    shutil.copy(os.path.join(LABELS_DIR, label_file), os.path.join(dest_labels, label_file))
    shutil.copy(os.path.join(TARGET, "images", img_file), os.path.join(dest_images, img_file))

"""Main function"""


def main():
    """
    Walks through annotations directory and makes a new directory that contains annotations that match YOLO8 annotation style
    """
    RAW_LABELS_DIR = os.path.join(TARGET, "annotations")
    RAW_IMAGES_DIR = os.path.join(TARGET, "images")
    clean_data(RAW_LABELS_DIR, RAW_IMAGES_DIR)

    files = [os.path.join(RAW_LABELS_DIR,f) for f in os.listdir(RAW_LABELS_DIR) if os.path.isfile(os.path.join(RAW_LABELS_DIR, f))]

    for file in files:
        img_id = os.path.basename(file).split('.')[0]

        annotations = convert_file(file)
        content = parse_annotations(annotations['object'], annotations['size'])

        with open(os.path.join(LABELS_DIR, img_id+'.txt'), 'w', encoding='utf-8') as f:
            f.write(content)

    train_size = int(len(files) * .8)
    val_size = len(files) - train_size

    print('Training size:', train_size)
    print('Validation size:', val_size)

    files = [os.path.basename(f).split('.')[0] + '.txt' for f in files]

    random.shuffle(files)

    for label_file in files[:train_size]:
        move_file_pair(label_file, LABELS_DIR_TRAIN, IMAGES_DIR_TRAIN)

    for label_file in files[train_size:]:
        move_file_pair(label_file, LABELS_DIR_VAL, IMAGES_DIR_VAL)

if __name__ == '__main__':
    main()