import os
import cv2
import random
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt

categories = ['table', 'list', 'title', 'text', 'figure']

def get_textImg_dicts(images):

    dataset_dicts = []
    for i, (_, image) in enumerate(images.items()):
        record = {}
        filename = image["file_name"]
        record["file_name"] = filename

        print(filename)

        record["height"] = image['size']['height']
        record["width"] = image['size']['width']

        annos = image['annotations']
        objs = []
        for anno in annos:
            x = anno["bndbox"]['xmin']
            y = anno["bndbox"]['ymin']
            w = anno["bndbox"]['xmax'] - x
            h = anno["bndbox"]['ymax'] - y

            obj = {
                "bbox": [x, y, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": categories.index(anno['name']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    print('textImg dits lenght: ', len(dataset_dicts))
    print('len of images :', len(images))
    return dataset_dicts


def draw_textImg_dicts(dataset_dicts, smp_num, textImg_metadata):

    for d in random.sample(dataset_dicts, smp_num):
        print('visualize...')
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=textImg_metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        img = vis.get_image()[:, :, ::-1]
        cv2.imwrite('test.jpg', img)
    print('visualize finished')

def draw_predImg_dicts(dataset_dicts, smp_num, textImg_metadata, predictor):

    for d in random.sample(dataset_dicts, smp_num):

        img = cv2.imread(d["file_name"])
        outputs = predictor(img)

        visualizer = Visualizer(img[:, :, ::-1], metadata=textImg_metadata, scale=0.5)
        vis = visualizer.draw_instance_predictions(outputs['instances'].to('cpu'))
        img = vis.get_image()[:, :, ::-1]
        plt.imshow(img)
        plt.show()
