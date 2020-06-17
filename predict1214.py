import utils1214 as utils
import json
import os
import time
import random
import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

valPath = "./publaynet_data/test_1214/"
# valjsonPath = "./publaynet_data/val.json"

json_files = os.listdir(valPath)
img_id = 0
for json_file in sorted(json_files):
    if json_file[-4:] != 'json':
        continue
    img_id += 1
    json_path = os.path.join(trainPath, json_file)

    #print(json_path)

    img_path = os.path.join(trainPath, json_file[:-4]+'jpg')

    #print(img_path)

    with open(json_path, 'r') as f:
        img_ann = json.load(f)
        images[str(img_id)] = {'file_name': img_path, 'annotations': img_ann['outputs']['object'], 'size': img_ann['size']}

categories = ['table', 'list', 'title', 'text', 'figure']
print('categories: ', categories)

DatasetCatalog.register("valSet", lambda I = images, P = valPath: utils.get_textImg_dicts(I, P))
MetadataCatalog.get("valSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("valSet")
print('textImg_metadata: ', textImg_metadata)

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainSet",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.MAX_ITER =180000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5

cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0199499.pth")
# cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TEST = ("valSet", )

predictor = DefaultPredictor(cfg)
print('predictor ready.')

'''
def timekeeping(predictor, test_num):
    time_used_total = 0
    for d in random.sample(utils.get_textImg_dicts(images, valPath), test_num):
        img = cv2.imread(d["file_name"])
        time_before_pred = time.time()
        output = predictor(img)
        time_used_total += time.time()-time_before_pred
    print('{0} s per sample when inferencing.'.format(time_used_total/test_num))
'''
def eval_visualization(predictor):
    utils.draw_predImg_dicts(utils.get_textImg_dicts(images, valPath), 10, textImg_metadata, predictor)

if __name__ == '__main__':
    # timekeeping(predictor, 100)
    eval_visualization(predictor)
