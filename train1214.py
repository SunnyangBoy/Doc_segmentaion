import utils1214
import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
setup_logger()

trainPath1 = "/home/ubuntu/cs/publaynet/publaynet_data/train_1214"
trainPath2 = "/home/ubuntu/cs/publaynet/publaynet_data/scanning_pdf_1214"

images = {}
img_id = 0

for trainPath in [trainPath1, trainPath2]:
    json_files = os.listdir(trainPath)

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

DatasetCatalog.register("trainSet", lambda I = images : utils1214.get_textImg_dicts(I))
MetadataCatalog.get("trainSet").set(thing_classes=categories)
textImg_metadata = MetadataCatalog.get("trainSet")
print('textImg_metadata: ', textImg_metadata)

# utils1214.draw_textImg_dicts(utils1214.get_textImg_dicts(images), 3, textImg_metadata)

cfg = get_cfg()
cfg.merge_from_file("./detectron2_repo/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("trainSet",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 6
# cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"
cfg.MODEL.WEIGHTS = os.path.join('/home/ubuntu/cs/publaynet/output_one', "model_final_paras.pth")
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.CHECKPOINT_PERIOD = 500
cfg.SOLVER.MAX_ITER = 300000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.SOLVER.LR_SCHEDULER_NAME = "PublaynetLR"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()
print('train finished')

