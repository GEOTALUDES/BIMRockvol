# IMPORT REQUIRED LIBRARIES 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

def load_model(config_path, weights_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_path))
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    return DefaultPredictor(cfg)

def detect_rocks(predictor, image):
    outputs = predictor(image)
    return outputs

def visualize_detection(image, outputs, metadata):
    v = Visualizer(image[:, :, ::-1],
                   metadata=metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE_BW
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]

def calculate_rock_volume(outputs, metadata, image_shape):
    total_area = image_shape[0] * image_shape[1]
    instances = outputs["instances"]
    areas_roca = []

    if hasattr(instances, 'pred_classes'):
        for i in range(len(instances)):
            if metadata.thing_classes[instances.pred_classes[i]] == "roca":
                mask = instances.pred_masks[i].cpu().numpy()
                area = mask.sum()
                areas_roca.append(area)
    else:
        print("Warning: 'pred_classes' not found in Instances object. Check predictor output.")

    area_ocupada_roca = sum(areas_roca)
    porcentaje_roca = (area_ocupada_roca / total_area) * 100
    
    if porcentaje_roca < 25:
        categoria_roca = "matriz"
    elif porcentaje_roca <= 75:
        categoria_roca = "BIM rock"
    else:
        categoria_roca = "macizo rocoso"

    return porcentaje_roca, categoria_roca
