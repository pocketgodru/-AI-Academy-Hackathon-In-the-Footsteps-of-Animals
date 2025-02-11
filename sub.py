import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from pycocotools import mask as mask_util
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from ultralytics import SAM

# Настройка Detectron2
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
cfg.MODEL.WEIGHTS = "/kaggle/input/10k/pytorch/default/6/model_final (1).pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.000000000000000005
# 0.000000000005
predictor = DefaultPredictor(cfg)

# Настройка SAM
sam_model_path = "/kaggle/input/test_sam/pytorch/default/6/sam2.1_b.pt"
print(sam_model_path)
sam_model = SAM(sam_model_path)

# Папки
test_folder = "/kaggle/input/xakaton4ik2/test"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Функция обработки одного изображения
def process_image(file_name):
    image_path = os.path.join(test_folder, file_name)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Выполнение предсказания с помощью Detectron2
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    pred_classes = instances.pred_classes.tolist()
    pred_boxes = instances.pred_boxes.tensor.tolist()
    pred_scores = instances.scores.tolist()
    pred_masks_detectron2 = instances.pred_masks.numpy()
    
    # Обработка предсказаний SAM
    pred_masks_sam = []
    for bbox in pred_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        input_box = [x_min, y_min, x_max, y_max]
        sam_result = sam_model(source=image, bboxes=[input_box], verbose=False, stream_buffer=True, augment=True, retina_masks=True)
        
        # Check if SAM returns a valid result
        if sam_result is not None:
            sam_mask = sam_result[0].masks[0]
            pred_masks_sam.append(sam_mask.data.cpu().numpy().astype(np.uint8))
        else:
            # If no mask is returned, append a zero mask
            pred_masks_sam.append(pred_masks_detectron2[i])
    
    # Формирование предсказаний
    image_predictions = []
    for i in range(len(pred_classes)):
        # Преобразование маски в формат RLE
        rle_list = mask_util.encode(np.asfortranarray(pred_masks_sam[i].astype(np.uint8)))
        # Преобразование "counts" в строку, если это байты
        for rle in rle_list:
            if isinstance(rle["counts"], bytes):
                rle["counts"] = rle["counts"].decode("utf-8")

        
        # Вычисление ширины и высоты bbox
        x_min, y_min, x_max, y_max = pred_boxes[i]
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        image_predictions.append({
            "image_name": file_name,
            "category_id": 1,  # Категория "Animal" имеет ID 1
            "bbox": [x_min, y_min, bbox_width, bbox_height],
            "score": pred_scores[i],
            "segmentation": {
                "size": [height, width],
                "counts": rle["counts"]
            }
        })
    return image_predictions


# Список изображений
image_files = os.listdir(test_folder)

# Обработка изображений по одному
predictions = []
for file_name in tqdm(image_files, total=len(image_files)):
    result = process_image(file_name)
    predictions.extend(result)

# Сохранение предсказаний в JSON
predictions_file = os.path.join(output_folder, "submission.json")
with open(predictions_file, "w") as f:
    json.dump(predictions, f, indent=4)

print(f"Предсказания сохранены в {predictions_file}")
