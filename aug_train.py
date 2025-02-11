import os
import json
import cv2
import numpy as np
import albumentations as A
from pycocotools import mask as coco_mask
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def augment_data(annotations_file, images_dir, output_dir):
    # Убедитесь, что папки для сохранения существуют
    os.makedirs(output_dir, exist_ok=True)

    # Загрузите аннотации COCO
    with open(annotations_file, "r") as f:
        coco_data = json.load(f)

    bbox_params = A.BboxParams(format="coco", label_fields=["category_ids"])

    # Настройка аугментаций
    augmentations_list = [
        ("horizontal_flip", A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=bbox_params)),
        ("shift_scale_rotate", A.Compose([A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=30, p=1.0)], bbox_params=bbox_params)),
        ("optical_distortion", A.Compose([A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0)], bbox_params=bbox_params)),
        ("gaussian_noise", A.Compose([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)], bbox_params=bbox_params)),
        ("brightness_contrast", A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0)], bbox_params=bbox_params)),
        ("iso_noise", A.Compose([A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0)], bbox_params=bbox_params)),      
        ("random_rain", A.Compose([A.RandomRain(blur_value=3, rain_type='heavy', p=1.0)], bbox_params=bbox_params)),
        ("random_snow", A.Compose([A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=1.0)], bbox_params=bbox_params)),
        ("random_fog", A.Compose([A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0)], bbox_params=bbox_params)),
    ]


    # Создайте новый список аннотаций для аугментированных данных
    augmented_annotations = {"images": [], "annotations": [], "categories": coco_data["categories"]}
    annotation_id = 1

    def process_image(img_info):
        nonlocal annotation_id

        img_path = os.path.join(images_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            print(f"Ошибка загрузки изображения: {img_path}")
            return []

        height, width, _ = image.shape

        # Получите аннотации для текущего изображения
        img_id = img_info["id"]
        img_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == img_id]

        # Преобразование сегментаций в маски
        masks = []
        category_ids = []
        bboxes = []

        for ann in img_annotations:
            rle = coco_mask.frPyObjects(ann["segmentation"], height, width)
            mask = coco_mask.decode(rle)
            if len(mask.shape) == 3:  # Если несколько сегментов
                mask = np.any(mask, axis=2)
            masks.append(mask.astype(np.uint8))
            category_ids.append(ann["category_id"])
            bboxes.append(ann["bbox"])

        # Объединяем маски в единый массив
        masks_array = np.stack(masks, axis=-1) if masks else np.zeros((height, width, 0), dtype=np.uint8)

        results = []
        # Применяем каждую аугментацию из списка
        for aug_name, aug in augmentations_list:
            augmented = aug(
                **{
                    "image": image,
                    "masks": list(masks_array.transpose(2, 0, 1)),
                    "bboxes": bboxes,
                    "category_ids": category_ids,
                }
            )

            # Сохраняем аугментированное изображение
            aug_image = augmented["image"]
            aug_masks = augmented["masks"]
            aug_bboxes = augmented["bboxes"]
            aug_category_ids = augmented["category_ids"]

            aug_file_name = f"{img_info['file_name'].split('.')[0]}_{aug_name}.jpg"
            aug_img_path = os.path.join(output_dir, aug_file_name)
            cv2.imwrite(aug_img_path, aug_image)

            # Добавляем новую информацию об изображении
            results.append({
                "image": {
                    "id": annotation_id,
                    "file_name": aug_file_name,
                    "width": aug_image.shape[1],
                    "height": aug_image.shape[0],
                },
                "annotations": [
                    {
                        "id": annotation_id,
                        "image_id": annotation_id,
                        "category_id": category_id,
                        "segmentation": {
                            "counts": coco_mask.encode(np.asfortranarray(mask))["counts"].decode("utf-8"),
                            "size": coco_mask.encode(np.asfortranarray(mask))["size"]
                        },
                        "bbox": bbox,
                        "iscrowd": 0,
                        "area": float(np.sum(mask)),
                    }
                    for mask, bbox, category_id in zip(aug_masks, aug_bboxes, aug_category_ids)
                ],
            })
            annotation_id += 1
        return results

    # Обрабатываем изображения в потоках
    with ThreadPoolExecutor() as executor:
        futures = list(tqdm(executor.map(process_image, coco_data["images"]), total=len(coco_data["images"]), desc="Processing images"))
        for future in futures:
            for result in future:
                augmented_annotations["images"].append(result["image"])
                augmented_annotations["annotations"].extend(result["annotations"])

    # Сохраняем аугментированные аннотации
    aug_annotations_path = os.path.join(output_dir, "_annotations.coco.json")
    with open(aug_annotations_path, "w") as f:
        json.dump(augmented_annotations, f)

    print(f"Аугментация завершена для {images_dir}!")

# Укажите свои пути к обучающим фотографиям и папке для сохранения аугментированных аннотаций

list_dir_to_aug = ["animal-2/train/"]
list_dir_to_aug_annt = ["animal-2/train/_annotations.coco.json"]
list_dit_to_out = ["animal-2/aug_train"]

for annotations_file, images_dir, output_dir in zip(list_dir_to_aug_annt, list_dir_to_aug, list_dit_to_out):
    augment_data(annotations_file, images_dir, output_dir)


annotations_path = "animal-2/aug_train/_annotations.coco.json"

with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

annotations = coco_data['annotations']
for idx, annotation in enumerate(annotations):
    annotation['id'] = idx  #

with open(annotations_path, 'w') as f:
    json.dump(coco_data, f, indent=4)

print("Done")
