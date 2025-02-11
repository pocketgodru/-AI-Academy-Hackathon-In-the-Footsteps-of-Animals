# Хакатон Академии ИИ «По следам животных»
## Решение команды *дрчки*

## Модели

В качестве моделей мы выбрали модели [sam2.1_hiera_base_plus](https://github.com/facebookresearch/sam2) и модель [detectron2](https://github.com/facebookresearch/detectron2/blob/main/configs/Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml) с основой cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv

Обучение sam2.1 длилось около 11-12 часов

Обучение detectron2 в сумме заняло около 25 часов основное обучение и 8 до обучения 

<br>

Модель sam2.1 обученная нами - https://www.kaggle.com/models/sassukeuchicha/test_sam/PyTorch/default/6

Параметры обучения для sam2 - https://www.kaggle.com/datasets/sergeneyzershan/yuukijke

	Необходимо поменять пути к датасету в `img_folder` и `gt_folder`

<br>

Модель detectron2 обученная нами(итоговая с до обучением) - https://www.kaggle.com/models/sassukeuchicha/10k/PyTorch/default/6

без до обучения - https://www.kaggle.com/models/sassukeuchicha/10k/PyTorch/default/5
<br>

Обучение detectron2 производилось локально на RTX3060 12G и 32G озу (до обучалась в Kaggle на P100 )

Обучение sam2.1 производилось в Kaggle на 2xT4 по 16G и 32G озу 

Обучение sam2.1 возможно только в Linux (для Windows можно попробовать использовать WSL, не проверялось)

## Описание файлов

#### Файлы

`aug_train.py` - код для аугментации датасета(поменять пути на ваши)

`train_detectron2.py` - код  для обучения detectron2(поменять пути на ваши)

`fine_tune_detectron2.ipynb` - код для до обучения detectron2(поменять пути на ваши)

`train_sam2.1.ipynb` - обучения sam2.1(поменять пути на ваши)

`train.yaml` - cfg для обучения sam2.1(поменять пути к датасету на ваши)

`sub.py`  - код для создания submission(поменять пути к моделям)

#### Папки 

`result_train_sam` - в папке находятся результаты обучения sam, открывать через tensorboard

`result_train_detectron` - результаты обучения detectron2, результаты представлены как в json так и в tensorboard

`result_fine_tune_detectron` - результаты до обучения detectron2, результаты представлены как в json так и в tensorboard

##  Установка зависимостей 

для detectron2 

```python
pip install roboflow ultralytics kagglehub
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

для sam2.1 
```python
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .[dev] -q
cd ./checkpoints && ./download_ckpts.sh
```

## Датасет 

За основу был взят предложенный датасет.

Для обучения sam2.1 был использован без аугументация для удобства использования он был загжен на [Roboflow](https://app.roboflow.com/sassuke-uchicha-6mbbd/animal-99law/2), так как там удобный конвертер в разные форматы(для sam требуется отдельный)
Код для скачивания датасета:
<br></br>
```python
from roboflow import Roboflow
rf = Roboflow(api_key="Qxn6CLTseFDo400st1GD")
project = rf.workspace("sassuke-uchicha-6mbbd").project("animal-99law")
version = project.version(3)
dataset = version.download("sam2")
```
<br></br>
Для обучения detectron2  мы аугументировали датасет до 120927 изображений.
Его вы можете скачать с [Kaggle](https://www.kaggle.com/datasets/sassukeuchicha/animal-seg-1/data)

или
<br></br>
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sassukeuchicha/animal-seg-1")

print("Path to dataset files:", path)
```
  Датасет скачается на системный диск и сам разархивируется, 
  минимально рекомендуем 60G свободного места.
  Необходимо позже заменить пути в коде на ваши, то есть куда скачались датасеты 

## Обучение detectron2  

Для запуска обучения detectron2  запустите `train_detectron2.py`

Для до обучения detectron2 откройте  `fine_tune_detectron2.ipynb`

## Обучение sam2.1

В качестве примера и основы был взят код из гайда от [Roboflow](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/add-fine-tune-sam-2.1/notebooks/fine-tune-sam-2.1.ipynb?ref=blog.roboflow.com#scrollTo=LJZGcpRpgevM)

Для обучения sam2.1 откройте `train_sam2.1.ipynb`

Для обучения необходим `train.yaml` в нём так же надо поменять пути к датасету

##  Создание submission.json

Надо будет поменять путь к моделям и изображениям 

Запустить `sub.py`
