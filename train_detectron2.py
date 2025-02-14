#обучение
def train():
    import os.path
    import logging
    import torch
    from collections import OrderedDict
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common libraries
    import numpy as np
    import os, json, cv2, random

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.utils.visualizer import ColorMode
    from detectron2.solver import build_lr_scheduler, build_optimizer
    from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
    from detectron2.utils.events import EventStorage
    from detectron2.modeling import build_model
    import detectron2.utils.comm as comm
    from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
    from detectron2.data import (
        MetadataCatalog,
        build_detection_test_loader,
        build_detection_train_loader,
    )
    from detectron2.evaluation import (
        CityscapesInstanceEvaluator,
        CityscapesSemSegEvaluator,
        COCOEvaluator,
        COCOPanopticEvaluator,
        DatasetEvaluators,
        LVISEvaluator,
        PascalVOCDetectionEvaluator,
        SemSegEvaluator,
        inference_on_dataset,
        print_csv_format,
    )

    from detectron2.data import detection_utils as utils
    from detectron2.data import transforms as T
    from detectron2.data import build_detection_train_loader


    from matplotlib import pyplot as plt
    from PIL import Image


    register_coco_instances("animal-2_train", {}, f"F:/dataset/animal-2-aug-1/train/_annotations.coco.json", f"F:/dataset/animal-2-aug-1/train")
    register_coco_instances("animal-2_valid", {}, f"F:/dataset/animal-2-aug-1/valid/_annotations.coco.json", f"F:/dataset/animal-2-aug-1/valid")
    register_coco_instances("animal-2_test", {}, f"F:/dataset/animal-2-aug-1/test/_annotations.coco.json", f"F:/dataset/animal-2-aug-1/test")

    dataset_train = DatasetCatalog.get("animal-2_train")

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.OUTPUT_DIR='F:\project\output2'

    # Dataset
    cfg.DATASETS.TRAIN = ("animal-2_train",)
    cfg.DATASETS.TEST = ("animal-2_valid",)
    cfg.DATALOADER.NUM_WORKERS = 8

    # Model Weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml") 

    # Solver parameters
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.OPTIMIZER = "AdamW"
    cfg.SOLVER.MAX_ITER = 30000

    # ROI Head parameters
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 740
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.EVAL_PERIOD = 30000

    # Input augmentations
    cfg.INPUT.MASK_FORMAT = "bitmask"


    PATIENCE = 30000 #Early stopping will occur after N iterations of no imporovement in total_loss

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    def get_evaluator(cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    def do_test(cfg, model):
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = get_evaluator(
                cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            )
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)
        if len(results) == 1:
            results = list(results.values())[0]
        return results


    logger = logging.getLogger("detectron2")
    resume=False
    model = build_model(cfg)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    BEST_LOSS = np.inf

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    prev_iter = start_iter
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    patience_counter = 0
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            if iteration > prev_iter:
                prev_iter = iteration
                if losses_reduced < BEST_LOSS:
                    BEST_LOSS = losses_reduced
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter % 100 == 0:
                        print(f"Loss has not improved for {patience_counter} iterations")
                    if patience_counter >= PATIENCE:
                        print(f"EARLY STOPPING")
                        break

    do_test(cfg, model)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    dataset_dicts = DatasetCatalog.get("animal-2_valid")
    for d in random.sample(dataset_dicts, 3):
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                    scale=0.5,
                    instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        im = Image.fromarray(out.get_image()[:, :, ::-1])

if __name__=='__main__':

    train()