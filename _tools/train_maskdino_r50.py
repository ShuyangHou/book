# -*- coding: utf-8 -*-
"""Train MaskDINO-R50 for book spine instance segmentation (first version).

Prerequisites:
    pip install torch torchvision
    pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
    git clone https://github.com/IDEA-Research/MaskDINO.git
    cd MaskDINO && pip install -e .

Usage:
    python _tools/train_maskdino_r50.py \
        --data-root book_spine_dataset/coco \
        --output-dir output/maskdino_r50_v1 \
        --num-gpus 1 \
        --max-iter 3000

The script registers the book_spine dataset, builds a MaskDINO-R50 config,
and launches training via Detectron2's DefaultTrainer.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def register_book_spine(data_root: str):
    """Register train/val splits as COCO instance datasets in Detectron2."""
    from detectron2.data.datasets import register_coco_instances

    for split in ("train", "val"):
        name = f"book_spine_{split}"
        json_file = os.path.join(data_root, split, f"instances_{split}.json")
        image_root = os.path.join(data_root, split, "images")
        register_coco_instances(name, {}, json_file, image_root)
        print(f"Registered: {name} ({json_file})")


def build_config(data_root: str, output_dir: str, max_iter: int, batch_size: int,
                 lr: float, num_gpus: int):
    """Build a Detectron2 CfgNode for MaskDINO-R50 instance segmentation."""
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    # Try to import MaskDINO config
    try:
        from maskdino import add_maskdino_config
        has_maskdino = True
    except ImportError:
        has_maskdino = False
        print("WARNING: MaskDINO not installed. Falling back to Mask R-CNN R50-FPN.")

    cfg = get_cfg()

    if has_maskdino:
        add_maskdino_config(cfg)
        # Use MaskDINO R50 config
        maskdino_config = Path(__file__).parent.parent / "MaskDINO" / "configs" / "coco" / "instance-segmentation" / "maskdino_R50_bs16_50ep_3s.yaml"
        if maskdino_config.exists():
            cfg.merge_from_file(str(maskdino_config))
        else:
            # Fallback: find any MaskDINO R50 config
            print(f"MaskDINO config not found at {maskdino_config}")
            print("Please clone MaskDINO repo alongside this project.")
            print("Falling back to Mask R-CNN.")
            has_maskdino = False

    if not has_maskdino:
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

    # Dataset
    cfg.DATASETS.TRAIN = ("book_spine_train",)
    cfg.DATASETS.TEST = ("book_spine_val",)

    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = 2

    # Solver - tuned for small dataset
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(max_iter * 0.6), int(max_iter * 0.85))
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_ITERS = min(200, max_iter // 10)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = max(500, max_iter // 5)

    # Model head - 1 class (book_spine)
    if has_maskdino:
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Evaluation
    cfg.TEST.EVAL_PERIOD = max(200, max_iter // 10)

    # Input - moderate augmentation for small dataset
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    # Output
    cfg.OUTPUT_DIR = output_dir

    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="book_spine_dataset/coco")
    parser.add_argument("--output-dir", default="output/maskdino_r50_v1")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # Register datasets
    register_book_spine(args.data_root)

    # Build config
    cfg = build_config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        num_gpus=args.num_gpus,
    )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Train
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator

    class BookSpineTrainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            if output_folder is None:
                output_folder = os.path.join(cfg.OUTPUT_DIR, "eval")
            return COCOEvaluator(dataset_name, output_dir=output_folder)

    trainer = BookSpineTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

    # Final evaluation
    print("\n=== Final evaluation on val ===")
    from detectron2.evaluation import inference_on_dataset
    from detectron2.data import build_detection_test_loader
    evaluator = COCOEvaluator("book_spine_val", output_dir=os.path.join(cfg.OUTPUT_DIR, "eval_final"))
    val_loader = build_detection_test_loader(cfg, "book_spine_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    print(results)

    print(f"\nModel saved to: {cfg.OUTPUT_DIR}")
    print(f"Best checkpoint: {cfg.OUTPUT_DIR}/model_final.pth")


if __name__ == "__main__":
    main()
