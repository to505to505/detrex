from typing import Tuple
from detectron2.config import CfgNode
from detectron2.modeling.backbone import D2SwinTransformer

class MaskDINOCustom(MaskDINO):
    def init(
        self,
        *,
        sem_seg_head,
        criterion,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
    ):
        # Create Swin config
        cfg = CfgNode()
        cfg.MODEL = CfgNode()
        cfg.MODEL.SWIN = CfgNode()

        # Set Swin parameters
        cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 384
        cfg.MODEL.SWIN.PATCH_SIZE = 4
        cfg.MODEL.SWIN.EMBED_DIM = 192
        cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        cfg.MODEL.SWIN.NUM_HEADS = [6, 12, 24, 48]
        cfg.MODEL.SWIN.WINDOW_SIZE = 12
        cfg.MODEL.SWIN.MLP_RATIO = 4.0
        cfg.MODEL.SWIN.QKV_BIAS = True
        cfg.MODEL.SWIN.QK_SCALE = None
        cfg.MODEL.SWIN.DROP_RATE = 0.0
        cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
        cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
        cfg.MODEL.SWIN.APE = False
        cfg.MODEL.SWIN.PATCH_NORM = True
        cfg.MODEL.SWIN.USE_CHECKPOINT = False
        cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

        # Create Swin backbone
        swin_backbone = D2SwinTransformer(cfg, input_shape=None)

        # Initialize parent class with our custom backbone
        super().init(
            backbone=swin_backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            semantic_on=semantic_on,
            panoptic_on=panoptic_on,
            instance_on=instance_on,
            test_topk_per_image=test_topk_per_image,
            pano_temp=pano_temp,
            focus_on_box=focus_on_box,
            transform_eval=transform_eval,
            semantic_ce_loss=semantic_ce_loss,
        )