import os
from .clip_encoder import CLIPVisionTower
from .siglip_encoder import SigLipVisionTower
from .siglip2_encoder import SigLip2VisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)

    if 'siglip2' in vision_tower.lower():
        return SigLip2VisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif 'siglip' in vision_tower.lower():
        return SigLipVisionTower(vision_tower, vision_tower_cfg=vision_tower_cfg, **kwargs)
    elif 'clip' in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')
