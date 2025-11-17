# ill just map categories
# e.x. bottle -> plastic waste

from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

class TrashType(str, Enum):
    ORGANIC = "Organic Waste"
    PAPER = "Paper/Cardboard"
    PLASTICS = "Plastics"
    OTHER = "Landfill/Other"

# Mapping YOLO COCO class names -> trash type
# NEED TO ADD OBJECTS WE CARE ABOUT
YOLO_TO_CATEGORY: Dict[str, TrashType] = {
    # Organic Waste
    "banana": TrashType.ORGANIC,
    "apple": TrashType.ORGANIC,
    "orange": TrashType.ORGANIC,
    "broccoli": TrashType.ORGANIC,
    "carrot": TrashType.ORGANIC,
    "sandwich": TrashType.ORGANIC,
    "hot dog": TrashType.ORGANIC,
    "pizza": TrashType.ORGANIC,
    "donut": TrashType.ORGANIC,
    "cake": TrashType.ORGANIC,

    # Paper/Cardboard
    "book": TrashType.PAPER,
    "bench": TrashType.PAPER,
    # !! might need to change/remove category, 
    # or train with more objects,
    # or check class list
    
    # Plastics
    "bottle": TrashType.PLASTICS,
    "cup": TrashType.PLASTICS,
    "spoon": TrashType.PLASTICS,
    "frisbee": TrashType.PLASTICS,

    # landfill/other is default
}

@dataclass
class DetectionResult:
    category: TrashType
    category_confidence: float
    yolo_class: str
    yolo_confidence: float
    bbox_xyxy: Tuple[float, float, float, float]
    image_center: Tuple[int, int]
    # if we want to use ZED2 (3D) in future
    position_3d: Optional[Tuple[float, float, float]] = None

def categorize_detection(yolo_class: str, yolo_conf: float) -> DetectionResult:
    # map YOLOv8 class name + conf -> 1 of 4 trash types
    # category_confidence = yolo_conf for now

    # default = landfill/other
    base_cat = YOLO_TO_CATEGORY.get(yolo_class, TrashType.OTHER)

    return DetectionResult(
        category=base_cat,
        category_confidence=float(yolo_conf),
        yolo_class=yolo_class,
        yolo_confidence=float(yolo_conf),
        bbox_xyxy=(0.0, 0.0, 0.0, 0.0),
        image_center=(0,0),
        position_3d=None,
    )