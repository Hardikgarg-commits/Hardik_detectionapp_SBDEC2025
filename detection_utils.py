import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from typing import List, Dict, Tuple
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@st.cache_resource
def load_yolo_model(weights_path: str):
    """
    Cache the YOLO model so it loads only once.
    """
    model = YOLO(weights_path)
    return model

def batch_infer(model, image_paths, imgsz=320, batch_size=8, conf=0.25):
    """
    Run fast batched inference on CPU.
    """
    results = []

    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]

        res = model.predict(
            source=chunk,
            imgsz=imgsz,
            device="cpu",
            batch=len(chunk),
            conf=conf,
            verbose=False,
            save=False,
            save_txt=False
        )

        results.extend(res)

    return results


class DetectionManager:
    def __init__(self):
        self.model_path = Path(__file__).parent.parent / "best.pt"
        self.model = None

        self.class_names = {
            0: "Stones / Stone Pillars / Stone Structures",
            1: "Crops / Farmland",
            2: "Non-archaeological (deserts, water, mountains, etc.)",
            3: "Heritage Sites (temples, palaces, forts, museums)"
        }

        self.class_colors = {
            0: (139, 69, 19),
            1: (34, 139, 34),
            2: (105, 105, 105),
            3: (184, 134, 11)
        }

        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(self.model_path):
                self.model = load_yolo_model(str(self.model_path))
                return True
            else:
                st.error(f"Model file not found at {self.model_path}")
                return False
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return False

   
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        if self.model is None:
            return []

        detections = []

        try:
            results = self.model(image)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls = int(boxes.cls[i].cpu().numpy())

                        detections.append({
                            "bbox": box.tolist(),
                            "confidence": conf,
                            "class_id": cls,
                            "class_name": self.class_names.get(cls, "Unknown"),
                            "color": self.class_colors.get(cls, (255, 255, 255))
                        })

            return detections
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            return []

    def detect_batch(self, image_paths, imgsz=320, batch_size=8, conf=0.25):
        if self.model is None:
            st.error("Model not loaded")
            return []
        return batch_infer(self.model, image_paths, imgsz, batch_size, conf)

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            color = det["color"]
            label = f"{det['class_name']} {det['confidence']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text_w, text_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - text_h - 6), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated

    def crop_detections(self, image: np.ndarray, detections: List[Dict]):
        crops = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
        return crops

    def process_video_frame(self, frame: np.ndarray):
        detections = self.detect_objects(frame)
        annotated = self.draw_detections(frame, detections)
        return annotated, detections


    def get_class_statistics(self, detections_list):
        all_dets = [d for group in detections_list for d in group]

        if not all_dets:
            return {
                "total_detections": 0,
                "class_counts": {},
                "confidence_avg": 0,
                "class_confidence_avg": {}
            }

        class_counts = {}
        class_conf = {}

        for d in all_dets:
            name = d["class_name"]
            conf = d["confidence"]

            class_counts[name] = class_counts.get(name, 0) + 1
            class_conf.setdefault(name, []).append(conf)

        conf_avg = sum(d["confidence"] for d in all_dets) / len(all_dets)

        class_conf_avg = {k: sum(v) / len(v) for k, v in class_conf.items()}

        return {
            "total_detections": len(all_dets),
            "class_counts": class_counts,
            "confidence_avg": conf_avg,
            "class_confidence_avg": class_conf_avg,
            "all_detections": all_dets
        }
