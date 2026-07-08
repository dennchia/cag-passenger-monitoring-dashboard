import os
os.environ["TRUST_REMOTE_CODE"] = "True"

# =================================================================
# THE BULLETPROOF WINDOWS BYPASS (Monkey Patch)
# =================================================================
import transformers.dynamic_module_utils
transformers.dynamic_module_utils.resolve_trust_remote_code = lambda *args, **kwargs: True

import transformers.models.auto.configuration_auto
if hasattr(transformers.models.auto.configuration_auto, 'resolve_trust_remote_code'):
    transformers.models.auto.configuration_auto.resolve_trust_remote_code = lambda *args, **kwargs: True

import transformers.models.auto.image_processing_auto
if hasattr(transformers.models.auto.image_processing_auto, 'resolve_trust_remote_code'):
    transformers.models.auto.image_processing_auto.resolve_trust_remote_code = lambda *args, **kwargs: True
# =================================================================

import cv2
import torch
import numpy as np
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor
import time 

class DemographicsEngine:
    def __init__(self):
        print("Initializing Official MiVOLO V2 (Transformers Engine)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.config = AutoConfig.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
        
        # --- UPGRADE: Added torch.float16 to cut VRAM usage in half! ---
        self.model = AutoModelForImageClassification.from_pretrained(
            "iitolstykh/mivolo_v2", 
            config=self.config, 
            trust_remote_code=True,
            torch_dtype=torch.float16 
        ).to(self.device)
        self.model.eval()
        
        self.processor = AutoImageProcessor.from_pretrained("iitolstykh/mivolo_v2", trust_remote_code=True)
        
        # Pull the official gender text dictionary from the model
        self.id2label = self.config.gender_id2label 
        print("MiVOLO V2 Online & Ready!")

    def analyze_batch(self, crop_list):
        import cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        frame_ages = []
        frame_genders = []

        for body_crop in crop_list:
            if body_crop.size == 0 or body_crop.shape[0] < 10 or body_crop.shape[1] < 10:
                continue

           # --- STEP 1: LIGHTWEIGHT FACE DETECTION ---
            gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))

            # THE FIX: Default the crop to None, not the tensor!
            face_crop = None

            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
                fx, fy, fw, fh = faces[0]

                h_pad = int(fh * 0.25)
                w_pad = int(fw * 0.20)
                y1 = max(0, fy - h_pad)
                y2 = min(body_crop.shape[0], fy + fh + h_pad)
                x1 = max(0, fx - w_pad)
                x2 = min(body_crop.shape[1], fx + fw + w_pad)

                face_crop = body_crop[y1:y2, x1:x2]

                os.makedirs("face_debug", exist_ok=True)
                cv2.imwrite(os.path.join("face_debug", f"face_crop_{time.time_ns()}.jpg"), face_crop)

            # --- STEP 2: NATIVE PROCESSOR EXTRACTION ---
            # If no face was found, face_crop is None. The processor translates it into a safe dummy tensor!
            faces_input = self.processor(images=[face_crop])["pixel_values"]
            body_input = self.processor(images=[body_crop])["pixel_values"]

            faces_tensor = faces_input.to(dtype=self.model.dtype, device=self.device)
            body_tensor = body_input.to(dtype=self.model.dtype, device=self.device)

            # --- STEP 3: NATIVE INFERENCE ---
            with torch.no_grad():
                outputs = self.model(faces_input=faces_tensor, body_input=body_tensor)

            # --- STEP 4: DIRECT PARSING ---
            # We deleted the massive 40-line guessing loop and just asked for the exact variables
            try:
                raw_age = outputs.age_output[0].item()
                gender_idx = outputs.gender_class_idx[0].item()
                
                # Let the official dictionary handle the "Male/Female" translation
                gender = self.id2label[gender_idx]

                # We keep our Zero-Filter safety net just in case it fails on a bad image
                if round(raw_age) > 1:
                    frame_ages.append(round(raw_age))
                    frame_genders.append(gender)
                    
            except Exception as e:
                print(f"[Demographics] Parse error: {e}")
                continue

        # Final statistical consensus across all 5 harvested snapshot frames
        final_age = int(np.mean(frame_ages)) if frame_ages else 0
        final_gender = max(set(frame_genders), key=frame_genders.count) if frame_genders else "Unknown"

        print(frame_ages)
        print(frame_genders)

        return final_age, final_gender