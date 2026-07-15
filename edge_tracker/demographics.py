import cv2
import torch
import numpy as np
from transformers import AutoModelForImageClassification, AutoConfig, AutoImageProcessor

MIVOLO_MODEL_ID = "iitolstykh/mivolo_v2"
MIVOLO_REVISION = "53393526c220e34cdd7b722b36d22b6f9e5f4241"


class DemographicsEngine:
    def __init__(self, device=None):
        print("Initializing Official MiVOLO V2 (Transformers Engine)...")
        requested_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = str(requested_device)
        model_dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        
        self.config = AutoConfig.from_pretrained(
            MIVOLO_MODEL_ID,
            revision=MIVOLO_REVISION,
            trust_remote_code=True,
        )
        
        # --- UPGRADE: Added torch.float16 to cut VRAM usage in half! ---
        self.model = AutoModelForImageClassification.from_pretrained(
            MIVOLO_MODEL_ID,
            revision=MIVOLO_REVISION,
            config=self.config, 
            trust_remote_code=True,
            torch_dtype=model_dtype
        ).to(self.device)
        self.model.eval()
        
        self.processor = AutoImageProcessor.from_pretrained(
            MIVOLO_MODEL_ID,
            revision=MIVOLO_REVISION,
            trust_remote_code=True,
        )
        
        # Pull the official gender text dictionary from the model
        self.id2label = self.config.gender_id2label 
        print("MiVOLO V2 Online & Ready!")

    def analyze_batch(self, crop_list):
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

            # --- STEP 2: NATIVE PROCESSOR EXTRACTION ---
            # If no face was found, face_crop is None. The processor translates it into a safe dummy tensor!
            # MiVOLO's official processor expects OpenCV BGR crops and performs
            # the BGR-to-RGB conversion itself.
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
