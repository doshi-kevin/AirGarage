#!/usr/bin/env python3
"""
Airgarage Recovery Pipeline (GPU FORCED)
Fixes "libcudnn.so.9 not found" by pre-loading NVIDIA libraries via ctypes.
"""
import os
import sys
import glob
import re
import cv2
import json
import queue
import threading
import requests
import numpy as np
import ctypes
from tqdm import tqdm

# ============= CRITICAL GPU SETUP =============
def force_load_nvidia_libs():
    """
    Aggressively finds and loads cuDNN/cuBLAS libraries to prevent
    ONNX Runtime from failing to find them.
    """
    print("[SETUP] Searching for NVIDIA libraries...")
    base_prefix = sys.prefix
    site_packages = glob.glob(f"{base_prefix}/lib/python*/site-packages") + \
                    glob.glob(f"{base_prefix}/Lib/site-packages")

    libs_found = 0
    for sp in site_packages:
        # Look for specific libraries required by ONNX Runtime 1.16+
        patterns = [
            f"{sp}/nvidia/cudnn/lib/libcudnn.so*",
            f"{sp}/nvidia/cublas/lib/libcublas.so*",
            f"{sp}/nvidia/cublas/lib/libcublasLt.so*",
            f"{sp}/nvidia/cuda_runtime/lib/libcudart.so*"
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            for lib_path in matches:
                try:
                    ctypes.CDLL(lib_path)
                    libs_found += 1
                except Exception as e:
                    pass
    
    if libs_found > 0:
        print(f"[SETUP] ✅ Force-loaded {libs_found} NVIDIA libraries into memory.")
    else:
        print("[SETUP] ❌ Could not find NVIDIA libraries. GPU will likely fail.")

# LOAD LIBS BEFORE IMPORTING AI
force_load_nvidia_libs()
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
from fast_alpr import ALPR
# ==============================================

# ============= CONFIG =============
INPUT_FAILED = "failed_extraction.txt"
MAIN_JSON_DB = "vehicle_metadata.json"

# Speed
DOWNLOAD_WORKERS = 32
QUEUE_SIZE = 100
RECOVERY_CONF_THRESHOLD = 0.40  
# ==================================

class GridRecoveryPipeline:
    def __init__(self):
        print("[INIT] Loading AI Models for Recovery...")
        self.using_gpu = False
        try:
            self.alpr = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model",
                detector_providers=['CUDAExecutionProvider'], # Strict GPU
                ocr_providers=['CUDAExecutionProvider']
            )
            # Test run to verify GPU
            try:
                dummy = np.zeros((100,100,3), dtype=np.uint8)
                self.alpr.predict(dummy)
                print("[INIT] ✅ GPU Mode CONFIRMED (Inference working)")
                self.using_gpu = True
            except Exception as e:
                print(f"[INIT] ⚠️ GPU loaded but failed inference: {e}")
                raise e
                
        except Exception as e:
            print(f"\n[CRITICAL] GPU Init Failed: {e}")
            print("Falling back to CPU (This will be SLOW 4s/img)...\n")
            self.alpr = ALPR(
                detector_model="yolo-v9-t-256-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model", 
                detector_providers=['CPUExecutionProvider'],
                ocr_providers=['CPUExecutionProvider']
            )

        self.gpu_queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.recovered_data = []
        self.still_failed = []
        self.download_done = threading.Event()

    def clean_plate(self, text):
        if not text: return ""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    # --- GRID SLICING ---
    def generate_crops(self, img):
        h, w = img.shape[:2]
        crops = []
        h_mid, w_mid = h // 2, w // 2
        h_over, w_over = int(h * 0.15), int(w * 0.15) # 15% Overlap

        # 4 Quadrants
        crops.append(img[0 : h_mid + h_over, 0 : w_mid + w_over]) # TL
        crops.append(img[0 : h_mid + h_over, w_mid - w_over : w]) # TR
        crops.append(img[h_mid - h_over : h, 0 : w_mid + w_over]) # BL
        crops.append(img[h_mid - h_over : h, w_mid - w_over : w]) # BR
        
        # Center Zoom (Focused)
        y1, y2 = int(h * 0.20), int(h * 0.80)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        center = img[y1:y2, x1:x2]
        # Upscale center slightly
        center = cv2.resize(center, None, fx=1.3, fy=1.3)
        crops.append(center)

        return crops

    def get_image_metrics(self, img):
        try:
            h, w, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            avg_color = np.average(np.average(img, axis=0), axis=0)
            rgb = [int(avg_color[2]), int(avg_color[1]), int(avg_color[0])]
            return {"width": w, "height": h, "blur_score": round(blur_score, 2), "brightness": round(brightness, 2), "avg_color_rgb": rgb}
        except:
            return None

    def predict_on_crops(self, img):
        """Run AI on 5 crops"""
        crops = self.generate_crops(img)
        best_plate = ""
        best_conf = 0.0

        for crop in crops:
            if crop.size == 0: continue
            crop = np.ascontiguousarray(crop) # Critical for CUDA
            
            try:
                results = self.alpr.predict(crop)
                if results:
                    for p in results:
                        if hasattr(p, 'ocr') and p.ocr:
                            if p.ocr.confidence > best_conf:
                                best_conf = p.ocr.confidence
                                best_plate = p.ocr.text
            except:
                pass

        return self.clean_plate(best_plate), best_conf

    # --- WORKERS ---
    def downloader(self, urls):
        with requests.Session() as s:
            for url in urls:
                try:
                    r = s.get(url, timeout=4)
                    if r.status_code == 200:
                        arr = np.frombuffer(r.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        self.gpu_queue.put((url, img))
                    else:
                        self.gpu_queue.put((url, None))
                except:
                    self.gpu_queue.put((url, None))

    def processor(self, total):
        pbar = tqdm(total=total, desc="🔍 Grid Searching")
        processed = 0
        while processed < total:
            try:
                try:
                    url, img = self.gpu_queue.get(timeout=5)
                except queue.Empty:
                    if self.download_done.is_set(): break
                    continue

                if img is None:
                    self.still_failed.append(url)
                else:
                    try:
                        plate, conf = self.predict_on_crops(img)
                        if conf >= RECOVERY_CONF_THRESHOLD and len(plate) >= 3:
                            metrics = self.get_image_metrics(img)
                            data = {"url": url, "plate": plate, "conf": round(conf, 4), "meta": metrics}
                            self.recovered_data.append(data)
                        else:
                            self.still_failed.append(url)
                    except:
                        self.still_failed.append(url)

                processed += 1
                pbar.update(1)
                self.gpu_queue.task_done()
            except:
                processed += 1
                pbar.update(1)
        pbar.close()

    def run(self):
        if not os.path.exists(INPUT_FAILED):
            print(f"❌ No {INPUT_FAILED} found.")
            return
        with open(INPUT_FAILED) as f:
            urls = [x.strip() for x in f if x.strip()]
        if not urls:
            print("No URLs to process.")
            return

        print(f"Attempting recovery on {len(urls)} images...")
        chunks = [urls[i::DOWNLOAD_WORKERS] for i in range(DOWNLOAD_WORKERS)]
        threads = []
        for c in chunks:
            t = threading.Thread(target=self.downloader, args=(c,), daemon=True)
            t.start()
            threads.append(t)

        self.processor(len(urls))
        self.download_done.set()
        for t in threads: t.join()

        print(f"\n[MERGING] Adding {len(self.recovered_data)} recoveries to DB...")
        full_data = []
        if os.path.exists(MAIN_JSON_DB):
            with open(MAIN_JSON_DB, 'r') as f:
                try: full_data = json.load(f)
                except: pass
        full_data.extend(self.recovered_data)
        
        with open(MAIN_JSON_DB, 'w') as f:
            json.dump(full_data, f, indent=2)

        print(f"[UPDATING] Writing remaining failures...")
        with open(INPUT_FAILED, 'w') as f:
            for u in self.still_failed: f.write(f"{u}\n")

        print("="*40)
        print(f"✅ Recovered: {len(self.recovered_data)}")
        print(f"❌ Remaining: {len(self.still_failed)}")
        print("="*40)

if __name__ == "__main__":
    pipeline = GridRecoveryPipeline()
    pipeline.run()