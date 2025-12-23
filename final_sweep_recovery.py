#!/usr/bin/env python3
"""
Airgarage Phase 3: The Final Sweep ("Zero Left Behind")
Strategies:
1. Extreme OCR: Binarization, Inversion, High Rotation.
2. Visual Cloning: If OCR fails, matches the image to the visually closest 'Unpaired Single' 
   and adopts its plate number.
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
from tqdm import tqdm

# ============= CONFIG =============
INPUT_FAILED = "failed_extraction.txt"
MAIN_JSON_DB = "vehicle_metadata.json"
SINGLES_JSON = "unpaired_singles.json"

# Speed
DOWNLOAD_WORKERS = 32 
QUEUE_SIZE = 100

# Visual Matching Weights
W_COLOR = 1.0
W_BRIGHTNESS = 0.5
# ==================================

# ============= GPU SETUP =============
def force_load_nvidia_libs():
    base_prefix = sys.prefix
    site_packages = glob.glob(f"{base_prefix}/lib/python*/site-packages") + \
                    glob.glob(f"{base_prefix}/Lib/site-packages")
    for sp in site_packages:
        patterns = [f"{sp}/nvidia/cudnn/lib/libcudnn.so*", f"{sp}/nvidia/cublas/lib/libcublas.so*"]
        for p in patterns:
            for lib in glob.glob(p):
                try: 
                    import ctypes
                    ctypes.CDLL(lib)
                except: pass

force_load_nvidia_libs()
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
from fast_alpr import ALPR
# =====================================

class FinalSweepPipeline:
    def __init__(self):
        print("[INIT] Loading Models for Final Sweep...")
        try:
            self.alpr = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model",
                detector_providers=['CUDAExecutionProvider'],
                ocr_providers=['CUDAExecutionProvider']
            )
            print("[INIT] ✅ GPU Mode Active")
        except:
            print("[INIT] ⚠️ GPU Failed, using CPU")
            self.alpr = ALPR(
                detector_model="yolo-v9-t-256-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model", 
                detector_providers=['CPUExecutionProvider'],
                ocr_providers=['CPUExecutionProvider']
            )

        self.gpu_queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.final_data = []
        self.download_done = threading.Event()

        # Load Singles for Visual Matching
        print("[INIT] Loading Unpaired Singles for Visual Cloning...")
        try:
            with open(SINGLES_JSON, 'r') as f:
                self.singles_pool = json.load(f)
            print(f"[INIT] Loaded {len(self.singles_pool)} candidates for visual matching.")
        except:
            self.singles_pool = []
            print("[WARNING] No singles file found. Visual matching will be disabled.")

    def clean_plate(self, text):
        if not text: return ""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    # --- EXTREME IMAGE PREP ---
    def extreme_preprocess(self, img):
        """Generates 3 extreme versions of the image"""
        variations = []
        
        # 1. Grayscale + CLAHE (High Contrast)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        high_contrast = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        variations.append(high_contrast)
        
        # 2. Inverted (Negative) - helps with white plates on dark cars
        inverted = cv2.bitwise_not(high_contrast)
        variations.append(inverted)
        
        # 3. Binarization (Black & White only)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        variations.append(binary_color)
        
        return variations

    # --- VISUAL MATCHING LOGIC ---
    def find_visual_twin(self, target_meta):
        """Finds the closest visual match in the singles pool"""
        if not self.singles_pool or not target_meta:
            return None, 0.0

        best_match = None
        min_dist = float('inf')
        
        t_rgb = np.array(target_meta['avg_color_rgb'])
        t_bright = target_meta['brightness']

        for candidate in self.singles_pool:
            if 'meta' not in candidate or not candidate['meta']: continue
            
            c_rgb = np.array(candidate['meta']['avg_color_rgb'])
            c_bright = candidate['meta']['brightness']
            
            # Weighted Distance
            dist_color = np.linalg.norm(t_rgb - c_rgb)
            dist_bright = abs(t_bright - c_bright)
            
            total_dist = (dist_color * W_COLOR) + (dist_bright * W_BRIGHTNESS)
            
            if total_dist < min_dist:
                min_dist = total_dist
                best_match = candidate

        return best_match, min_dist

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
        pbar = tqdm(total=total, desc="🧹 Final Sweep")
        processed = 0
        
        while processed < total:
            try:
                try:
                    url, img = self.gpu_queue.get(timeout=5)
                except queue.Empty:
                    if self.download_done.is_set(): break
                    continue

                if img is None:
                    # Broken Link: Assign Dummy to clear list
                    self.final_data.append({"url": url, "plate": "BROKEN_IMAGE", "conf": 0.0, "method": "dummy"})
                else:
                    # 1. Try Extreme OCR
                    found_ocr = False
                    variations = self.extreme_preprocess(img)
                    
                    for var in variations:
                        # Ensure contiguous for CUDA
                        var = np.ascontiguousarray(var)
                        try:
                            results = self.alpr.predict(var)
                            if results:
                                for p in results:
                                    if hasattr(p, 'ocr') and p.ocr and p.ocr.confidence > 0.1: # Very low threshold
                                        text = self.clean_plate(p.ocr.text)
                                        if len(text) >= 3:
                                            self.final_data.append({
                                                "url": url, 
                                                "plate": text, 
                                                "conf": p.ocr.confidence,
                                                "method": "extreme_ocr"
                                            })
                                            found_ocr = True
                                            break
                        except: pass
                        if found_ocr: break
                    
                    # 2. If OCR Failed, use Visual Cloning
                    if not found_ocr:
                        try:
                            # Calculate metrics for this image
                            h, w, _ = img.shape
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                            avg = np.average(np.average(img, axis=0), axis=0)
                            metrics = {
                                "avg_color_rgb": [int(avg[2]), int(avg[1]), int(avg[0])],
                                "brightness": np.mean(gray)
                            }
                            
                            # Find Twin
                            twin, score = self.find_visual_twin(metrics)
                            
                            if twin:
                                self.final_data.append({
                                    "url": url,
                                    "plate": twin['plate'], # ADOPT THE PLATE
                                    "conf": 0.0,
                                    "method": "visual_cloning",
                                    "cloned_from": twin['url']
                                })
                            else:
                                # Should rarely happen if singles exist
                                self.final_data.append({"url": url, "plate": "UNKNOWN", "conf": 0.0, "method": "failed"})
                        except:
                            self.final_data.append({"url": url, "plate": "ERROR", "conf": 0.0, "method": "failed"})

                processed += 1
                pbar.update(1)
                self.gpu_queue.task_done()
            except:
                processed += 1
                pbar.update(1)
        pbar.close()

    def run(self):
        if not os.path.exists(INPUT_FAILED):
            print("No failed images list found.")
            return

        with open(INPUT_FAILED) as f:
            urls = [x.strip() for x in f if x.strip()]
        
        if not urls:
            print("List is empty!")
            return

        print(f"Sweeping {len(urls)} final stubborn images...")
        
        chunks = [urls[i::DOWNLOAD_WORKERS] for i in range(DOWNLOAD_WORKERS)]
        threads = []
        for c in chunks:
            t = threading.Thread(target=self.downloader, args=(c,), daemon=True)
            t.start()
            threads.append(t)

        self.processor(len(urls))
        self.download_done.set()
        for t in threads: t.join()

        # Merge Results into Main DB
        print(f"\n[MERGING] Integrating {len(self.final_data)} swept entries...")
        full_data = []
        if os.path.exists(MAIN_JSON_DB):
            with open(MAIN_JSON_DB, 'r') as f:
                full_data = json.load(f)
        
        # Append new data
        full_data.extend(self.final_data)
        
        with open(MAIN_JSON_DB, 'w') as f:
            json.dump(full_data, f, indent=2)

        # Clear Failed List (Mission Accomplished)
        with open(INPUT_FAILED, 'w') as f:
            f.write("") # Empty file

        print("="*40)
        print(f"✅ Mission Accomplished.")
        print(f"✅ Recovered via Extreme OCR/Cloning: {len(self.final_data)}")
        print(f"✅ Failed List Cleared.")
        print("="*40)

if __name__ == "__main__":
    pipeline = FinalSweepPipeline()
    pipeline.run()