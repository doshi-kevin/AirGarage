#!/usr/bin/env python3
"""
Airgarage High-Speed Feature Extraction
Optimization: Moves CPU-heavy math (blur/color stats) to download threads.
Result: GPU runs 100% of the time without waiting.
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
import time
from tqdm import tqdm
from fast_alpr import ALPR

# ============= CONFIG =============
INPUT_FILE = "vehicle_images_input_part2 (4).txt"
OUTPUT_JSON = "vehicle_metadata.json"
OUTPUT_FAILED = "failed_extraction.txt"

# Concurrency - CRANKED UP
DOWNLOAD_WORKERS = 64    # Higher worker count to handle CPU math + Downloading
QUEUE_SIZE = 100         # Slightly larger buffer

# OCR Thresholds
CONF_THRESHOLD_HIGH = 0.75
CONF_THRESHOLD_LOW = 0.35
# ==================================

# ============= GPU SETUP =============
def setup_gpu_paths():
    base_prefix = sys.prefix
    site_packages = []
    if sys.platform != 'win32':
        site_packages = glob.glob(f"{base_prefix}/lib/python*/site-packages")
    else:
        site_packages = [f"{base_prefix}/Lib/site-packages"]

    nvidia_libs = []
    for sp in site_packages:
        for lib in ["cudnn", "cublas", "cuda_runtime"]:
            found = glob.glob(f"{sp}/nvidia/{lib}/lib")
            if found: nvidia_libs.extend(found)
    
    if nvidia_libs:
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_libs) + ":" + current_ld

setup_gpu_paths()
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
# =====================================

class FastMetadataPipeline:
    def __init__(self):
        print("[INIT] Loading AI Models...")
        # Force GPU
        try:
            self.alpr = ALPR(
                detector_model="yolo-v9-s-608-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model",
                detector_providers=['CUDAExecutionProvider'],
                ocr_providers=['CUDAExecutionProvider']
            )
            print("[INIT] ✅ GPU Mode Active")
        except:
            print("[INIT] ⚠️ GPU Failed, falling back to CPU")
            self.alpr = ALPR(
                detector_model="yolo-v9-t-256-license-plate-end2end",
                ocr_model="cct-xs-v1-global-model", 
                detector_providers=['CPUExecutionProvider'],
                ocr_providers=['CPUExecutionProvider']
            )

        self.gpu_queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.success_data = []
        self.failed_urls = []
        self.download_done = threading.Event()

    def clean_plate(self, text):
        if not text: return ""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    # --- OFF-LOADED TO WORKER THREADS ---
    def get_image_metrics(self, img):
        """Calculates stats on CPU."""
        try:
            h, w, _ = img.shape
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Blur score (Variance of Laplacian)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            # Fast average color
            avg_color = np.average(np.average(img, axis=0), axis=0)
            rgb = [int(avg_color[2]), int(avg_color[1]), int(avg_color[0])]

            return {
                "width": w, "height": h,
                "blur_score": round(blur_score, 2),
                "brightness": round(brightness, 2),
                "avg_color_rgb": rgb
            }
        except:
            return None

    # --- GPU THREAD LOGIC (AI ONLY) ---
    def preprocess_zoom(self, img, crop_box):
        """Helper to zoom in on plates if first pass fails"""
        try:
            x1, y1, x2, y2 = map(int, crop_box)
            h, w = img.shape[:2]
            pad = int((x2 - x1) * 0.1)
            x1, y1 = max(0, x1-pad), max(0, y1-pad)
            x2, y2 = min(w, x2+pad), min(h, y2+pad)
            roi = img[y1:y2, x1:x2]
            if roi.size == 0: return None
            # Upscale 2.5x
            return cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)
        except:
            return None

    def smart_predict(self, img_np):
        # 1. Fast Pass
        img_np = np.ascontiguousarray(img_np)
        results = self.alpr.predict(img_np)
        best_res = self._extract_best(results)
        
        # If good confidence, return immediately (FAST PATH)
        if best_res['conf'] > CONF_THRESHOLD_HIGH:
            return best_res

        # 2. Recovery Pass (Zoom) - Only if we found a box but couldn't read text
        if results and hasattr(results[0], 'detection'):
            det = results[0].detection
            box = None
            if hasattr(det, 'box'): box = det.box
            elif hasattr(det, 'xyxy'): box = det.xyxy
            elif hasattr(det, 'x1'): box = [det.x1, det.y1, det.x2, det.y2]

            if box:
                zoomed = self.preprocess_zoom(img_np, box)
                if zoomed is not None:
                    zoomed = np.ascontiguousarray(zoomed)
                    z_results = self.alpr.predict(zoomed)
                    z_best = self._extract_best(z_results)
                    if z_best['conf'] > best_res['conf']:
                        best_res = z_best

        return best_res

    def _extract_best(self, alpr_results):
        text, conf = "", 0.0
        if not alpr_results: return {'plate': text, 'conf': conf}
        for p in alpr_results:
            if hasattr(p, 'ocr') and p.ocr:
                if p.ocr.confidence > conf:
                    conf = p.ocr.confidence
                    text = p.ocr.text
        return {'plate': self.clean_plate(text), 'conf': conf}

    # --- WORKERS ---
    def downloader_worker(self, urls):
        """Downloads AND calculates metrics to save GPU time."""
        with requests.Session() as s:
            for url in urls:
                try:
                    r = s.get(url, timeout=4)
                    if r.status_code == 200:
                        arr = np.frombuffer(r.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            # HEAVY MATH HAPPENS HERE NOW
                            metrics = self.get_image_metrics(img)
                            self.gpu_queue.put((url, img, metrics))
                        else:
                            self.gpu_queue.put((url, None, None))
                    else:
                        self.gpu_queue.put((url, None, None))
                except:
                    self.gpu_queue.put((url, None, None))

    def gpu_worker(self, total):
        pbar = tqdm(total=total, desc="🚀 Speed Extraction")
        processed = 0
        while processed < total:
            try:
                try:
                    # Get pre-processed data
                    url, img, metrics = self.gpu_queue.get(timeout=5)
                except queue.Empty:
                    if self.download_done.is_set(): break
                    continue

                if img is None:
                    self.failed_urls.append(url)
                else:
                    try:
                        # GPU ONLY DOES INFERENCE NOW
                        result = self.smart_predict(img)
                        
                        if result['conf'] >= CONF_THRESHOLD_LOW and len(result['plate']) >= 3:
                            data = {
                                "url": url,
                                "plate": result['plate'],
                                "conf": round(result['conf'], 4),
                                "meta": metrics
                            }
                            self.success_data.append(data)
                        else:
                            self.failed_urls.append(url)
                    except:
                        self.failed_urls.append(url)

                processed += 1
                pbar.update(1)
                self.gpu_queue.task_done()
            except:
                processed += 1
                pbar.update(1)
        pbar.close()

    def run(self):
        with open(INPUT_FILE) as f: urls = [x.strip() for x in f if x.strip()]
        
        # Threads
        chunks = [urls[i::DOWNLOAD_WORKERS] for i in range(DOWNLOAD_WORKERS)]
        threads = []
        for c in chunks:
            t = threading.Thread(target=self.downloader_worker, args=(c,), daemon=True)
            t.start()
            threads.append(t)

        # GPU
        self.gpu_worker(len(urls))
        self.download_done.set()
        for t in threads: t.join()

        # Save
        print(f"\n[SAVING] JSON...")
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(self.success_data, f, indent=2)
        
        with open(OUTPUT_FAILED, 'w') as f:
            for u in self.failed_urls:
                f.write(f"{u}\n")
        
        print(f"Done. Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    pipeline = FastMetadataPipeline()
    pipeline.run()