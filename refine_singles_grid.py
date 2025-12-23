#!/usr/bin/env python3
"""
Airgarage Phase 4: Smart Refinement (Lazy Grid Search)
Optimization:
1. "Lazy" Search: Tries Center Zoom first. If good, skips the rest. (3x Faster)
2. Anti-Freeze: Uses a Sentinel-based queue system so it never gets stuck at 99%.
3. Output: Updates 'vehicle_metadata.json' in place.
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
import time
from tqdm import tqdm

# ============= CONFIG =============
INPUT_SINGLES = "unpaired_singles.json"
MAIN_JSON_DB = "vehicle_metadata.json"

# Performance
DOWNLOAD_WORKERS = 48  # High IO
QUEUE_SIZE = 200       # Bigger buffer
CONF_ACCEPTABLE = 0.70 # If we hit this, stop searching this image

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
                try: ctypes.CDLL(lib)
                except: pass

force_load_nvidia_libs()
os.environ["ORT_TENSORRT_UNAVAILABLE"] = "1"
from fast_alpr import ALPR
# =====================================

class SmartRefiner:
    def __init__(self):
        print("[INIT] Loading AI Models...")
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
        self.updates = {}
        self.sentinel_count = 0

    def clean_plate(self, text):
        if not text: return ""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    # --- LAZY GRID LOGIC ---
    def smart_grid_search(self, img):
        """
        Tries Center first. Only tries others if Center fails.
        """
        h, w = img.shape[:2]
        best_plate = ""
        best_conf = 0.0

        # 1. Define Crops Priority
        # Priority A: Center Zoom (Most likely)
        y1, y2 = int(h * 0.20), int(h * 0.80)
        x1, x2 = int(w * 0.20), int(w * 0.80)
        center_crop = cv2.resize(img[y1:y2, x1:x2], None, fx=1.3, fy=1.3)
        
        # Priority B: Quadrants
        h_mid, w_mid = h // 2, w // 2
        h_over, w_over = int(h * 0.15), int(w * 0.15)
        quadrants = [
            img[0 : h_mid+h_over, 0 : w_mid+w_over], # TL
            img[h_mid-h_over : h, w_mid-w_over : w], # BR
            img[0 : h_mid+h_over, w_mid-w_over : w], # TR
            img[h_mid-h_over : h, 0 : w_mid+w_over]  # BL
        ]

        # 2. Run Priority A
        p, c = self._predict_one(center_crop)
        if c > best_conf: best_plate, best_conf = p, c
        
        # FAST EXIT: If Center is good, stop here!
        if best_conf > CONF_ACCEPTABLE:
            return best_plate, best_conf

        # 3. Run Priority B (Quadrants) only if needed
        for crop in quadrants:
            p, c = self._predict_one(crop)
            if c > best_conf: best_plate, best_conf = p, c
            # Semi-Fast Exit
            if best_conf > CONF_ACCEPTABLE:
                return best_plate, best_conf

        return best_plate, best_conf

    def _predict_one(self, img_np):
        if img_np.size == 0: return "", 0.0
        img_np = np.ascontiguousarray(img_np)
        try:
            results = self.alpr.predict(img_np)
            if results:
                for p in results:
                    if hasattr(p, 'ocr') and p.ocr:
                        return self.clean_plate(p.ocr.text), p.ocr.confidence
        except: pass
        return "", 0.0

    # --- WORKERS ---
    def downloader(self, items):
        with requests.Session() as s:
            for item in items:
                try:
                    r = s.get(item['url'], timeout=3) # Strict timeout
                    if r.status_code == 200:
                        arr = np.frombuffer(r.content, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        self.gpu_queue.put((item, img))
                    else:
                        self.gpu_queue.put((item, None)) # Signal failure
                except:
                    self.gpu_queue.put((item, None)) # Signal failure
        
        # Send Sentinel when this thread is done
        self.gpu_queue.put("SENTINEL")

    def processor(self, total_items, total_threads):
        pbar = tqdm(total=total_items, desc="🚀 Smart Refining")
        processed = 0
        finished_downloaders = 0
        updated = 0

        while True:
            try:
                # Non-blocking check if possible, or short timeout
                data = self.gpu_queue.get(timeout=2)
                
                # Check for Thread End Signal
                if data == "SENTINEL":
                    finished_downloaders += 1
                    if finished_downloaders == total_threads and self.gpu_queue.empty():
                        break # All done
                    continue
                
                item, img = data
                
                if img is not None:
                    # Smart Search
                    new_plate, new_conf = self.smart_grid_search(img)
                    
                    # Logic: Only keep if better
                    old_conf = item.get('conf', 0.0)
                    
                    # If we found a plate where there was none, or improved confidence significantly
                    if len(new_plate) >= 3:
                        if new_conf > (old_conf + 0.05) or len(new_plate) > len(item.get('plate', '')):
                            self.updates[item['url']] = {
                                "plate": new_plate,
                                "conf": round(new_conf, 4)
                            }
                            updated += 1
                
                processed += 1
                pbar.update(1)
                self.gpu_queue.task_done()

            except queue.Empty:
                # If queue is empty but downloaders aren't done, wait.
                # If downloaders ARE done, we exit.
                if finished_downloaders == total_threads:
                    break
        
        pbar.close()
        return updated

    def run(self):
        if not os.path.exists(INPUT_SINGLES):
            print("No unpaired singles file.")
            return

        with open(INPUT_SINGLES) as f:
            singles = json.load(f)
        
        print(f"Loaded {len(singles)} singles.")
        
        # Split work
        chunks = [singles[i::DOWNLOAD_WORKERS] for i in range(DOWNLOAD_WORKERS)]
        threads = []
        for c in chunks:
            if not c: continue
            t = threading.Thread(target=self.downloader, args=(c,), daemon=True)
            t.start()
            threads.append(t)

        updated_count = self.processor(len(singles), len(threads))
        
        # Join threads just to be safe (they should be dead)
        for t in threads: t.join()

        # Update Database
        print(f"\n[MERGING] Updating {updated_count} improved records...")
        
        with open(MAIN_JSON_DB, 'r') as f:
            full_data = json.load(f)

        count = 0
        for entry in full_data:
            if entry['url'] in self.updates:
                up = self.updates[entry['url']]
                entry['plate'] = up['plate']
                entry['conf'] = up['conf']
                entry['refined'] = True
                count += 1
        
        with open(MAIN_JSON_DB, 'w') as f:
            json.dump(full_data, f, indent=2)

        print("="*40)
        print(f"✅ Refined & Updated: {count} images")
        print("="*40)

if __name__ == "__main__":
    sr = SmartRefiner()
    sr.run()