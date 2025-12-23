#!/usr/bin/env python3
"""
Airgarage Exact Pair Segregator
Rule: STRICTLY matches plates that appear EXACTLY twice.
- Count == 2: Saved as Fixed Pair.
- Count == 1: Saved as Single (for visual matching later).
- Count >= 3: Saved as Multiple (Ambiguous, discarded from pairs).
"""
import json
from collections import defaultdict

# ============= CONFIG =============
INPUT_JSON = "vehicle_metadata.json"

# Outputs
OUTPUT_PAIRS = "fixed_pairs.txt"           # The Gold Standard pairs
OUTPUT_SINGLES = "unpaired_singles.json"   # Count = 1
OUTPUT_MULTIPLES = "ambiguous_multiples.json" # Count >= 3
# ==================================

def process_strict_pairs():
    print(f"[LOADING] Reading {INPUT_JSON}...")
    try:
        with open(INPUT_JSON, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: {INPUT_JSON} not found. Please run the extraction script first.")
        return

    # 1. Group all data by Plate Number
    print(f"[PROCESSING] analyzing {len(data)} records...")
    groups = defaultdict(list)
    for entry in data:
        # Normalize plate just in case (remove spaces/special chars)
        plate = entry['plate'].strip().upper()
        groups[plate].append(entry)

    pairs = []
    singles = []
    multiples = []

    # 2. Apply STRICT Filtering Logic
    for plate, entries in groups.items():
        count = len(entries)
        
        if count == 2:
            # ✅ STRICT MATCH: Exactly 2 images share this plate
            # Sort by URL to keep the pair order deterministic
            entries.sort(key=lambda x: x['url'])
            pairs.append((entries[0], entries[1]))
            
        elif count == 1:
            # ⚠️ SINGLE: Only 1 image has this plate
            singles.extend(entries)
            
        else:
            # ❌ MULTIPLE: 3 or more images share this plate. 
            # These are discarded from the pairs file as per instructions.
            multiples.extend(entries)

    # 3. Write Results
    
    # Save Valid Pairs
    print(f"[SAVING] Writing {len(pairs)} pairs to {OUTPUT_PAIRS}...")
    with open(OUTPUT_PAIRS, 'w') as f:
        for p1, p2 in pairs:
            # Writing format: url1,url2
            f.write(f"{p1['url']},{p2['url']}\n")

    # Save Singles (Metadata preserved for visual matching later)
    with open(OUTPUT_SINGLES, 'w') as f:
        json.dump(singles, f, indent=2)

    # Save Multiples (For debugging)
    with open(OUTPUT_MULTIPLES, 'w') as f:
        json.dump(multiples, f, indent=2)

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print("STRICT SEGREGATION RESULTS")
    print("="*50)
    print(f"✅ EXACT PAIRS (Count=2):   {len(pairs)} pairs")
    print(f"   (Ready for submission: {OUTPUT_PAIRS})")
    print("-" * 50)
    print(f"⚠️  SINGLES (Count=1):      {len(singles)} images")
    print(f"   (Saved to: {OUTPUT_SINGLES})")
    print("-" * 50)
    print(f"❌ MULTIPLES (Count>=3):    {len(multiples)} images")
    print(f"   (Excluded from pairs: {OUTPUT_MULTIPLES})")
    print("="*50)

if __name__ == "__main__":
    process_strict_pairs()