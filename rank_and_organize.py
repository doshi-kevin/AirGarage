#!/usr/bin/env python3
"""
Airgarage Step 5: Rank & Organize (Detailed)
Purpose: Groups plates by size (1, 2, 3...10+) and sorts them.
Output: 'ranked_plates.json'
Structure:
- Section 1: All Singles (Count=1)
- Section 2: All Exact Pairs (Count=2)
- Section 3: All Triples (Count=3)
...
- Section 10: All Decuples (Count=10)
"""
import json
import os
from collections import defaultdict

# ============= CONFIG =============
INPUT_DB = "vehicle_metadata.json"
OUTPUT_RANKED = "ranked_plates.json"
MAX_STAT_REPORT = 10  # Show specific stats up to this group size
# ==================================

def rank_data():
    if not os.path.exists(INPUT_DB):
        print(f"❌ File {INPUT_DB} not found.")
        return

    print(f"[LOADING] Reading {INPUT_DB}...")
    with open(INPUT_DB, 'r') as f:
        data = json.load(f)

    # 1. Group by Plate
    print(f"[GROUPING] Organizing {len(data)} records...")
    groups = defaultdict(list)
    for entry in data:
        plate = entry.get('plate', '').strip().upper()
        if not plate: continue
        groups[plate].append(entry)

    # 2. Convert to List of Groups
    group_list = []
    for plate, entries in groups.items():
        # Sort entries inside the group by URL (A-Z)
        entries.sort(key=lambda x: x['url'])
        
        # Tag entries with their group size
        for e in entries:
            e['group_size'] = len(entries)
            
        group_list.append({
            "count": len(entries),
            "plate": plate,
            "entries": entries
        })

    # 3. Sort Groups
    # Primary: Count (1 -> 2 -> 3 -> 4...)
    # Secondary: Plate Name (A -> Z)
    print(f"[RANKING] Sorting groups (Singles -> Pairs -> ... -> {MAX_STAT_REPORT}+)...")
    group_list.sort(key=lambda x: (x['count'], x['plate']))

    # 4. Flatten & Stats
    ranked_output = []
    stats = defaultdict(int)

    for g in group_list:
        stats[g['count']] += 1
        ranked_output.extend(g['entries'])

    # 5. Save
    print(f"[SAVING] Writing ranked data to {OUTPUT_RANKED}...")
    with open(OUTPUT_RANKED, 'w') as f:
        json.dump(ranked_output, f, indent=2)

    # --- DETAILED STATS REPORT ---
    print("\n" + "="*50)
    print("RANKING STATISTICS (Distribution)")
    print("="*50)
    
    # Loop 1 to 10 (or Config limit)
    for i in range(1, MAX_STAT_REPORT + 1):
        num_groups = stats.get(i, 0)
        total_imgs = num_groups * i
        
        label = f"Size {i}"
        if i == 1: label = "Singles (1)"
        elif i == 2: label = "Pairs (2)"
        
        print(f"  {label:<15}: {num_groups:6d} groups ({total_imgs:7d} images)")
    
    # Sum up anything larger than 10
    large_groups = sum(stats[k] for k in stats if k > MAX_STAT_REPORT)
    large_imgs = sum(stats[k]*k for k in stats if k > MAX_STAT_REPORT)
    
    if large_groups > 0:
        print("-" * 50)
        print(f"  Size >{MAX_STAT_REPORT:<14}: {large_groups:6d} groups ({large_imgs:7d} images)")

    print("="*50)
    print(f"✅ Total Images: {len(ranked_output)}")
    print(f"📂 Sorted File:  {OUTPUT_RANKED}")
    print("="*50)

if __name__ == "__main__":
    rank_data()