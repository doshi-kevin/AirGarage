import gradio as gr
import json
import os
import re
from collections import defaultdict

# ================= CONFIGURATION =================
DATA_FILE = "ranked_plates.json"
OUTPUT_FILE = "output_pairs.txt"

# ================= DATA PROCESSING =================
def normalize_plate(p):
    """Standardizes plate numbers to group them effectively."""
    if not p: return "UNKNOWN"
    return re.sub(r'[^A-Z0-9]', '', str(p).upper())

def load_data():
    """Loads JSON, groups by plate, and sorts by group size."""
    if not os.path.exists(DATA_FILE):
        print(f"Warning: {DATA_FILE} not found. Creating empty dataset.")
        return []

    try:
        with open(DATA_FILE, 'r') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

    # Flatten list if necessary
    items = []
    if isinstance(raw_data, list):
        for item in raw_data:
            if isinstance(item, list): items.extend(item)
            else: items.append(item)

    # Group by Normalized Plate
    grouped = defaultdict(list)
    for entry in items:
        plate = entry.get('plate', 'UNKNOWN')
        clean_plate = normalize_plate(plate)
        
        img_data = {
            "url": entry.get('url', ''),
            "plate": plate,
            "score": entry.get('score', entry.get('confidence', 'N/A')),
            "meta": entry.get('meta', {})
        }
        grouped[clean_plate].append(img_data)

    sorted_groups = []
    for plate, images in grouped.items():
        sorted_groups.append({
            "group_plate": plate,
            "images": images,
            "count": len(images)
        })

    # Sort: Largest groups first
    sorted_groups.sort(key=lambda x: (-x['count'], x['group_plate']))
    
    print(f"Loaded {len(sorted_groups)} groups.")
    return sorted_groups

# Load Data
DATASET = load_data()

# ================= APP LOGIC =================

def get_current_group_data(index):
    if not DATASET: return None
    idx = max(0, min(index, len(DATASET)-1))
    return DATASET[idx]

def refresh_ui(index, _):
    """Refreshes the Gallery and Headers."""
    if not DATASET:
        return [
            [], 
            "## No Data Found", 
            "Please check ranked_plates.json",
            {},
            gr.update(interactive=False)
        ]

    group = get_current_group_data(index)
    
    # Header Info
    header = f"### 🚗 Group {index + 1} / {len(DATASET)} : Plate `{group['group_plate']}` ({group['count']} images)"
    
    # Prepare Gallery: List of (url, label)
    gallery_data = []
    for img in group['images']:
        label = f"{img['plate']} | Conf: {img['score']}"
        gallery_data.append((img['url'], label))

    return [
        gallery_data,
        header,
        "Select images to view details...",
        None,
        gr.update(interactive=False)
    ]

def on_select(evt: gr.SelectData, index, current_selections):
    """Handles image selection."""
    group = get_current_group_data(index)
    selected_index = evt.index
    selected_img_data = group['images'][selected_index]
    selected_url = selected_img_data['url']

    # Update Selections (Max 2, FIFO)
    new_selections = list(current_selections)
    if selected_url in new_selections:
        new_selections.remove(selected_url)
    else:
        if len(new_selections) < 2:
            new_selections.append(selected_url)
        else:
            new_selections.pop(0)
            new_selections.append(selected_url)

    # Metadata View
    meta_view = {
        "EXTRACTED_PLATE": selected_img_data['plate'],
        "CONFIDENCE": selected_img_data['score'],
        "URL": selected_url,
        "METADATA": selected_img_data['meta']
    }

    status = f"**Selected: {len(new_selections)} / 2**"
    can_save = (len(new_selections) == 2)

    return new_selections, meta_view, status, gr.update(interactive=can_save)

def save_pair(current_selections):
    """Writes to file."""
    if len(current_selections) != 2:
        return "Error: Select 2 images", gr.update(), current_selections
    
    line = f"{current_selections[0]},{current_selections[1]}\n"
    
    try:
        with open(OUTPUT_FILE, 'a') as f:
            f.write(line)
        return f"✅ Saved Pair!", gr.update(value=[], interactive=False), []
    except Exception as e:
        return f"Error: {e}", gr.update(), current_selections

def navigation(action, current_index):
    new_index = current_index + (1 if action == "next" else -1)
    new_index = max(0, min(new_index, len(DATASET) - 1))
    return new_index, []

# ================= UI LAYOUT =================
# FIX: Removed arguments from Blocks() to prevent crashes in newer Gradio versions
with gr.Blocks() as app:
    
    # State
    current_index = gr.State(0)
    selected_urls = gr.State([])

    gr.Markdown("# 🚘 AirGarage Pair Merger")

    # Header & Nav
    with gr.Row():
        with gr.Column(scale=3):
            header_display = gr.Markdown("### Loading...")
        with gr.Column(scale=1):
            with gr.Row():
                prev_btn = gr.Button("⬅️ Prev")
                next_btn = gr.Button("Next ➡️", variant="primary")

    # Main Area
    with gr.Row():
        # Gallery
        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="Image Group", 
                columns=[3], 
                height=600, 
                object_fit="contain",
                allow_preview=False,
                interactive=True
            )

        # Sidebar
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Inspector")
            status_display = gr.Markdown("Selected: 0 / 2")
            save_btn = gr.Button("🔗 SAVE PAIR", variant="stop", interactive=False)
            gr.Markdown("---")
            gr.Markdown("**Live Image Details:**")
            json_inspector = gr.JSON(label="Extracted Data")


    # Events
    app.load(refresh_ui, inputs=[current_index, selected_urls], outputs=[gallery, header_display, status_display, json_inspector, save_btn])

    gallery.select(
        on_select, 
        inputs=[current_index, selected_urls], 
        outputs=[selected_urls, json_inspector, status_display, save_btn]
    )

    save_btn.click(
        save_pair,
        inputs=[selected_urls],
        outputs=[status_display, save_btn, selected_urls]
    )

    prev_btn.click(lambda idx: navigation("prev", idx), inputs=[current_index], outputs=[current_index, selected_urls]).then(
        refresh_ui, inputs=[current_index, selected_urls], outputs=[gallery, header_display, status_display, json_inspector, save_btn]
    )

    next_btn.click(lambda idx: navigation("next", idx), inputs=[current_index], outputs=[current_index, selected_urls]).then(
        refresh_ui, inputs=[current_index, selected_urls], outputs=[gallery, header_display, status_display, json_inspector, save_btn]
    )

if __name__ == "__main__":
    # FIX: Pass allowed_paths=['.'] to ensure local files/jsons can be read if needed
    app.launch(allowed_paths=['.'])