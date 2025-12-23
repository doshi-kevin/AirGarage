import streamlit as st
import json
import pandas as pd
import math
from collections import defaultdict

# ================= CONFIGURATION =================
PAGE_TITLE = "🚗 AirGarage Neural Visualizer"
DEFAULT_JSON_FILE = "ranked_plates.json"  # Falls back to vehicle_metadata.json if missing
ITEMS_PER_PAGE = 20  # Number of PLATES per page (not images)
# =================================================

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    .plate-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stImage {
        border-radius: 8px;
        transition: transform 0.2s;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    /* Hide fullscreen button on images to keep UI clean */
    button[title="View fullscreen"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADER (CACHED) ---
@st.cache_data(show_spinner=True)
def load_and_group_data(filename):
    """
    Loads JSON and groups it by plate.
    Returns: dict {plate: [list of entries]}, dict (stats)
    """
    data = []
    
    # Try loading the preferred file, fallback to main DB
    files_to_try = [filename, "vehicle_metadata.json", "ranked_plates.json"]
    loaded_file = None
    
    for f in files_to_try:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
                loaded_file = f
                break
        except FileNotFoundError:
            continue
            
    if not data:
        return {}, {"error": "No JSON file found. Please ensure 'ranked_plates.json' exists."}

    # Grouping Logic
    grouped = defaultdict(list)
    total_images = 0
    
    for entry in data:
        plate = entry.get('plate', 'UNKNOWN').strip().upper()
        # Ensure we have a valid entry
        if plate:
            grouped[plate].append(entry)
            total_images += 1

    # Calculate Stats
    stats = {
        "file": loaded_file,
        "total_plates": len(grouped),
        "total_images": total_images,
        "pairs": sum(1 for v in grouped.values() if len(v) == 2),
        "singles": sum(1 for v in grouped.values() if len(v) == 1),
        "multiples": sum(1 for v in grouped.values() if len(v) > 2),
    }

    # Sort plates by size (Multiples first, then Pairs, then Singles)
    # Inside each category, sort alphanumerically
    sorted_plates = sorted(
        grouped.items(), 
        key=lambda x: (len(x[1]) == 2, len(x[1]) > 2, x[0]), 
        reverse=True
    )
    
    # Convert back to dict for easy access, but keep order
    return dict(sorted_plates), stats

# --- SIDEBAR CONTROLS ---
st.sidebar.title("🛠️ Mission Control")

# Load Data
plate_groups, stats = load_and_group_data(DEFAULT_JSON_FILE)

if "error" in stats:
    st.error(stats["error"])
    st.stop()

# Stats Display
with st.sidebar.expander("📊 Dataset Metrics", expanded=True):
    col_a, col_b = st.columns(2)
    col_a.metric("Plates", stats['total_plates'])
    col_b.metric("Images", stats['total_images'])
    st.divider()
    st.caption(f"Source: `{stats['file']}`")
    st.progress(stats['pairs'] / stats['total_plates'] if stats['total_plates'] else 0, text=f"Pairs: {stats['pairs']}")
    st.progress(stats['multiples'] / stats['total_plates'] if stats['total_plates'] else 0, text=f"Multiples: {stats['multiples']}")

# Filters
st.sidebar.header("🔍 Filters")
search_query = st.sidebar.text_input("Search Plate Number", "").upper()
filter_type = st.sidebar.radio(
    "Show Group Size:",
    ["All", "Exact Pairs (2)", "Multiples (3+)", "Singles (1)"],
    index=0
)

# Apply Filters
filtered_keys = list(plate_groups.keys())

# 1. Type Filter
if filter_type == "Exact Pairs (2)":
    filtered_keys = [k for k in filtered_keys if len(plate_groups[k]) == 2]
elif filter_type == "Multiples (3+)":
    filtered_keys = [k for k in filtered_keys if len(plate_groups[k]) > 2]
elif filter_type == "Singles (1)":
    filtered_keys = [k for k in filtered_keys if len(plate_groups[k]) == 1]

# 2. Search Filter
if search_query:
    filtered_keys = [k for k in filtered_keys if search_query in k]

# --- PAGINATION LOGIC ---
total_filtered = len(filtered_keys)
total_pages = math.ceil(total_filtered / ITEMS_PER_PAGE)

if total_pages > 1:
    page_number = st.sidebar.number_input(
        f"Page (1 - {total_pages})", 
        min_value=1, 
        max_value=total_pages, 
        value=1
    )
else:
    page_number = 1

start_idx = (page_number - 1) * ITEMS_PER_PAGE
end_idx = start_idx + ITEMS_PER_PAGE
current_batch_keys = filtered_keys[start_idx:end_idx]

# --- MAIN UI RENDER ---
st.title(PAGE_TITLE)
st.markdown(f"**Showing {len(current_batch_keys)} plates** (Indices {start_idx}-{end_idx} of {total_filtered})")
st.divider()

if not current_batch_keys:
    st.warning("No plates match your filter criteria.")
    st.stop()

# Loop through the current page's plates
for plate in current_batch_keys:
    entries = plate_groups[plate]
    count = len(entries)
    
    # Sort entries by URL to maintain timeline order
    entries.sort(key=lambda x: x.get('url', ''))

    # Determine Color/Icon based on group size
    if count == 2:
        icon = "✅"
        color = "green"
        label = "Verified Pair"
    elif count > 2:
        icon = "⚠️"
        color = "orange"
        label = "Ambiguous Group"
    else:
        icon = "❌"
        color = "red"
        label = "Single"

    # Container for the Plate Group
    with st.container():
        # Header
        st.markdown(f"""
        <div class='plate-header' style='border-left: 5px solid {color};'>
            <h3>{icon} {plate} <span style='font-size: 0.6em; color: gray;'>({count} images - {label})</span></h3>
        </div>
        """, unsafe_allow_html=True)

        # --- DYNAMIC LAYOUT ENGINE ---
        
        # SCENARIO A: PAIR (Side by Side)
        if count == 2:
            cols = st.columns(2)
            for i, entry in enumerate(entries):
                with cols[i]:
                    st.image(entry['url'], use_column_width=True)
                    # Metadata Stats
                    meta = entry.get('meta', {})
                    if meta:
                        st.caption(f"Blur: {meta.get('blur_score', '?')} | Bright: {meta.get('brightness', '?')}")
        
        # SCENARIO B: SINGLE (Centered)
        elif count == 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(entries[0]['url'], use_column_width=True)
                st.caption("Unpaired Single Entry")

        # SCENARIO C: MULTIPLES (Grid)
        else:
            # Grid of 4 columns
            grid_cols = 4
            rows = math.ceil(count / grid_cols)
            
            for row in range(rows):
                cols = st.columns(grid_cols)
                for c in range(grid_cols):
                    idx = row * grid_cols + c
                    if idx < count:
                        entry = entries[idx]
                        with cols[c]:
                            st.image(entry['url'], use_column_width=True)
                            st.caption(f"Img {idx+1}")

        # Expandable Details (JSON dump for debugging)
        with st.expander(f"View Raw Data for {plate}"):
            st.json(entries)
        
        st.markdown("---")  # Separator between plates

# Footer
st.markdown("<div style='text-align: center; color: gray;'>AirGarage Extraction Pipeline • v2.0</div>", unsafe_allow_html=True)