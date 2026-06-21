import gradio as gr
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import zipfile
import tempfile

# --- THEME CONFIGURATION ---
fidel_theme = gr.themes.Soft(
    primary_hue="amber",
    secondary_hue="stone",
    neutral_hue="stone",
).set(
    # Backgrounds
    body_background_fill="#1a120b",        # Deep dark brown
    block_background_fill="#2b1d16",       # Slightly lighter brown for cards
    
    # Block Labels / Titles (e.g., "Edit Single Fidel Data")
    block_label_text_color="#d1d1d1",      # Light Grey
    block_title_text_color="#d1d1d1",      # Light Grey
    
    # Table Styling
    table_header_text_color="#a67c52",     # Warm Brown for headers
    table_header_background_fill="#3d2b1f", # Darker brown for header background
    table_text_color="#e3d5ca",            # Soft cream for table content
    
    # General Text
    body_text_color="#e3d5ca",             # Soft cream/tan text
    
    # Borders and Inputs
    block_border_width="1px",
    border_color_primary="#4a3a2e",
    input_background_fill="#1a120b",
    
    # Buttons
    button_primary_background_fill="#5c3d2e",
    button_primary_text_color="white",
    button_secondary_background_fill="#3d2b1f",
    button_secondary_text_color="#e3d5ca",
)

COL_NAMES = ["ID", "-Left/+Right", "-Up/+Down", "Width", "Height", "Label"]

def detect_fidels(img):
    if img is None: return []
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 8 and h > 8: raw_boxes.append([x, y, w, h])
                
        sorted_boxes = sorted(raw_boxes, key=lambda b: (b[1] // 50, b[0]))
        return [[i, b[0], b[1], b[2], b[3], ""] for i, b in enumerate(sorted_boxes)]
    except Exception as e:
        print(f"DETECTION ERROR: {e}"); return []

def update_ui_and_filter(img, master_df, focus_id):
    if img is None or master_df is None or len(master_df) == 0:
        return img, None, pd.DataFrame(columns=COL_NAMES)
    
    canvas = img.copy()
    zoom_crop = None
    thickness = max(2, int(canvas.shape[1] / 1200) * 2)
    focus_id = int(focus_id)

    for _, row in master_df.iterrows():
        idx = int(row["ID"])
        x, y, w, h = int(row["-Left/+Right"]), int(row["-Up/+Down"]), int(row["Width"]), int(row["Height"])
        
        is_focused = (idx == focus_id)
        color = (255, 191, 0) if is_focused else (0, 165, 255) 
        curr_t = thickness + 4 if is_focused else thickness
        
        if is_focused:
            pad = 15
            y_s, y_e = max(0, y-pad), min(img.shape[0], y+h+pad)
            x_s, x_e = max(0, x-pad), min(img.shape[1], x+w+pad)
            zoom_crop = img[y_s:y_e, x_s:x_e]
        
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, curr_t)
        cv2.putText(canvas, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, curr_t)
    
    filtered_df = master_df[master_df["ID"] == focus_id].copy()
    return canvas, zoom_crop, filtered_df

def on_image_click(img, master_df, evt: gr.SelectData):
    if master_df is None: return 0, None, None, None
    click_x, click_y = evt.index[0], evt.index[1]
    
    for _, row in master_df.iterrows():
        x, y, w, h = int(row["-Left/+Right"]), int(row["-Up/+Down"]), int(row["Width"]), int(row["Height"])
        if x <= click_x <= x + w and y <= click_y <= y + h:
            f_id = int(row["ID"])
            p, z, f = update_ui_and_filter(img, master_df, f_id)
            return f_id, p, z, f
    return gr.update(), gr.update(), gr.update(), gr.update()

def process_upload(img):
    if img is None: return None, None, pd.DataFrame(columns=COL_NAMES), 0, None, None
    boxes = detect_fidels(img)
    m_df = pd.DataFrame(boxes, columns=COL_NAMES).astype(object)
    p, z, f = update_ui_and_filter(img, m_df, 0)
    return p, z, f, 0, m_df, m_df

def save_changes(filtered_df, master_df, img, focus_id):
    try:
        if filtered_df is None or master_df is None or len(filtered_df) == 0:
            return master_df, master_df, gr.update(), gr.update(), gr.update(), "⚠️ Nothing to save."
        
        master_df = master_df.copy()
        row_data = filtered_df.iloc[0]
        current_id = int(float(row_data["ID"]))
        mask = master_df["ID"] == current_id
        
        if not mask.any():
            return master_df, master_df, gr.update(), gr.update(), gr.update(), f"❌ ID {current_id} not found."

        master_df.loc[mask, "-Left/+Right"] = int(float(row_data["-Left/+Right"]))
        master_df.loc[mask, "-Up/+Down"] = int(float(row_data["-Up/+Down"]))
        master_df.loc[mask, "Width"] = int(float(row_data["Width"]))
        master_df.loc[mask, "Height"] = int(float(row_data["Height"]))
        master_df.loc[mask, "Label"] = str(row_data["Label"])

        p, z, f_updated = update_ui_and_filter(img, master_df, focus_id)
        return master_df, master_df, p, z, f_updated, f"✅ Saved ID {current_id}"
    except Exception as e:
        return master_df, master_df, gr.update(), gr.update(), gr.update(), f"❌ Save Error: {str(e)}"

def export_dataset(img, master_df):
    if img is None or master_df is None: return None
    temp_dir = tempfile.mkdtemp()
    img_dir = os.path.join(temp_dir, "fidels")
    os.makedirs(img_dir)
    meta = []
    count = 0
    for _, row in master_df.iterrows():
        try:
            label = str(row["Label"]).strip()
            if not label or label == "" or label.lower() == "none": continue
            x, y, w, h = int(row["-Left/+Right"]), int(row["-Up/+Down"]), int(row["Width"]), int(row["Height"])
            crop = img[y:y+h, x:x+w]
            if crop.size == 0: continue
            pil_img = Image.fromarray(crop).convert('L')
            pil_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
            new_img = Image.new("L", (32, 32), (255))
            new_img.paste(pil_img, ((32 - pil_img.size[0]) // 2, (32 - pil_img.size[1]) // 2))
            fname = f"fidel_{int(row['ID'])}.png"
            new_img.save(os.path.join(img_dir, fname))
            meta.append({"file": fname, "label": label, "original_id": int(row['ID'])})
            count += 1
        except: continue
    if count == 0: return None
    pd.DataFrame(meta).to_csv(os.path.join(temp_dir, "metadata.csv"), index=False)
    zip_path = os.path.join(tempfile.gettempdir(), "geez_dataset.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for r, _, fs in os.walk(temp_dir):
            for f in fs: zf.write(os.path.join(r, f), arcname=f)
    return zip_path

# --- UI ---
with gr.Blocks(theme=fidel_theme, title="Ge'ez Dataset Creator") as demo:
    master_state = gr.State()
    gr.Markdown("# Ge'ez Fidel Dataset Creator for Machine Learning")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Scan here", type="numpy")
            with gr.Group():
                focus_id = gr.Number(label="Target ID", value=0, precision=0)
                zoom_view = gr.Image(label="Zoomed View", interactive=False)
            status_msg = gr.Markdown("**Ready.**")
            export_btn = gr.Button("Build & Download Labeled Dataset", variant="primary")
            download_link = gr.File(label="Output Zip")
            
        with gr.Column(scale=2):
            preview_img = gr.Image(label="Full Preview (CLICK TO SELECT CHARACTER)")
            
            current_row_table = gr.Dataframe(
                headers=COL_NAMES, 
                datatype=["number", "number", "number", "number", "number", "str"], 
                interactive=True, 
                label="Edit Single Fidel Data"
            )
            save_btn = gr.Button("💾 Save Edits & Refresh View", variant="secondary")

    gr.Markdown("### 📋 Main Dataset Table (Full Progress)")
    master_table_view = gr.Dataframe(headers=COL_NAMES, interactive=False)

    input_img.upload(process_upload, inputs=input_img, outputs=[preview_img, zoom_view, current_row_table, focus_id, master_state, master_table_view])
    preview_img.select(on_image_click, [input_img, master_state], [focus_id, preview_img, zoom_view, current_row_table])
    focus_id.change(update_ui_and_filter, [input_img, master_state, focus_id], [preview_img, zoom_view, current_row_table])
    save_btn.click(save_changes, inputs=[current_row_table, master_state, input_img, focus_id], outputs=[master_state, master_table_view, preview_img, zoom_view, current_row_table, status_msg])
    export_btn.click(export_dataset, [input_img, master_state], download_link)

demo.launch()