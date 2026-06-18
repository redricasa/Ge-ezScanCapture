import gradio as gr
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import zipfile
import tempfile

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
    """Refreshes images and filters the single-row editor table."""
    if img is None or master_df is None or len(master_df) == 0:
        return img, None, pd.DataFrame(columns=COL_NAMES)
    
    canvas = img.copy()
    zoom_crop = None
    thickness = max(2, int(canvas.shape[1] / 1200) * 2)

    for _, row in master_df.iterrows():
        idx, x, y, w, h = int(row["ID"]), int(row["-Left/+Right"]), int(row["-Up/+Down"]), int(row["Width"]), int(row["Height"])
        color = (255, 0, 0) if idx == focus_id else (0, 255, 0)
        curr_t = thickness + 4 if idx == focus_id else thickness
        
        if idx == focus_id:
            pad = 15
            y_s, y_e = max(0, y-pad), min(img.shape[0], y+h+pad)
            x_s, x_e = max(0, x-pad), min(img.shape[1], x+w+pad)
            zoom_crop = img[y_s:y_e, x_s:x_e]
        
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, curr_t)
        cv2.putText(canvas, str(idx), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, curr_t)
    
    filtered_df = master_df[master_df["ID"] == focus_id]
    return canvas, zoom_crop, filtered_df

def on_image_click(img, master_df, evt: gr.SelectData):
    """Finds which ID was clicked based on coordinates."""
    if master_df is None: return 0, None, None, None
    click_x, click_y = evt.index[0], evt.index[1]
    
    for _, row in master_df.iterrows():
        x, y, w, h = row["-Left/+Right"], row["-Up/+Down"], row["Width"], row["Height"]
        if x <= click_x <= x + w and y <= click_y <= y + h:
            focus_id = int(row["ID"])
            preview, zoom, filtered = update_ui_and_filter(img, master_df, focus_id)
            return focus_id, preview, zoom, filtered
    return gr.update(), gr.update(), gr.update(), gr.update()

def process_upload(img):
    if img is None: return None, None, pd.DataFrame(columns=COL_NAMES), 0, None, None
    boxes = detect_fidels(img)
    master_df = pd.DataFrame(boxes, columns=COL_NAMES)
    preview, zoom, filtered_df = update_ui_and_filter(img, master_df, 0)
    return preview, zoom, filtered_df, 0, master_df, master_df

def save_changes(filtered_df, master_df, img, focus_id):
    """Syncs the single-row edit back to the master list and refreshes UI."""
    if filtered_df is None or master_df is None or len(filtered_df) == 0:
        return master_df, master_df, gr.update(), gr.update(), gr.update()
    
    # Save edits from small table to master list
    current_id = filtered_df.iloc[0]["ID"]
    master_df.loc[master_df["ID"] == current_id, COL_NAMES] = filtered_df.iloc[0].values
    
    preview, zoom, filtered_updated = update_ui_and_filter(img, master_df, focus_id)
    return master_df, master_df, preview, zoom, filtered_updated

def export_dataset(img, master_df):
    if img is None or master_df is None: return None
    temp_dir = tempfile.mkdtemp()
    img_dir = os.path.join(temp_dir, "fidels")
    os.makedirs(img_dir)
    meta = []
    for _, row in master_df.iterrows():
        try:
            x, y, w, h, label = int(row["-Left/+Right"]), int(row["-Up/+Down"]), int(row["Width"]), int(row["Height"]), str(row["Label"])
            crop = img[y:y+h, x:x+w]
            if crop.size == 0: continue
            pil_img = Image.fromarray(crop).convert('L')
            pil_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
            new_img = Image.new("L", (32, 32), (255))
            new_img.paste(pil_img, ((32 - pil_img.size[0]) // 2, (32 - pil_img.size[1]) // 2))
            fname = f"fidel_{int(row['ID'])}.png"
            new_img.save(os.path.join(img_dir, fname))
            meta.append({"file": fname, "label": label})
        except: continue
    pd.DataFrame(meta).to_csv(os.path.join(temp_dir, "metadata.csv"), index=False)
    zip_path = os.path.join(tempfile.gettempdir(), "geez_dataset.zip")
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for r, _, fs in os.walk(temp_dir):
            for f in fs: zf.write(os.path.join(r, f), arcname=f)
    return zip_path

with gr.Blocks(title="Ge'ez Dataset Creator") as demo:
    master_state = gr.State()
    gr.Markdown("# 🇪🇹 Ge'ez Fidel Dataset Creator")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="1. Upload Scan", type="numpy")
            with gr.Group():
                focus_id = gr.Number(label="Current ID (Type or Click Image)", value=0, precision=0)
                zoom_view = gr.Image(label="Zoomed View", interactive=False)
            export_btn = gr.Button("Build & Download Dataset", variant="primary")
            download_link = gr.File(label="Output Zip")
            
        with gr.Column(scale=2):
            preview_img = gr.Image(label="Full Preview (CLICK A BOX TO SELECT IT)")
            
            # SINGLE ROW EDITOR
            current_row_table = gr.Dataframe(headers=COL_NAMES, datatype=["number"]*5 + ["str"], interactive=True, label="Edit Current Fidel")
            save_btn = gr.Button("💾 Save Edits & Refresh View", variant="secondary")

    gr.Markdown("### 📋 Full Dataset Table (Full View)")
    master_table_view = gr.Dataframe(headers=COL_NAMES, interactive=False)

    # Handlers
    input_img.upload(process_upload, inputs=input_img, outputs=[preview_img, zoom_view, current_row_table, focus_id, master_state, master_table_view])
    
    # Click image to focus
    preview_img.select(on_image_click, [input_img, master_state], [focus_id, preview_img, zoom_view, current_row_table])
    
    # Manual ID change
    focus_id.change(update_ui_and_filter, [input_img, master_state, focus_id], [preview_img, zoom_view, current_row_table])
    
    # Save Button
    save_btn.click(save_changes, [current_row_table, master_state, input_img, focus_id], [master_state, master_table_view, preview_img, zoom_view, current_row_table])
    
    export_btn.click(export_dataset, [input_img, master_state], download_link)

demo.launch()