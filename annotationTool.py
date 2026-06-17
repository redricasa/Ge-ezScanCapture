import gradio as gr
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import shutil
import zipfile
import tempfile

def detect_fidels(img):
    """Initial segmentation using OpenCV contours."""
    if img is None:
        return []
    
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.find_centers, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    contours, _ = cv2.find_contours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out tiny noise (dots)
        if w > 5 and h > 5:
            boxes.append([x, y, w, h, "unlabeled"])
            
    # Sort boxes top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))
    return boxes

def update_preview(img, dataframe):
    """Draws bounding boxes on the image based on the dataframe values."""
    if img is None or dataframe is None:
        return None
    
    canvas = img.copy()
    for i, row in dataframe.iterrows():
        try:
            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
            label = str(row['label'])
            # Draw box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw ID/Label
            cv2.putText(canvas, f"{i}:{label}", (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        except:
            continue
    return canvas

def process_upload(img):
    """Triggered when image is uploaded."""
    boxes = detect_fidels(img)
    df = pd.DataFrame(boxes, columns=['x', 'y', 'w', 'h', 'label'])
    preview = update_preview(img, df)
    return df, preview

def export_dataset(img, dataframe):
    """Crops, resizes to 32x32, and zips the results."""
    if img is None or dataframe.empty:
        return None

    temp_dir = tempfile.mkdtemp()
    img_dir = os.path.join(temp_dir, "fidels")
    os.makedirs(img_dir)
    
    metadata = []

    for i, row in dataframe.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        label = str(row['label'])
        
        # Crop
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue
        
        # Convert to PIL for better resizing with padding
        pil_img = Image.fromarray(crop)
        
        # Maintain aspect ratio and pad to 32x32
        pil_img.thumbnail((32, 32), Image.Resampling.LANCZOS)
        new_img = Image.new("L", (32, 32), (255)) # White background
        new_img.paste(pil_img, ((32 - pil_img.size[0]) // 2, (32 - pil_img.size[1]) // 2))
        
        # Save image
        fname = f"fidel_{i}.png"
        new_img.save(os.path.join(img_dir, fname))
        metadata.append({"file": fname, "label": label, "x": x, "y": y, "w": w, "h": h})

    # Save CSV
    pd.DataFrame(metadata).to_csv(os.path.join(temp_dir, "metadata.csv"), index=False)
    
    # Zip everything
    zip_path = os.path.join(tempfile.gettempdir(), "geez_dataset.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(temp_dir):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)
                
    return zip_path

# --- Gradio UI ---

with gr.Blocks(title="Ge'ez Fidel Segmenter") as demo:
    gr.Markdown("# 🇪🇹 Ge'ez Fidel Dataset Creator")
    gr.Markdown("1. Upload handwriting scan. 2. Adjust boxes in the table. 3. Label characters. 4. Download.")
    
    with gr.Row():
        input_img = gr.Image(label="Upload Scanned Image", type="numpy")
        preview_img = gr.Image(label="Preview Segments")
    
    with gr.Row():
        with gr.Column(scale=1):
            refresh_btn = gr.Button("Update Preview", variant="secondary")
            export_btn = gr.Button("Build & Download Dataset", variant="primary")
            download_file = gr.File(label="Download Zip")
            
        with gr.Column(scale=2):
            box_data = gr.Dataframe(
                headers=["x", "y", "w", "h", "label"],
                datatype=["number", "number", "number", "number", "str"],
                col_count=(5, "fixed"),
                interactive=True,
                label="Edit Bounding Boxes and Labels"
            )

    # Logic
    input_img.upload(process_upload, inputs=[input_img], outputs=[box_data, preview_img])
    refresh_btn.click(update_preview, inputs=[input_img, box_data], outputs=[preview_img])
    export_btn.click(export_dataset, inputs=[input_img, box_data], outputs=[download_file])

demo.launch()
