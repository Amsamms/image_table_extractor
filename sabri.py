import streamlit as st
from PIL import Image
import pandas as pd
from transformers import TableTransformerForObjectDetection, AutoImageProcessor
import requests

# Define utility functions
def rescale_bboxes(out_bbox, size):
    width, height = size
    b = [(x * width, y * height) for x, y in out_bbox]
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label != 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})
    return objects

# Load models
detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")

st.title("Table Detection and OCR with Transformers")

# Upload image
uploaded_file = st.file_uploader("Upload an image containing a table", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect tables
    encoding = image_processor(images=image, return_tensors="pt")
    outputs = detection_model(**encoding)
    detected_tables = outputs_to_objects(outputs, image.size, detection_model.config.id2label)

    st.write("Detected tables:")
    st.write(detected_tables)

    if detected_tables:
        # Process detected tables to recognize structure
        table = detected_tables[0]  # For simplicity, use the first detected table
        bbox = table['bbox']
        cropped_image = image.crop(bbox)
        st.image(cropped_image, caption='Cropped Table', use_column_width=True)

        encoding = image_processor(images=cropped_image, return_tensors="pt")
        outputs = structure_model(**encoding)
        cells = outputs_to_objects(outputs, cropped_image.size, structure_model.config.id2label)

        st.write("Detected cells:")
        st.write(cells)

        # Extract cell contents and convert to CSV
        data = []
        for cell in cells:
            # Dummy OCR: replace this with actual OCR call
            data.append({"bbox": cell['bbox'], "content": "text"})

        df = pd.DataFrame(data)
        st.write("Extracted Table Data:")
        st.dataframe(df)

        # Save results as CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download CSV", data=csv, file_name="table.csv", mime="text/csv")
