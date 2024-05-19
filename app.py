import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

# Load the YOLO model
model = YOLO('saral4.pt')

# Set up the Streamlit app
st.title("Image Segmentation using Custom Trained YOLOv8")
st.write("Disclaimer - This segmentation might show incorrect results. Models are only as good as the data")
st.write("Please try to upload high quality images.")
# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if st.button('Predict'):
        # Make predictions using the model
        if uploaded_file is not None:
    # Convert the uploaded file to an image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Image', use_column_width=True)
            results = model.predict(image, show=False, conf=0.6)

        # Retrieve the image with bounding boxes
            annotated_image = results[0].plot()  # Assume results[0] contains the desired output
        
        # Convert annotated image back to RGB format
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        
        # Display the annotated image
            st.image(annotated_image_rgb, caption='Segmentation', use_column_width=True)
        else:
              st.error('Please upload an image')    
