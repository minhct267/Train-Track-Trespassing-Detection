# Python In-built packages
from pathlib import Path
import PIL
import numpy as np

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Rail Track and Human Detection",
    page_icon='images/logo.png',
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Rail Track and Human Detection using YOLOv8")
status = st.empty()

# Sidebar
st.sidebar.header("Image/Video Configuration")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

# Get the model path
model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose Image", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                default_image = default_image.resize((300, 300))
                st.image(default_image, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            default_detected_image = default_detected_image.resize((300, 300))
            st.image(default_detected_image, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=0.3, show_boxes=False)
                masks = res[0].masks
                boxes = res[0].boxes

                res_plotted = res[0].plot()[:, :, ::-1]
                #res_plotted = helper.mask_generate(masks.data.numpy(), np.array(uploaded_image))

                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                
                alert = helper.alert_level(boxes, masks)
                if alert == 0:
                    status.success("No people in the image.")
                elif alert == 1:
                    status.warning("People in the image, but do not step on the rail.")
                elif alert == 2:
                    status.error("Alert! People are stepping on the rail.")

                try:
                    with st.expander("Detection Results"):
                        for mask in masks:
                            st.write(mask.data)
                        
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    st.warning("The video function is not developed yet.")

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(model)

else:
    st.error("Please select a valid source type!")
