import PIL
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
from pytube import YouTube

import helper

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
        alert_level (int): The alert level based on the detected objects.
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf, show_boxes=False)
    masks = res[0].masks
    boxes = res[0].boxes

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    
    return helper.alert_level(boxes, masks)


def play_youtube_video(model, conf=0.3):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video URL")
    status = st.empty()

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()

            while (vid_cap.isOpened()):
                success, image = vid_cap.read()

                if success:
                    alert = _display_detected_frames(conf, model, st_frame, image)
                    status.empty()
                    if alert == 0:
                        status.success("No people in the image.")
                    elif alert == 1:
                        status.warning("People in the image, but do not step on the rail.")
                    elif alert == 2:
                        status.error("Alert! People are stepping on the rail.")
                else:
                    vid_cap.release()
                    break
                    
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    
def alert_level(boxes, masks):
    if boxes is None or masks is None:
        return 0
    
    box_class = boxes.cls.tolist()
    box_list = boxes.xyxy.tolist()
    mask_list = masks.xy
    person_box = []
    rail_mask = []
    if len(box_class) == 0:
        return 0
    for i, object_class in enumerate(box_class):
        if object_class == 0:
            person_box.append(box_list[i])
        if object_class == 1:
            rail_mask.append(mask_list[i])
    if len(person_box) == 0:
        return 0
    for i, box in enumerate(person_box):
        left_upper = (int(box[0]),int(box[1]))
        right_bottom = (int(box[2]),int(box[3]))
        right_upper = (int(box[2]),int(box[1]))
        left_bottom = (int(box[0]),int(box[3]))
        for mask in rail_mask:
            for point in mask:
                x,y = point
                if abs(int(y) - left_upper[1]) < 5:
                    if abs(int(x) - left_upper[0]) < 5:
                        return 2
                if abs(int(y) - right_bottom[1]) < 5:
                    if abs(int(x) - right_bottom[0]) < 5:
                        return 2
                if abs(int(y) - right_upper[1]) < 5:
                    if abs(int(x) - right_upper[0]) < 5:
                        return 2
                if abs(int(y) - left_bottom[1]) < 5:
                    if abs(int(x) - left_bottom[0]) < 5:
                        return 2
    return 1

def mask_generate(masks, image):
    # Apply each mask to the image
    for mask in masks:
        if mask.ndim == 2:
            mask = np.stack([mask] * 3, axis=-1)
        elif mask.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected number of dimensions in mask: {mask.ndim}")

        # Scale mask values to 255 (assuming mask is normalized to 1)
        mask = (mask * 255).astype(np.uint8)

        # Apply the mask to the image
        image = cv2.addWeighted(image, 1, mask, 0.5, 0)
        
    # Convert the image array back to an image
    return PIL.Image.fromarray(image)
