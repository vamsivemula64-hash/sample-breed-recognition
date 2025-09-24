import streamlit as st
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import cv2
import numpy as np
import random

# ------------------ CLASS COLORS ------------------
CLASS_COLORS = {}

def get_class_color(cls_id):
    if cls_id not in CLASS_COLORS:
        CLASS_COLORS[cls_id] = [random.randint(0, 255) for _ in range(3)]
    return CLASS_COLORS[cls_id]

# ------------------ LOGIN ------------------
def login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "breed" and password == "25004":
            st.session_state.logged_in = True
            st.success("Login successful!")
        else:
            st.error("Incorrect username or password.")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ------------------ MODEL LOADING ------------------
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

st.title("🐄 Cattle & Buffalo Breed Recognition (Image + Video)")

# ------------------ MODEL SELECTION ------------------
model_file = st.file_uploader("Upload YOLO model (.pt)", type=["pt"])
if not model_file:
    st.warning("Please upload a YOLO model file (.pt) to continue.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
    tmp.write(model_file.read())
    model_path = tmp.name

model = load_model(model_path)
if model is None:
    st.stop()

# ------------------ IMAGE/VIDEO INPUT ------------------
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        st.subheader("📷 Image Detection")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=512)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
            image.save(tmp_img.name)
            results = model.predict(tmp_img.name)
            result = results[0]

        img = cv2.imread(tmp_img.name)

        for box in result.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = result.names[cls_id]
            color = get_class_color(cls_id)

            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            cv2.putText(
                img,
                f"{label} {conf:.2f}",
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Show prediction
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Prediction", width=512)

        # Download button
        result_img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as img_file:
            result_img_pil.save(img_file.name)
            with open(img_file.name, "rb") as file:
                st.download_button(
                    label="📥 Download Predicted Image",
                    data=file,
                    file_name="predicted_image.jpg",
                    mime="image/jpeg"
                )

    elif uploaded_file.type == "video/mp4":
        st.subheader("🎥 Video Detection")

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        progress = st.progress(0, text="Processing video...")

        processed = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)
            result = results[0]
            frame_out = frame.copy()

            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                label = result.names[cls_id]
                color = get_class_color(cls_id)

                cv2.rectangle(frame_out, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(
                    frame_out,
                    f"{label} {conf:.2f}",
                    (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            out.write(frame_out)
            processed += 1
            progress.progress(min(processed / frame_count, 1.0), text=f"Processing frame {processed}/{frame_count}")

        cap.release()
        out.release()

        st.success("✅ Video processing complete!")
        st.video(output_path)

        with open(output_path, "rb") as file:
            st.download_button(
                label="📥 Download Processed Video",
                data=file,
                file_name="predicted_video.mp4",
                mime="video/mp4"
            )
