import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

st.set_page_config(page_title="Nhận dạng đối tượng - YOLOv8", layout="centered")
st.title("🎯 Nhận dạng đối tượng bằng YOLOv8")

# Hiển thị ảnh hoặc khung video
FRAME_WINDOW = st.image([])

# Chọn mô hình
model_type = st.selectbox("Chọn mô hình YOLO", ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'])
model = YOLO(model_type)

# Chọn nguồn dữ liệu
source_type = st.radio("Nguồn dữ liệu", ['Webcam', 'Tải ảnh'])

if source_type == 'Webcam':
    start_button = st.button("📷 Bắt đầu webcam")
    stop_button = st.button("⛔ Dừng webcam")

    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    if start_button:
        st.session_state.camera_running = True

    if stop_button:
        st.session_state.camera_running = False

    if st.session_state.camera_running:
        cap = cv2.VideoCapture(1)

        while st.session_state.camera_running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Không lấy được hình ảnh từ camera!")
                break

            results = model(frame, verbose=False)[0]
            annotated_frame = results.plot()

            FRAME_WINDOW.image(annotated_frame, channels="BGR")

            # Ngắt nếu người dùng nhấn nút Dừng
            if stop_button:
                break

        cap.release()

else:  # Người dùng chọn tải ảnh
    uploaded_file = st.file_uploader("📁 Tải ảnh lên", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Đọc và hiển thị ảnh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Nhận dạng
        results = model(image, verbose=False)[0]
        annotated_image = results.plot()

        st.image(annotated_image, channels="BGR", caption="Kết quả nhận dạng")
