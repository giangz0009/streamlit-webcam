import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 as cv
import numpy as np
import joblib

# Load model nhận diện
svc = joblib.load('svc.pkl')
mydict = ['BanKiet', 'BanNghia', 'BanNguyen', 'BanThanh', 'SangSang', 'ThayDuc']

# Load detector và recognizer
detector = cv.FaceDetectorYN.create(
    'face_detection_yunet_2023mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv.FaceRecognizerSF.create(
    'face_recognition_sface_2021dec.onnx', ""
)

# Cờ xác định đã set kích thước detector chưa
input_size_set = False

# Giao diện Streamlit
st.title("Nhận diện khuôn mặt bằng Streamlit WebRTC")

# Hàm callback xử lý frame (thay cho VideoTransformerBase)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    global input_size_set

    tm = cv.TickMeter()

    img = frame.to_ndarray(format="bgr24")

    if not input_size_set:
        h, w, _ = img.shape
        detector.setInputSize([w, h])
        input_size_set = True

    tm.start()
    faces = detector.detect(img)
    tm.stop()

    if faces[1] is not None:
        face_align = recognizer.alignCrop(img, faces[1][0])
        face_feature = recognizer.feature(face_align)
        prediction = svc.predict(face_feature)
        result = mydict[prediction[0]]
        cv.putText(img, result, (1, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), 2)
            for i in range(5):
                x, y = coords[4 + i*2], coords[5 + i*2]
                cv.circle(img, (x, y), 2, (255, 0, 0), 2)

    fps_text = 'FPS: {:.2f}'.format(tm.getFPS())
    cv.putText(img, fps_text, (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Tạo webrtc stream
webrtc_streamer(
    key="face-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration={"iceServers": [{
   "urls": [ "stun:ss-turn2.xirsys.com" ]
}, {
   "username": "842WZG-XeDKmIK4bVyiFPMrMkL62rRByOy1apVF1I5dgC0Hn2H9gmH88j_eR25srAAAAAGgHhxxnaWFuZ3owMDA5",
   "credential": "b952f87c-1f72-11f0-aaa6-0242ac140004",
   "urls": [
       "turn:ss-turn2.xirsys.com:80?transport=udp",
       "turn:ss-turn2.xirsys.com:3478?transport=udp",
       "turn:ss-turn2.xirsys.com:80?transport=tcp",
       "turn:ss-turn2.xirsys.com:3478?transport=tcp",
       "turns:ss-turn2.xirsys.com:443?transport=tcp",
       "turns:ss-turn2.xirsys.com:5349?transport=tcp"
   ]
}]}
)
