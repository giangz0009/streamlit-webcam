import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2 as cv
import numpy as np
import joblib

# Load các model nhận diện
svc = joblib.load('svc.pkl')
mydict = ['BanKiet', 'BanNghia', 'BanNguyen', 'BanThanh', 'SangSang', 'ThayDuc']

# Load detector và recognizer của OpenCV
detector = cv.FaceDetectorYN.create(
    'face_detection_yunet_2023mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

recognizer = cv.FaceRecognizerSF.create(
    'face_recognition_sface_2021dec.onnx',
    ""
)

# Giao diện Streamlit
st.title("Nhận diện khuôn mặt bằng Streamlit WebRTC")

class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.tm = cv.TickMeter()
        self.detector = detector
        self.recognizer = recognizer
        self.svc = svc
        self.labels = mydict
        self.input_size_set = False

    def visualize(self, input, faces, fps, thickness=2):
        if faces[1] is not None:
            for idx, face in enumerate(faces[1]):
                coords = face[:-1].astype(np.int32)
                cv.rectangle(input, (coords[0], coords[1]),
                             (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
                for i in range(5):
                    x, y = coords[4 + i*2], coords[5 + i*2]
                    cv.circle(input, (x, y), 2, (255, 0, 0), thickness)
        cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")

        if not self.input_size_set:
            h, w, _ = img.shape
            self.detector.setInputSize([w, h])
            self.input_size_set = True

        self.tm.start()
        faces = self.detector.detect(img)
        self.tm.stop()

        if faces[1] is not None:
            face_align = self.recognizer.alignCrop(img, faces[1][0])
            face_feature = self.recognizer.feature(face_align)
            prediction = self.svc.predict(face_feature)
            result = self.labels[prediction[0]]
            cv.putText(img, result, (1, 50), cv.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)

        self.visualize(img, faces, self.tm.getFPS())
        return img

webrtc_streamer(key="face-detection", video_transformer_factory=FaceRecognitionTransformer,rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
