import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pytesseract
import tempfile
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Corrige encoding no Streamlit Cloud
import locale
locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"

# Carrega o modelo
model = YOLO('plaquinhas.pt')

# Página com menu
st.title("Reconhecimento de Placas")
option = st.sidebar.selectbox("Escolha o modo", ["📷 Upload de Imagem", "🎥 Webcam em tempo real"])

# Função de OCR
def ocr_from_box(image_rgb, x1, y1, x2, y2):
    roi = image_rgb[y1:y2, x1:x2]
    text = pytesseract.image_to_string(roi, config='--psm 7')
    return text.strip()

# 📷 Modo Upload
if option == "📷 Upload de Imagem":
    uploaded_file = st.file_uploader("Envie uma imagem", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            image.save(temp_file.name)
            results = model.predict(temp_file.name, device='cpu')
        
        result = results[0]
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            text = ocr_from_box(img_array, x1, y1, x2, y2)
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_array, text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            st.code(f"Placa detectada: {text}")
        
        st.image(img_array, caption="Resultado", use_column_width=True)

# 🎥 Modo Webcam
elif option == "🎥 Webcam em tempo real":
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = model.predict(image_rgb, device='cpu')
            result = results[0]

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                text = ocr_from_box(image_rgb, x1, y1, x2, y2)
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_rgb, text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    webrtc_streamer(key="placa", video_processor_factory=VideoProcessor)
