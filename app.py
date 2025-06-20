import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile

# Carrega o modelo treinado
model = YOLO('plaquinhas.pt')

st.title("Detector de Placas com YOLOv8")

uploaded_file = st.file_uploader("Envie uma imagem de um carro", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lê a imagem enviada
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    # Salva temporariamente para passar para o modelo
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model.predict(temp_file.name, device='cpu')

    result = results[0]

    # Desenhar bounding boxes
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Desenhar retângulo e rótulo
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f'{label}: {conf*100:.1f}%', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    st.image(img_array, caption='Resultado da Detecção', use_column_width=True)
