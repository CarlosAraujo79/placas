import streamlit as st
import cv2
from PIL import Image
import numpy as np
from ultralytics import YOLO
import pytesseract
import tempfile
import locale

# Corrige erro de encoding no Streamlit Cloud
def getpreferredencoding(do_setlocale=True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding

# Opcional: Defina o caminho do Tesseract em sistemas locais Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Carrega o modelo
model = YOLO('plaquinhas.pt')

st.title("Reconhecimento de Placas com YOLOv8 + OCR")
uploaded_file = st.file_uploader("Envie uma imagem de um carro", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converte imagem para array (RGB)
    image = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image)

    # Salva imagem temporária para predição com YOLO
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model.predict(temp_file.name, device='cpu')
    
    result = results[0]
    image_ocr = img_array.copy()
    placas_detectadas = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Recorta e aplica OCR na ROI
        roi = image_ocr[y1:y2, x1:x2]
        text = pytesseract.image_to_string(roi, config='--psm 7')
        text = text.strip()
        placas_detectadas.append(text)

        # Desenha a box + texto na imagem
        cv2.rectangle(image_ocr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_ocr, text, (x1, y2 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    st.image(image_ocr, caption="Resultado com OCR", use_column_width=True)

    if placas_detectadas:
        st.subheader("Placas detectadas:")
        for placa in placas_detectadas:
            st.code(placa)
    else:
        st.warning("Nenhuma placa detectada.")
