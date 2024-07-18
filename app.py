import streamlit as st
import cv2
import numpy as np
from rockvolDetect import load_model, detect_rocks, visualize_detection, calculate_rock_volume
from detectron2.data import MetadataCatalog

# Cargar el modelo
@st.cache_resource
def load_cached_model():
    config_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights_path = "/content/drive/MyDrive/ENTRENADOS/model_final.pth"
    return load_model(config_path, weights_path)

predictor = load_cached_model()

# Configurar la página de Streamlit
st.set_page_config(page_title="Detección de Rocas y Cálculo de Volumen", layout="wide")
st.title("Detección de Rocas y Cálculo de Volumen")

# Subida de imagen
uploaded_file = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir la imagen subida a un array de numpy
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Detectar rocas
    outputs = detect_rocks(predictor, image)
    
    # Visualizar la detección
    metadata = MetadataCatalog.get("my_dataset_train")
    visualized_image = visualize_detection(image, outputs, metadata)
    
    # Calcular el volumen de roca
    porcentaje_roca, categoria_roca = calculate_rock_volume(outputs, metadata, image.shape)
    
    # Mostrar resultados
    col1, col2 = st.columns(2)
    with col1:
        st.image(visualized_image, caption="Detección de Rocas", use_column_width=True)
    with col2:
        st.write(f"Porcentaje de volumen de roca: {porcentaje_roca:.2f}%")
        st.write(f"Categoría de roca: {categoria_roca}")

st.write("Nota: Asegúrate de que el modelo esté entrenado y las rutas de los archivos sean correctas.")
