import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# Configuração
use_cam = True
model_path = "yolov8n.pt"  # Use um modelo YOLO pré-treinado, ex: yolov8n.pt

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0 if use_cam else 'video.mp4')

# Verificar se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir a camera ou vídeo!")
    exit()

# Carregar o modelo YOLO
model = YOLO(model_path)

# Loop para capturar frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura!")
        break

    # Realizar a detecção
    results = model(frame)

    # Processar resultados
    annotated_frame = results[0].plot()  # Adiciona as bounding boxes no frame

    # Exibir o frame
    cv2.imshow("Detecao de Objetos", annotated_frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
