import cv2
import os
import time
import numpy as np
from ultralytics import YOLO

# Configuração
use_cam = True
model_path = "yolov8n.pt"  # Use um modelo YOLO pré-treinado, ex: yolov8n.pt
target_label = "cell phone"  # Nome do objeto que queres detetar (ajusta conforme o modelo)


# Inicializar captura de vídeo
cap = cv2.VideoCapture(0 if use_cam else 'video.mp4')

# Verificar se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao abrir a camera ou vídeo!")
    exit()

# Carregar o modelo YOLO
model = YOLO(model_path)


# Função para calcular o ponto central do retângulo de deteção
def get_center_y(bbox):
    x1, y1, x2, y2 = bbox  # Coordenadas do retângulo (x1, y1, x2, y2)
    center_x = int((x1 + x2) / 2)  # Ponto central horizontal
    center_y = int((y1 + y2) / 2)  # Ponto central vertical
    return center_y


# Loop para capturar frames
while True:
    ret, frame = cap.read()
    if not ret:
        print("Fim do vídeo ou erro na leitura!")
        break

    # Realizar a deteção
    results = model(frame)

    # Processar resultados
    for result in results:
        boxes = result.boxes  # Pega todas as caixas detetadas
        for box in boxes:
            # Obter as coordenadas do retângulo e o rótulo do objeto
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas da bounding box
            label = result.names[int(box.cls[0])]  # Nome do objeto detetado

            # Verificar se o rótulo corresponde ao objeto alvo
            if label == target_label:
                center_y = get_center_y((x1, y1, x2, y2))  # Calcular o centro vertical
                print(f"Centro vertical (y) do objeto '{target_label}': {center_y}")

                # Desenhar o retângulo e o ponto central no frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int((x1 + x2) / 2), center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{label} (y={center_y})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                            2)

    # Exibir o frame anotado
    cv2.imshow("Detecao de Objetos", frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
