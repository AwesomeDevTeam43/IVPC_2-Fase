import cv2
from ultralytics import YOLO

# Configuração
model_path = "yolov8n.pt"  # Use um modelo YOLO pré-treinado
target_label = "cell phone"  # Nome do objeto que queres detetar
cap = cv2.VideoCapture(0)  # Abrir a câmera

# Verificar se a câmera abriu corretamente
if not cap.isOpened():
    raise Exception("Erro ao abrir a câmera!")

# Carregar o modelo YOLO
model = YOLO(model_path)


# Função para calcular o ponto central do retângulo de deteção
def get_center_y(bbox):
    x1, y1, x2, y2 = bbox  # Coordenadas do retângulo (x1, y1, x2, y2)
    return int((y1 + y2) / 2)  # Retorna o centro vertical


# Função `update` para detecção
def update():
    ret, frame = cap.read()  # Ler um frame da câmera
    if not ret:
        print("Erro ao capturar frame!")
        return None

    # Realizar a detecção
    results = model(frame)

    # Processar os resultados
    for result in results:
        boxes = result.boxes  # Pega todas as caixas detetadas
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do retângulo
            label = result.names[int(box.cls[0])]  # Nome do objeto detetado

            # Verificar se o rótulo corresponde ao objeto alvo
            if label == target_label:
                center_y = get_center_y((x1, y1, x2, y2))  # Calcular o centro vertical

                # Desenhar no frame (opcional)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int((x1 + x2) / 2), center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{label} (y={center_y})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Exibir o frame (opcional)
                cv2.imshow("Detecao de Objetos", frame)

                return center_y  # Retorna o centro vertical do objeto alvo

    # Caso nenhum objeto alvo seja encontrado
    cv2.imshow("Detecao de Objetos", frame)  # Exibir o frame sem deteção (opcional)
    return None


# Liberar recursos
def release_resources():
    cap.release()
    cv2.destroyAllWindows()
