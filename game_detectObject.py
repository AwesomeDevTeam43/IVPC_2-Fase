import cv2
from ultralytics import YOLO

# Configuração
model_path = "yolov8s.pt"  # Modelo YOLO pré-treinado
target_labels = ["cell phone", "bottle"]  # Objetos que queremos detetar
cap = cv2.VideoCapture(0)  # Abrir a câmera

# Verificar se a câmera abriu corretamente
if not cap.isOpened():
    raise Exception("Erro ao abrir a câmera!")

# Carregar o modelo YOLO
model = YOLO(model_path)

# Definir o tamanho mínimo e máximo para o filtro de bounding boxes
min_area = 1000  # Área mínima aceitável
max_area = 100000  # Área máxima aceitável


# Função para calcular o ponto central vertical
def get_center_y(bbox):
    _, y1, _, y2 = bbox  # Coordenadas do retângulo (x1, y1, x2, y2)
    return int((y1 + y2) / 2)  # Retorna o centro vertical


# Função `update` para deteção
def update():
    ret, frame = cap.read()  # Ler um frame da câmera
    if not ret:
        print("Erro ao capturar frame!")
        return None, None

    # Realizar a deteção
    results = model(frame)
    y_centers = {label: None for label in target_labels}  # Inicializar dicionário para as coordenadas y

    # Processar os resultados
    for result in results:
        boxes = result.boxes  # Pega todas as caixas detetadas
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas do retângulo
            label = result.names[int(box.cls[0])]  # Nome do objeto detetado

            # Verificar se o rótulo corresponde a um dos objetos alvo
            if label in target_labels:
                # Calcular a área do bounding box
                area = (x2 - x1) * (y2 - y1)
                if min_area <= area <= max_area:  # Aplicar o filtro de tamanho
                    center_y = get_center_y((x1, y1, x2, y2))  # Calcular o centro vertical
                    y_centers[label] = center_y  # Atualizar o dicionário com a coordenada `y`

                    # Desenhar no frame (opcional)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (int((x1 + x2) / 2), center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"{label} (y={center_y})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o frame (opcional)
    cv2.imshow("Detecao de Objetos", frame)

    # Retorna as coordenadas `y` dos objetos
    return y_centers[target_labels[0]], y_centers[target_labels[1]]


# Liberar recursos
def release_resources():
    cap.release()
    cv2.destroyAllWindows()
