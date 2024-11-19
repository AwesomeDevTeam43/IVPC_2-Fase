import cv2
from ultralytics import YOLO

# Configuração
model_path = "yolov8s.pt"  # Modelo YOLO pré-treinado
target_labels = ["cell phone", "bottle"]  # Objetos que queremos detetar

use_cam = True

# Abrir a câmera
if use_cam:
    cap = cv2.VideoCapture(0)
else:
    print("hello world")

# Verificar se a câmera abriu corretamente

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
    if not cap.isOpened():
        cap.open(0)
    _,image = cap.read()

    # Realizar a deteção
    results = model(image, verbose=False)

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
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(image, (int((x1 + x2) / 2), center_y), 5, (0, 0, 255), -1)
                    cv2.putText(image, f"{label} (y={center_y})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o frame (opcional)
    cv2.imshow("Detecao de Objetos", image)

    # Retorna as coordenadas `y` dos objetos
    return y_centers[target_labels[0]], y_centers[target_labels[1]]


# Liberar recursos
def release_resources():
    cap.release()
    cv2.destroyAllWindows()
