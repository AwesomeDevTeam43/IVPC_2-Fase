import cv2
from ultralytics import YOLO
import time


model_path = "yolov8s.pt"
# Objetos selecionados
target_labels = ["cell phone", "bottle"]

use_cam = True

# Abrir a câmera
if use_cam:
    cap = cv2.VideoCapture(0)
else:
    print("hello world")

model = YOLO(model_path)

# Definir o tamanho mínimo e máximo para o filtro de area das bounding boxes
min_area = 1000
max_area = 100000


# Função para calcular o ponto central vertical
def get_center_y(bbox):
    _, y1, _, y2 = bbox
    return int((y1 + y2) / 2)


# Variáveis para calcular os FPS
fps = 0
frame_counter = 0
start_time = time.time()


# Função `update` para a deteção de objetos
def update():
    global fps, frame_counter, start_time

    if not cap.isOpened():
        cap.open(0)
    _, image = cap.read()

    # Contador de frames
    frame_counter += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1:
        fps = frame_counter
        frame_counter = 0
        start_time = time.time()

    # Realizar a deteção
    results = model(image, verbose=False)

    y_centers = {label: None for label in target_labels}

    # Processar os resultados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]

            # Verificar se o rótulo corresponde a um dos objetos alvo
            if label in target_labels:
                # Calcular a área do bounding box
                area = (x2 - x1) * (y2 - y1)
                if min_area <= area <= max_area:
                    center_y = get_center_y((x1, y1, x2, y2))
                    y_centers[label] = center_y

                    # Desenhar no frame
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(image, (int((x1 + x2) / 2), center_y), 5, (0, 0, 255), -1)
                    cv2.putText(image, f"{label} (y={center_y})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Exibir o FPS no canto superior esquerdo
    cv2.putText(image, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Exibir o frame
    cv2.imshow("Detecao de Objetos", image)

    # Retorna as coordenadas `y` dos objetos
    return y_centers[target_labels[0]], y_centers[target_labels[1]]


# libertar recursos
def release_resources():
    cap.release()
    cv2.destroyAllWindows()


# Loop principal
if __name__ == "__main__":
    try:
        while True:
            update()
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        release_resources()
