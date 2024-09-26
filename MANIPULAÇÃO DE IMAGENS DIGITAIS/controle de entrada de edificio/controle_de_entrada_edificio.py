import cv2
import numpy as np
from imutils.video import VideoStream
import imutils

# Função para detectar pessoas
def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if classID == personIdx and confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    return results

# Parâmetros
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

total_in = 0
total_out = 0
people_inside = 0

# Captura de vídeo das duas câmeras IP
camera_in = VideoStream("rtsp://user:password@ip_camera_entrada").start()
camera_out = VideoStream("rtsp://user:password@ip_camera_saida").start()

while True:
    # Ler frames de entrada e saída
    frame_in = camera_in.read()
    frame_out = camera_out.read()

    frame_in = imutils.resize(frame_in, width=700)
    frame_out = imutils.resize(frame_out, width=700)

    results_in = detect_people(frame_in, net, ln)
    results_out = detect_people(frame_out, net, ln)

    # Contagem de pessoas entrando
    if len(results_in) > 0:
        total_in += len(results_in)
    
    # Contagem de pessoas saindo
    if len(results_out) > 0:
        total_out += len(results_out)
    
    # Calcular o número total de pessoas dentro do edifício
    people_inside = total_in - total_out

    # Exibir resultados
    cv2.putText(frame_in, f"Total Inside: {people_inside}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Entrada", frame_in)
    cv2.imshow("Saida", frame_out)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera_in.stop()
camera_out.stop()
cv2.destroyAllWindows()