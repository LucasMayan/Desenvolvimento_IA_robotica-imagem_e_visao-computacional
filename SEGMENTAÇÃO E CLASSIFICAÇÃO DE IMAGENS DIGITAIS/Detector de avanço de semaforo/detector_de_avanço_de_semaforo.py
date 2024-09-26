import cv2
import numpy as np

# Defina a região de interesse (ROI) onde a faixa de semáforo está localizada
roi_top_left = (100, 200)
roi_bottom_right = (500, 250)

# Inicializa a câmera
cap = cv2.VideoCapture(0)

# Define a cor da faixa (ex: branca)
lower_bound = np.array([0, 0, 200])
upper_bound = np.array([180, 25, 255])

# Função para detectar avanço de faixa
def detect_line_crossing(frame):
    # Converte para escala de cinza e depois para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Aplicar a máscara na imagem
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Extrair a região de interesse
    roi = mask[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
    
    # Detecta contornos na região de interesse
    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Verifica se há algum contorno, ou seja, se há algo cruzando a faixa
    if contours:
        return True
    return False

while True:
    # Captura frame da câmera
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta se houve avanço de faixa
    if detect_line_crossing(frame):
        print("Avanço de faixa detectado!")

    # Desenha a região de interesse na tela
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), 2)
    
    # Exibe o frame
    cv2.imshow("Frame", frame)

    # Para sair, pressione 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()