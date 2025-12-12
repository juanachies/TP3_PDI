import cv2
import numpy as np

cap = cv2.VideoCapture('tirada_2.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.33, fy=0.33)

    # HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango del verde
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Máscara del verde
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Limpiar un poco la máscara
    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Contornos del verde
    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Fondo verde = contorno más grande
        c = max(contours, key=cv2.contourArea)

        # Crear máscara vacía
        mask_area = np.zeros(mask_green.shape, dtype=np.uint8)

        # Dibujar el área verde completa
        cv2.drawContours(mask_area, [c], -1, 255, thickness=cv2.FILLED)

        # Aplicar la máscara al frame ORIGINAL
        resultado = cv2.bitwise_and(frame, frame, mask=mask_area)

        gray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
        edges = cv2.Canny(blurred, 50, 100)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 8, cv2.CV_32S)

        for stat in stats:
            x, y, w, h, area = stat
            aspect_ratio = w / h
            if area < 800 and aspect_ratio > 0.7 and aspect_ratio < 1.3:
               cv2.rectangle(closed, (x, y), (x + w, y + h), (255, 255, 255), 2)

        cv2.imshow("Solo area verde (color)", closed)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()