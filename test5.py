import cv2
import numpy as np

cap = cv2.VideoCapture('tirada_1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.33, fy=0.33)

    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango del verde
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Máscara del verde
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Limpiar un poco la máscara
    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Encontrar los contornos
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

        # Convertir a escala de grises y difuminar
        gray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Umbralización adaptativa
        th = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            5
        )

        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(th, kernel, iterations=1)
        th = cv2.dilate(eroded, kernel, iterations=1)

        # Componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8, cv2.CV_32S)

        conteo_dado = []  # Cambié el diccionario por una lista para contar los dados
        for stat in stats[1:]:
            x, y, w, h, area = stat
            aspect_ratio = w / h

            # DADO completo
            if 2000 < area < 15000 and 0.8 < aspect_ratio < 1.2:
                roi = gray[y:y+h, x:x+w]
                roi_blur = cv2.GaussianBlur(roi, (9, 9), 1.5)

                circles = cv2.HoughCircles(
                    roi_blur,
                    cv2.HOUGH_GRADIENT,
                    dp=1.2,
                    minDist=20,
                    param1=100,
                    param2=15,
                    minRadius=6,
                    maxRadius=15
                )

                num_puntos = 0
                if circles is not None:
                    num_puntos = len(circles[0])

                cv2.rectangle(gray, (x, y), (x+w, y+h), (255,255,255), 2)
                cv2.putText(gray, str(num_puntos), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                # DEBUG
                cv2.imshow("ROI dado", roi)

                # Mostrar la imagen resultante
                cv2.imshow("Solo area verde (color)", gray)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
