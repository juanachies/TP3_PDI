import cv2
import numpy as np
import math

cap = cv2.VideoCapture('tirada_4.mp4')

centroides_previos = []
frames_quietos = 0
FRAMES_REPOSO = 8
UMBRAL_MOVIMIENTO = 2.5

# Historial y valores finales
historial_valores = {}
valores_finales = {}   # <-- ACA está la clave

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.33, fy=0.33)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((7, 7), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)

        mask_area = np.zeros(mask_green.shape, dtype=np.uint8)
        cv2.drawContours(mask_area, [c], -1, 255, thickness=cv2.FILLED)

        resultado = cv2.bitwise_and(frame, frame, mask=mask_area)

        gray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        th = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21, 5
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, 8)

        centroides_actuales = []
        dados_detectados = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            aspect_ratio = w / h

            if 300 < area < 3000 and 0.6 < aspect_ratio < 1.2:
                cx, cy = centroids[i]
                centroides_actuales.append((cx, cy))

                roi = gray[y:y+h, x:x+w]
                roi = cv2.GaussianBlur(roi, (5, 5), 0)

                circles = cv2.HoughCircles(
                    roi,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=10,
                    param1=80,
                    param2=6,
                    minRadius=6,
                    maxRadius=10
                )

                num_puntos = 0
                if circles is not None:
                    num_puntos = len(circles[0])

                dados_detectados.append((x, y, w, h, num_puntos))

        # ---------------------------
        # Detección de reposo
        # ---------------------------
        en_reposo = False

        if centroides_previos and len(centroides_actuales) == len(centroides_previos):
            desplazamientos = [
                math.dist(centroides_actuales[i], centroides_previos[i])
                for i in range(len(centroides_actuales))
            ]

            if np.mean(desplazamientos) < UMBRAL_MOVIMIENTO:
                frames_quietos += 1
            else:
                frames_quietos = 0
                historial_valores.clear()
                valores_finales.clear()

            if frames_quietos >= FRAMES_REPOSO:
                en_reposo = True

        centroides_previos = centroides_actuales

        # ---------------------------
        # ACUMULAR mientras están quietos
        # ---------------------------
        if en_reposo:
            for i, (_, _, _, _, valor) in enumerate(dados_detectados):
                if i not in historial_valores:
                    historial_valores[i] = []
                historial_valores[i].append(valor)

            # Calcular UNA SOLA VEZ el valor final
            if not valores_finales:
                print("Dados detenidos:")
                for i in historial_valores:
                    valor_final = max(
                        set(historial_valores[i]),
                        key=historial_valores[i].count
                    )
                    valores_finales[i] = valor_final
                    print(f"Dado {i+1}: {valor_final}")

        # ---------------------------
        # DIBUJAR SOLO EL VALOR FINAL
        # ---------------------------
        for i, (x, y, w, h, _) in enumerate(dados_detectados):
            if i in valores_finales:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{valores_finales[i]}",
                    (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Resultado", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
