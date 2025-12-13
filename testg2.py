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
valores_finales = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.33, fy=0.33)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ---------------------------
    # Máscara verde (fondo)
    # ---------------------------
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

            # filtro de dado
            if 300 < area < 3000 and 0.6 < aspect_ratio < 1.2:
                cx, cy = centroids[i]
                centroides_actuales.append((cx, cy))

                # ---------------------------
                # ROI DEL DADO
                # ---------------------------
                roi = frame[y:y+h, x:x+w]
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # máscara de puntos blancos
                lower_white = np.array([0, 0, 180])
                upper_white = np.array([180, 60, 255])
                mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)

                # contar puntos
                num_puntos = 0
                n_labels, _, stats_w, _ = cv2.connectedComponentsWithStats(mask_white, 8)

                for stat in stats_w[1:]:
                    area_p = stat[cv2.CC_STAT_AREA]
                    if 30 < area_p < 300:
                        num_puntos += 1

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
        # ACUMULAR valores en reposo
        # ---------------------------
        if en_reposo:
            for i, (_, _, _, _, valor) in enumerate(dados_detectados):
                historial_valores.setdefault(i, []).append(valor)

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
        # DIBUJAR SOLO VALOR FINAL
        # ---------------------------
        for i, (x, y, w, h, _) in enumerate(dados_detectados):
            if i in valores_finales:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    str(valores_finales[i]),
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
