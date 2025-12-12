import cv2
import numpy as np

cap = cv2.VideoCapture('tirada_2.mp4')

# Rangos HSV
lower_green = np.array([35, 40, 40])
upper_green = np.array([85, 255, 255])

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.33, fy=0.33)

    # ------------------ AREA VERDE ------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        continue

    c = max(contours, key=cv2.contourArea)
    mask_area = np.zeros(mask_green.shape, dtype=np.uint8)
    cv2.drawContours(mask_area, [c], -1, 255, cv2.FILLED)

    area_valida = cv2.bitwise_and(frame, frame, mask=mask_area)

    # ------------------ DADOS ROJOS ------------------
    hsv_area = cv2.cvtColor(area_valida, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv_area, lower_red1, upper_red1),
        cv2.inRange(hsv_area, lower_red2, upper_red2)
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel, iterations=2)

    # ------------------ COMPONENTES CONEXOS ------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask_red, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        if area > 200:  # filtrar ruido
            cv2.rectangle(area_valida, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Resultado", area_valida)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
