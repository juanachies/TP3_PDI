import cv2
import numpy as np
import math

# Definimos variables para matener la misma visualización
COLOR_BOX = (0, 255, 0)        
COLOR_TEXT = (0, 255, 0)       
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
BOX_THICKNESS = 2

def crear_mascara_verde(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    return mask


def obtener_area_mesa(mask_green, kernel_size=(5, 5), erode_iter=2):
    kernel = np.ones(kernel_size, np.uint8)
    mask_clean = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    c_mesa = max(contours, key=cv2.contourArea)
    mask_mesa = np.zeros(mask_green.shape, dtype=np.uint8)
    cv2.drawContours(mask_mesa, [c_mesa], -1, 255, thickness=cv2.FILLED)
    mask_mesa = cv2.erode(mask_mesa, kernel, iterations=erode_iter)
    
    return mask_mesa

def dibujar_resultado(frame, dados_info):
    for dado in dados_info:
        x, y, w, h = dado["box"]
        puntos = dado["puntos"]
        num_dado = dado["id"]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_BOX, BOX_THICKNESS)
        cv2.putText(frame, f'Dado {num_dado}: {puntos}', (x+5, y-5), FONT, FONT_SCALE, COLOR_TEXT, FONT_THICKNESS)


def verificar_reposo(centroides_actuales, centroides_previos, frames_quietos, umbral_movimiento, frames_reposo_req):
    if not centroides_previos:
        return False, 0
    
    if len(centroides_actuales) != len(centroides_previos):
        return False, 0
    
    if len(centroides_actuales) < 1:
        return False, 0
    
    desplazamientos = [math.dist(centroides_actuales[i], centroides_previos[i]) for i in range(len(centroides_actuales))]
    
    if np.mean(desplazamientos) < umbral_movimiento:
        frames_quietos += 1
    else:
        frames_quietos = 0
    
    en_reposo = frames_quietos >= frames_reposo_req
    return en_reposo, frames_quietos


def detectar_dados(nombre_video):
    dados = {}
    cap = cv2.VideoCapture(nombre_video)

    centroides_previos = []
    frames_quietos = 0
    FRAMES_REPOSO = 15
    UMBRAL_MOVIMIENTO = 1.5
    SCALE_FACTOR = 0.33
    
    analisis_realizado = False
    dados_dibujo = []

    while True:
        ret, frame_original = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame_original, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        
        mask_green = crear_mascara_verde(frame_small)
        kernel = np.ones((7, 7), np.uint8)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            mask_area = np.zeros(mask_green.shape, dtype=np.uint8)
            cv2.drawContours(mask_area, [c], -1, 255, thickness=cv2.FILLED)

            gray_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            blurred_small = cv2.GaussianBlur(gray_small, (5, 5), 0)
            th_small = cv2.adaptiveThreshold(blurred_small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
            
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            th_small = cv2.morphologyEx(th_small, cv2.MORPH_OPEN, kernel_morph)
            th_small = cv2.bitwise_and(th_small, th_small, mask=mask_area)

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(th_small, 8)

            centroides_actuales = []
            valid_dados_indices = []

            for i in range(1, num_labels):
                x, y, w, h, area = stats[i]
                aspect_ratio = w / h
                if 300 < area < 3000 and 0.6 < aspect_ratio < 1:
                    cx, cy = centroids[i]
                    centroides_actuales.append((cx, cy))
                    valid_dados_indices.append(i)

            en_reposo, frames_quietos = verificar_reposo(centroides_actuales, centroides_previos, frames_quietos, UMBRAL_MOVIMIENTO, FRAMES_REPOSO)
            
            if not en_reposo:
                if frames_quietos == 0:
                    analisis_realizado = False
                    dados_dibujo = []

            centroides_previos = centroides_actuales

            if en_reposo and not analisis_realizado:
                dados_dibujo = []
                
                for idx_lista, idx_stat in enumerate(valid_dados_indices):
                    xs, ys, ws, hs, _ = stats[idx_stat]
                    inv_scale = 1.0 / SCALE_FACTOR
                    x_real = int(xs * inv_scale)
                    y_real = int(ys * inv_scale)
                    w_real = int(ws * inv_scale)
                    h_real = int(hs * inv_scale)
                    
                    h_img, w_img = frame_original.shape[:2]
                    x_real = max(0, x_real)
                    y_real = max(0, y_real)
                    w_real = min(w_img - x_real, w_real)
                    h_real = min(h_img - y_real, h_real)

                    roi_hd = frame_original[y_real:y_real+h_real, x_real:x_real+w_real]
                    roi_gray = cv2.cvtColor(roi_hd, cv2.COLOR_BGR2GRAY)
                    roi_blur = cv2.GaussianBlur(roi_gray, (9, 9), 2)
                    
                    min_dist = max(15, w_real // 7) 
                    min_rad = max(4, w_real // 25) 
                    max_rad = max(min_rad + 1, w_real // 7)

                    circles = cv2.HoughCircles(roi_blur, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=120, param2=12, minRadius=min_rad, maxRadius=max_rad)

                    roi_puntos_count = 0
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        roi_puntos_count = len(circles[0, :])
                        
                    dados_dibujo.append({"id": idx_lista + 1, "puntos": roi_puntos_count, "box": (x_real, y_real, w_real, h_real)})

                analisis_realizado = True

            if analisis_realizado and dados_dibujo:
                dibujar_resultado(frame_original, dados_dibujo)
                for dado in dados_dibujo:
                    dados[f'Dado {dado["id"]}'] = dado["puntos"]

        vis_frame = cv2.resize(frame_original, None, fx=0.5, fy=0.5)
        cv2.imshow("Video Principal", vis_frame)
        
        key = cv2.waitKey(1 if analisis_realizado else 25)
        if key & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return dados


def detectar_dados2(nombre_video):
    dados = {}

    cap = cv2.VideoCapture(nombre_video)

    SCALE_FACTOR = 0.5
    FRAMES_REPOSO = 8
    UMBRAL_MOVIMIENTO = 3.0
    CANTIDAD_DADOS_BUSCADOS = 5
    ITERACIONES_EROSION = 3    
    MIN_AREA_DETECTADA = 25 

    centroides_previos = []
    frames_quietos = 0
    analisis_realizado = False
    dados_dibujo = []

    while True:
        ret, frame_original = cap.read()
        if not ret:
            break

        frame_small = cv2.resize(frame_original, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        
        mask_green = crear_mascara_verde(frame_small)
        mask_mesa = obtener_area_mesa(mask_green)

        if mask_mesa is not None:
            frame_masked = cv2.bitwise_and(frame_small, frame_small, mask=mask_mesa)
            gray_small = cv2.cvtColor(frame_masked, cv2.COLOR_BGR2GRAY)

            _, th_dados = cv2.threshold(gray_small, 75, 255, cv2.THRESH_BINARY_INV)
            
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            th_dados = cv2.morphologyEx(th_dados, cv2.MORPH_OPEN, kernel_noise)
            kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            th_dados = cv2.morphologyEx(th_dados, cv2.MORPH_CLOSE, kernel_morph)

            th_separated = cv2.erode(th_dados, kernel_morph, iterations=ITERACIONES_EROSION)

            cont_dados, _ = cv2.findContours(th_separated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            dados_candidatos = []
            centroides_actuales = []

            for c in cont_dados:
                area = cv2.contourArea(c)
                if area < MIN_AREA_DETECTADA or area > 1000:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                aspect = w / h

                if 0.55 < aspect < 1.6:
                    padding = int(12 + (ITERACIONES_EROSION * 1.8))
                    x_real = max(0, x - padding)
                    y_real = max(0, y - padding)
                    w_real = w + (padding * 2)
                    h_real = h + (padding * 2)
                    cx = x_real + w_real / 2
                    cy = y_real + h_real / 2
                    dados_candidatos.append({"coords": (x_real, y_real, w_real, h_real)})
                    centroides_actuales.append((cx, cy))

            centroides_actuales.sort(key=lambda p: p[0])
            en_reposo = False
            
            if centroides_previos and len(centroides_actuales) == len(centroides_previos):
                desplazamientos = [math.dist(ca, cp) for ca, cp in zip(centroides_actuales, centroides_previos)]
                if desplazamientos and np.mean(desplazamientos) < UMBRAL_MOVIMIENTO:
                    frames_quietos += 1
                else:
                    frames_quietos = 0
                    analisis_realizado = False
                    dados_dibujo = []
                
                if frames_quietos >= FRAMES_REPOSO:
                    if len(centroides_actuales) == CANTIDAD_DADOS_BUSCADOS:
                        en_reposo = True
            else:
                frames_quietos = 0
                analisis_realizado = False
                dados_dibujo = []
            
            centroides_previos = centroides_actuales

            if en_reposo and not analisis_realizado:
                dados_dibujo = []
                H_img, W_img, _ = frame_original.shape
                inv = 1.0 / SCALE_FACTOR

                for idx, dado in enumerate(dados_candidatos):
                    xs, ys, ws, hs = dado["coords"]
                    x = int(xs * inv)
                    y = int(ys * inv)
                    w = int(ws * inv)
                    h = int(hs * inv)
                    
                    x = max(0, x)
                    y = max(0, y)
                    if x+w > W_img:
                        w = W_img - x
                    if y+h > H_img:
                        h = H_img - y

                    roi = frame_original[y:y+h, x:x+w]
                    if roi.size == 0:
                        continue

                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
                    
                    circles = cv2.HoughCircles(blur_roi, cv2.HOUGH_GRADIENT, dp=1, minDist=max(5, w // 6), param1=50, param2=15, minRadius=2, maxRadius=int(w/5))

                    puntos = 0
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for i in circles[0, :]:
                            cx_p, cy_p = i[0], i[1]
                            if 3 < cx_p < w-3 and 3 < cy_p < h-3:
                                puntos += 1

                    dados_dibujo.append({"id": idx + 1, "puntos": puntos, "box": (x, y, w, h)})
                
                analisis_realizado = True

            if analisis_realizado and dados_dibujo:
                dibujar_resultado(frame_original, dados_dibujo)
                for dado in dados_dibujo:
                    dados[f'Dado {dado["id"]}'] = dado["puntos"]

            vis_frame = cv2.resize(frame_original, None, fx=0.4, fy=0.4)
            cv2.imshow("Video", vis_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

    return dados


def detectar_dados_robusto(nombre_video, cantidad_esperada=5):
    resultado_general = detectar_dados(nombre_video)
    cantidad_detectada = len(resultado_general)

    if cantidad_detectada == cantidad_esperada:
        return resultado_general
    else:
        return detectar_dados2(nombre_video)


# Ejecucion
for i in range(1, 5):
    print(f"\nVIDEO {i}")
    resultado = detectar_dados_robusto(f"tirada_{i}.mp4")
    
    if isinstance(resultado, dict):
        for dado, puntos in resultado.items():
            print(f"    {dado}: {puntos}")