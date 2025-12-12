import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("frames", exist_ok = True)

cap = cv2.VideoCapture('tirada_4.mp4')  # Abre el archivo de video especificado ('tirada_1.mp4') para su lectura.
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Obtiene el ancho del video en píxeles usando la propiedad CAP_PROP_FRAME_WIDTH.
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Obtiene la altura del video en píxeles usando la propiedad CAP_PROP_FRAME_HEIGHT.
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Obtiene los cuadros por segundo (FPS) del video usando CAP_PROP_FPS.
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Obtiene el número total de frames en el video usando CAP_PROP_FRAME_COUNT.


frame_number = 100
while (cap.isOpened()): # Verifica si el video se abrió correctamente.

    ret, frame = cap.read() # 'ret' indica si la lectura fue exitosa (True/False) y 'frame' contiene el contenido del frame si la lectura fue exitosa.

    if ret == True:  
        frame = cv2.resize(frame, dsize=(int(width/3), int(height/3))) # Redimensiona el frame capturado.
        #cv2.imshow('Frame', frame) # Muestra el frame redimensionado.
        #cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), frame) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
        edges = cv2.Canny(blurred, 20, 150)  

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Rango de verde (ajustable)
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        # Crear máscara
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Aplicar máscara
        result = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('resultado', result)




        # cv2.imshow('Frame', edges) # Muestra el frame redimensionado.
        # cv2.imwrite(os.path.join("frames", f"frame_{frame_number}.jpg"), edges) # Guarda el frame en el archivo './frames/frame_{frame_number}.jpg'.



        frame_number += 1
        if cv2.waitKey(25) & 0xFF == ord('q'): # Espera 25 milisegundos a que se presione una tecla. Si se presiona 'q' se rompe el bucle y se cierra la ventana.
            break
    else:  
        break  

#cap.release() # Libera el objeto 'cap', cerrando el archivo.
#cv2.destroyAllWindows() # Cierra todas las ventanas abiertas.