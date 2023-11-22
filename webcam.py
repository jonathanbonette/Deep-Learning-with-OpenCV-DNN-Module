# Código para Webcam
# Adaptado de http://docs.opencv.org/3.2.0/df/d9d/tutorial_py_colorspaces.html
import cv2
import numpy as np
color_to_find=100 #importante: faixa da componente H no OpenCV: [0, 179]
thres=10
cap = cv2.VideoCapture(0) # para usar webcam
while(1):
    # Take each frame
    _, frame = cap.read()
    
    cv2.imshow('frame',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()