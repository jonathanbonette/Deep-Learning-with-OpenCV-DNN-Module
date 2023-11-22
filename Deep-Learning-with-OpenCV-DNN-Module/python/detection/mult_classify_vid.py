# imports
import cv2
import time
import numpy as np

# carrega o modelo DNN
with open('../../input/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# categoriza por cores os objetos analisados
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# carrega o modelo DNN
model = cv2.dnn.readNet(model='../../input/frozen_inference_graph.pb',
                        config='../../input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

# carregamento da imagem
cap = cv2.VideoCapture('../../    /video1.mp4')

# faz um tratamento do tamano do vídeo
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# cria um objeto do tipo `VideoWriter()`
out = cv2.VideoWriter('../../outputs/vid_result.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

# faz o loop sobre cada detecção de cada frame do vídeo
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        image = frame
        image_height, image_width, _ = image.shape
        
        # criar o blob, para trabalhar com armazenamentos de dados binários
        blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                                     swapRB=True)
        
        # tratamento do começo do tempo do vídeo
        start = time.time()
        model.setInput(blob)
        output = model.forward()    
        
        # tratamento do final do tempo do vídeo
        end = time.time()
        
        # calcula o FPS atual de cada detecção
        fps = 1 / (end-start)
        
        # faz o loop sobre cada detecção
        for detection in output[0, 0, :, :]:
            
            # extrai o grau de confiabilidade da detecção
            confidence = detection[2]
            
            # colocar quadrados em volta dos padrões se o grau de confiabilidade for maior que um certo limite, ou pula a detecção
            if confidence > .4:
                
                # obtem o ID do objeto
                class_id = detection[1]
                
                # mapeia a classe
                class_name = class_names[int(class_id)-1]
                color = COLORS[int(class_id)]
                
                # obtem as coordenadas da caixa que ficará em volta dos objetos observados
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                
                # pega o limite da caixa e define uma altura e largura
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height

                # desenha a caixa em volta de cada objeto detectado
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                
                # coloca o nome da classe na caixa criada
                cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # faz a contagem do FPS
                cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
        
        # gera a imagem, coloca o texto e salva a imagem no diretório output (q == sair)
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

# libera a vídeo
cap.release()
cv2.destroyAllWindows()
