# imports
import cv2
import numpy as np

# carrega os nomes da classe do módulo COCO
with open('../../input/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

# categoriza por cores os objetos analisados
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# carrega o modelo DNN
model = cv2.dnn.readNet(model='../../input/frozen_inference_graph.pb',
                        config='../../input/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', 
                        framework='TensorFlow')

# carregamento da imagem
image = cv2.imread('../../input/image3.jpg')
image_height, image_width, _ = image.shape

# criar o blob, para trabalhar com armazenamentos de dados binários
blob = cv2.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), 
                             swapRB=True)
# define o blob inicial para a rede neural
model.setInput(blob)

# conversão para o framework da ferramenta
output = model.forward()

# faz o loop sobre cada detecção
for detection in output[0, 0, :, :]:
    
    # extrai o grau de confiabilidade da detecção
    confidence = detection[2]
    
    # colocar quadrados em volta dos padrões se o grau de confiabilidade for maior que um certo limite, ou pula a detecção
    if confidence > .4:
        
        # obtem o ID doobjeto
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
        
        # coloca o FPS na imagem
        cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

# gera a imagem, coloca o texto e salva a imagem no diretório output
cv2.imshow('image', image)
cv2.imwrite('../../outputs/image_result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
