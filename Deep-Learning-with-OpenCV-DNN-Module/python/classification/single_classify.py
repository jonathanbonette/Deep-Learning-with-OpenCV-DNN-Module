# imports
import cv2
import numpy as np

# leitura das classes de dados treinadas do módulo ImageNet
with open('../../input/classification_classes_ILSVRC2012.txt', 'r') as f:
   image_net_names = f.read().split('\n')
   
# classes categorizadas item por item do arquivo de texto
class_names = [name.split(',')[0] for name in image_net_names]

# carregamento dos modelos neurais
model = cv2.dnn.readNet(model='../../input/DenseNet_121.caffemodel', config='../../input/DenseNet_121.prototxt', framework='Caffe')

# carregamento da imagem
image = cv2.imread('../../input/image1.jpg') 

# criar o blob, para trabalhar com armazenamentos de dados binários
blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))

# define o blob inicial para a rede neural
model.setInput(blob)

# conversão para o framework da ferramenta
outputs = model.forward()
final_outputs = outputs[0]

# faz a análise dos IDs na imagem fixa
final_outputs = final_outputs.reshape(1000, 1)

# pega o ID da classe de dados
label_id = np.argmax(final_outputs)

# converte a saída e faz a análise das possibilidades dos padrões
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

# faz um refinamento das probabilidades, multiplicando por 100
final_prob = np.max(probs) * 100.

# mapeia pela máxima confiança dos IDs reconhecidos
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"

# gera a imagem, coloca o texto e salva a imagem no diretório output
cv2.putText(image, out_text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.imwrite('../../outputs/result_image.jpg', image)