from ultralytics import YOLO
import os
import cv2
from math import sqrt

diretorio_raiz = ''
arquivo_config = os.path.join(diretorio_raiz, 'configs_detection.yaml')

model = YOLO('yolov8s.yaml')

resultados = model.train(data=arquivo_config, epochs=250, imgsz=640, name='yolov8s_ensemble', batch=3) # Treinando o modelo
