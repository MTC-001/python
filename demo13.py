#图片识别
import sys
sys.path.append('./ultralytics') 
from ultralytics import YOLO
yolo = YOLO("./yolov8n.pt", task="detect")
result = yolo(source="./1.webp", save=True)