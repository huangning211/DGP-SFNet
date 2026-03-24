from ultralytics import YOLO
import os
import torch
from torch import nn

model_yaml = os.path.abspath(r'model/DGP-SFNet.yaml')
data_yaml = os.path.abspath(r"USMD.yaml")

model = YOLO(model_yaml, task='detect')


if __name__ == '__main__':
    results = model.train(
        data=data_yaml,         
        batch=48,              
        epochs=300,                
        imgsz=640,                
        resume=False,
        workers=8,                
        pretrained=False,          
        optimizer='AdamW',
        lr0=0.0025,             
        lrf=0.1,                 

        augment=True,
        hsv_h=0.05,               
        hsv_s=0.2,                  
        hsv_v=0.3,                
        degrees=0,             
        translate=0.1,
        flipud=0.0,                
        fliplr=0.5,                

        close_mosaic = 50,   
        weight_decay=0.0015,       
        label_smoothing=0.1,

        patience=15,               
        save_period=10,
        cache='ram',              

        exist_ok=True,
        plots=True,
        seed=42
    )
