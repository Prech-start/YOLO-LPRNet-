import cv2
from ultralytics import YOLO
from LPRNet_Pytorch.model.LPRNet import build_lprnet
import torch
from windows.mainwindows import mian_window



if __name__ == '__main__':
    YOLO_model = YOLO("runs/detect/train4/weights/best.pt")
    LPR_model = build_lprnet(phase='test')
    # dict_name = f'./LPRNet_Pytorch/Final_LPRNet_model.pth'
    dict_name = f'./LPRNet_Pytorch/weights/LPRNet__iteration_62000.pth'
    LPR_model.load_state_dict(torch.load(dict_name, map_location=torch.device('cpu')))
    
    # 读取图像
    image = cv2.imread("car2.png")
    mian_window(YOLO_model=YOLO_model, LPR_model=LPR_model)
    
    
