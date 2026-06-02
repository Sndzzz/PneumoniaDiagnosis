# utils/visualization.py
import sys
import os

# Projenin ana dizinini (zatürre-tespit-projesi) Python'ın arama yoluna ekle
# __file__, mevcut dosyanın yolu (utils/visualization.py)
# os.path.dirname(os.path.dirname(__file__)) bu kodun 2 üst dizine (ana klasöre) çıkmasını sağlar
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Artık import satırların sistemde kurulu kütüphaneyi değil, 
# klasördeki yerel kopyanı kullanacak
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

def generate_gradcam(model, pillow_image, device): # <--- 3. parametreyi ekledik
    # 1. Görseli ön işleme al
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize işlemini eğitimdekiyle aynı tutmalısın
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    if pillow_image.mode != "RGB":
        pillow_image = pillow_image.convert("RGB")
        
    # 2. Tensor'ü cihaza taşı
    input_tensor = transform(pillow_image).unsqueeze(0).to(device)
    
    # 3. ConvNeXtV2 için hedef katman (Modeli cihaza taşıdığından emin ol)
    model = model.to(device)
    target_layers = [model.stages[-1].blocks[-1]]
    
    # 4. GradCAM oluştur
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # 5. Isı haritasını üret
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    
    # 6. Görseli birleştir
    rgb_img = np.float32(pillow_image.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return cam_image