# utils/model_loader.py

import torch
import timm
import os
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F

def predict(image, model, device):
    """
    Görüntüyü alır; katı renk kontrolü, belirsizlik analizi ve tahmin yapar.
    -2: Renkli veya RGB formatında (Hata)
    -1: Belirsiz (Kararsız tahmin)
    """
    
    # 1. KATI RENK FİLTRESİ
    img_array = np.array(image)
    
    # Görüntü RGB formatında 3 kanallı ise
    if img_array.ndim == 3 and img_array.shape[2] == 3:
        # Kanallar arasındaki mutlak farkı hesapla (R-G ve G-B)
        # Siyah-beyaz bir röntgende R, G ve B değerleri eşittir, fark 0 olmalıdır.
        diff_rg = np.abs(img_array[:,:,0] - img_array[:,:,1]).mean()
        diff_gb = np.abs(img_array[:,:,1] - img_array[:,:,2]).mean()
        
        # Eğer toplam fark 5'ten büyükse, bu görselde renkli piksel yoğunluğu vardır
        if (diff_rg + diff_gb) > 5: 
            return -2, 0.0 

    # 2. Ön işleme
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # 3. Tahmin
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # En yüksek olasılık ve indeks
        confidence, pred_idx = torch.max(probabilities, 1)
        
        # 4. İSTATİSTİKSEL FİLTRE (Belirsizlik kontrolü)
        prob_diff = torch.abs(probabilities[0][0] - probabilities[0][1])
        if prob_diff < 0.2: 
            return -1, 0.0 

        return pred_idx.item(), confidence.item()

def get_clean_state_dict(checkpoint_path, device):
    """
    Dosyayı yükler ve içindeki gerçek ağırlıkları (state_dict) ayıklar.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        keys = ['model_state_dict', 'state_dict', 'model', 'net']
        for key in keys:
            if key in checkpoint:
                return checkpoint[key]
        return checkpoint
    
    return checkpoint

def get_model(device=None):
    """
    Modeli mimariye göre oluşturur ve ağırlıkları yükler.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    model = timm.create_model('convnextv2_base', pretrained=False, num_classes=2)
    checkpoint_path = "models/best_convnextv2_base.fcmae_ft_in22k_in1k.pth"
    
    try:
        state_dict = get_clean_state_dict(checkpoint_path, device)
        model.load_state_dict(state_dict)
        print("Model ağırlıkları başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Model ağırlıkları yüklenemedi: {e}")
        raise e
    
    model.to(device)
    model.eval()
    return model