import sys
import os
import streamlit as st
import torch
from PIL import Image

# 1. Modül yollarını projenin ana dizini olarak tanımla
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# 2. Yardımcı dosyalarından fonksiyonları import et
from utils.model_loader import get_model, predict 
from utils.visualization import generate_gradcam

# Sayfa Yapılandırması
st.set_page_config(page_title="PneumoVision - Klinik Karar Destek", layout="wide")

# 3. Model Yükleme (Önbellekli)
@st.cache_resource
def load_backend_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    return model, device

model, device = load_backend_model()

# Başlık ve Açıklama
st.title("🩺 PneumoVision: Akciğer Röntgeni ile Zatürre Teşhis Sistemi")
st.write("Doktorlar için geliştirilmiş yapay zekâ tabanlı klinik karar destek arayüzü.")
st.markdown("---")

# Sol panel: Dosya Yükleme
st.sidebar.header("Hasta Veri Girişi")
uploaded_file = st.sidebar.file_uploader("Bir Akciğer Röntgeni (X-Ray) Yükleyin", type=["jpg", "jpeg", "png"])

class_names = ["SAĞLIKLI (NORMAL)", "ZATÜRRE (PNEUMONIA)"]
CONFIDENCE_THRESHOLD = 0.85 

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        
        # Temel boyut kontrolü
        if image.size[0] < 200 or image.size[1] < 200:
            st.error("❌ **Yetersiz Çözünürlük:** Yüklenen görsel klinik analiz için gereken minimum standartların altındadır.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Yüklenen Röntgen Görüntüsü")
                st.image(image, use_container_width=True)
            
            # --- Analiz Bölümü ---
            with st.spinner("Model görüntüyü analiz ediyor..."):
                pred_idx, confidence = predict(image, model, device)
                
                # Grad-CAM'i sadece analiz başarılıysa (pred_idx >= 0) üret
                cam_image = None
                if pred_idx >= 0:
                    cam_image = generate_gradcam(model, image, device)
            
            # --- Karar Mekanizması ---
            if pred_idx == -2:
                st.error("❌ **Görüntü Formatı Uygun Değil:** Yüklenen dosya renkli veri içermektedir. Tanısal doğruluk için lütfen standart gri tonlamalı (monokrom) bir akciğer röntgeni yükleyiniz.")
            
            elif pred_idx == -1:
                st.error("❌ **Analiz Edilemedi:** Görüntü, klinik standartlardaki radyolojik verilerle eşleşmemektedir. Lütfen daha net bir röntgen görüntüsü ile tekrar deneyiniz.")
            
            elif confidence < CONFIDENCE_THRESHOLD:
                st.warning(f"⚠️ **Düşük Tanısal Güven:** Model sonuçtan emin değil (Güven: %{confidence*100:.2f}). Lütfen farklı bir görüntü ile tekrar deneyiniz.")
            
            else:
                # Başarılı durum
                with col2:
                    st.subheader("XAI - Model Odaklanma Alanı (Grad-CAM)")
                    st.image(cam_image, use_container_width=True)
                
                st.markdown("---")
                st.subheader("📊 Analiz Sonucu")
                
                if pred_idx == 1:
                    st.error(f"Tahmin: {class_names[pred_idx]} (Güven: %{confidence*100:.2f})")
                    st.warning("💬 **Klinik Yorum:** Akciğer dokusunda zatürre ile uyumlu opasite bulguları izlenmiştir. Grad-CAM haritasını ilgili radyolojik alanlarla karşılaştırınız.")
                else:
                    st.success(f"Tahmin: {class_names[pred_idx]} (Güven: %{confidence*100:.2f})")
                    st.info("💬 **Klinik Yorum:** İnceleme sonucunda zatürre ile uyumlu belirgin bir patolojik bulguya rastlanmamıştır.")

            st.caption("⚠️ Yasal Uyarı: Bu yazılım bir karar destek aracıdır. Kesin teşhis hekim tarafından konulmalıdır.")

    except Exception as e:
        st.error(f"❌ Hata: Görüntü işlenirken sistemde bir sorun oluştu: {e}")
else:
    st.info("Lütfen analiz başlatmak için sol panelden bir röntgen görseli yükleyin.")