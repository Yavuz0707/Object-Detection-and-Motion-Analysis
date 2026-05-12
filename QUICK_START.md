# 🚀 HIZLI BAŞLANGIÇ GÜİDÜ / QUICK START GUIDE

**Görüntü İşleme Dersi - Deney 9: Nesne Tespiti ve Hareket Analizi**

---

## ⚡ 5 DAKİKA İÇİNDE BAŞLAYIN / START IN 5 MINUTES

### ADIM 1: Terminal Açın
```powershell
cd "c:\Users\sukru\Desktop\New folder\image_processing_lab"
```

### ADIM 2: Programı Çalıştırın
```bash
python main.py
```

### ADIM 3: Menüden Seçim Yapın
```
1. Background Subtraction (Arka Plan Çıkarma)
2. HOG Descriptor (HOG Tanımlayıcısı)
3. Optical Flow Analysis (Optik Akış Analizi)
4. Integrated Warehouse System (Entegre Depo Sistemi)
5. Run All Examples (Tüm Örnekleri Çalıştır)
6. View Documentation (Belgeleri Görüntüle)
0. Exit (Çık)
```

---

## 📖 ÖRNEK KULLANIMLARI / EXAMPLE USAGES

### Arka Plan Çıkarma Kullanması
```python
from src.algorithms.background_subtraction import MOG2BackgroundSubtractor

# MOG2 oluştur
mog2 = MOG2BackgroundSubtractor()

# Video dosyasını aç
import cv2
cap = cv2.VideoCapture('video.mp4')

# Her frame işle
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Ön plan maskesini al
    foreground = mog2.process_frame(frame)
    
    # Görüntüle
    cv2.imshow('Foreground', foreground)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### HOG Özellik Çıkarımı
```python
from src.algorithms.hog_descriptor import HOGDescriptor

# HOG oluştur
hog = HOGDescriptor()

# Görüntüyü yükle
import cv2
image = cv2.imread('image.jpg')

# Özelliği çıkar
features = hog.extract_features(image)
print(f"Feature vector size: {len(features)}")
```

### Optik Akış Analizi
```python
from src.algorithms.optical_flow import OpticalFlowFarneback

# Farneback optik akış
farneback = OpticalFlowFarneback()

# Videodan akış hesapla
import cv2
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Yoğun optik akış
    vis, flow = farneback.process_frame(frame)
    
    # Görüntüle
    cv2.imshow('Dense Optical Flow', vis)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

---

## 📁 DOSYA YAPISI / FILE STRUCTURE

```
image_processing_lab/
├── main.py                    ← Buradan başlayın / Start here
├── README.md                  ← Detaylı belgeler
├── SETUP.md                   ← Kurulum kılavuzu
├── PROJECT_STATUS.md          ← Proje durumu
├── requirements.txt           ← Bağımlılıklar
├── config.py                  ← Yapılandırma
│
├── src/
│   ├── algorithms/
│   │   ├── background_subtraction.py
│   │   ├── hog_descriptor.py
│   │   └── optical_flow.py
│   └── utils/
│       └── image_handler.py
│
├── examples/
│   ├── 01_background_subtraction.py
│   ├── 02_hog_descriptor.py
│   ├── 03_optical_flow.py
│   └── 04_integrated_system.py
│
└── output/                    ← Çıkış dosyaları
```

---

## 🎯 DÖRDÜNCÜ GÖREV / FOUR MAIN TASKS

### 1️⃣ BACKGROUND SUBTRACTION
**Amaç:** Hareketli nesneleri statik arka plandan ayırmak

**Uygulanan Yöntemler:**
- Klasik frame farkı
- MOG2 (Gaussian Mixture Model)
- Morfolojik işlemler

**Çalıştırın:**
```bash
python examples/01_background_subtraction.py
```

---

### 2️⃣ HOG DESCRIPTOR
**Amaç:** İnsan tespiti için özellik vektörü çıkarmak

**Uygulanan Yöntemler:**
- Gradient hesaplama
- Hücre histogramları (8x8)
- L2 normalizasyon
- Scikit-image entegrasyonu

**Çalıştırın:**
```bash
python examples/02_hog_descriptor.py
```

---

### 3️⃣ OPTICAL FLOW
**Amaç:** Nesnelerin hareketi analiz etmek

**Uygulanan Yöntemler:**
- Lucas-Kanade (seyrek)
- Farneback (yoğun)
- HSV görselleştirmesi

**Çalıştırın:**
```bash
python examples/03_optical_flow.py
```

---

### 4️⃣ INTEGRATED SYSTEM
**Amaç:** Tüm algoritmalar birleştirilerek depo sistemi uygulaması

**Özellikler:**
- Nesne tespiti
- Hareket analizi
- İstatistikler
- Gerçek zamanlı işleme

**Çalıştırın:**
```bash
python examples/04_integrated_system.py
```

---

## ❓ SIKI SORULAR VE CEVAPLARI / FAQ

**S: Program açılmıyor / Program won't start**
```bash
# Bağımlılıkları yeniden yükleyin / Reinstall dependencies
pip install -r requirements.txt
```

**S: Kamera bulunamıyor / No camera found**
→ Program otomatik olarak test videoları oluşturur

**S: OpenCV pencereleri görmüyorum / No OpenCV windows**
```bash
# Linux kullanıyorsanız / If using Linux
sudo apt-get install libsm6 libxext6
```

**S: Hızı nasıl arttırırım? / How to speed up?**
→ `config.py` dosyasında `VIDEO_CONFIG` ayarlarını düşürün

---

## 🔧 AYARLAMALAR / QUICK SETTINGS

### config.py Dosyasında:

```python
# Video çözünürlüğü
VIDEO_CONFIG = {
    'default_width': 320,   # 640 → 320 (hızlı)
    'default_height': 240,  # 480 → 240 (hızlı)
}

# Background Subtraction eşiği
BACKGROUND_SUBTRACTION_CONFIG = {
    'classical': {
        'threshold': 50,    # 30 → 50 (daha az alarm)
    }
}
```

---

## 📊 ÖNEMLİ FORMÜLLER / KEY FORMULAS

### Frame Farkı
```
Foreground = |I[t] - Background| > Threshold
```

### HOG Normalizasyon
```
Normalized = Feature / √(Σ Feature²)
```

### Optical Flow
```
Lucas-Kanade:  ∇I · v = -∂I/∂t
Farneback:     f(x) ≈ polynomial expansion
```

---

## ⏱️ PERFORMANS / PERFORMANCE

| Algoritma | Hız | Doğruluk | CPU |
|-----------|-----|----------|-----|
| Classical BG | ⚡⚡⚡ | ⭐⭐ | Düşük |
| MOG2 | ⚡⚡ | ⭐⭐⭐ | Orta |
| HOG | ⚡ | ⭐⭐⭐ | Orta |
| Lucas-Kanade | ⚡⚡⚡ | ⭐⭐ | Düşük |
| Farneback | ⚡ | ⭐⭐⭐ | Yüksek |

---

## 📚 KAYNAKLAR / RESOURCES

- OpenCV Docs: https://docs.opencv.org/
- Scikit-image: https://scikit-image.org/
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/

---

## 🎓 REFERANSLAR / REFERENCES

1. Bradski & Kaehler (2008) - Learning OpenCV
2. Dalal & Triggs (2005) - HOG for Human Detection
3. Lucas & Kanade (1981) - Optical Flow
4. Farnebäck (2003) - Dense Motion Estimation
5. Stauffer & Grimson (1999) - Adaptive Background Subtraction

---

## ✅ KONTROL LİSTESİ / CHECKLIST

Programı çalıştırmadan önce:

- [ ] Python 3.7+ yüklü
- [ ] Bağımlılıklar kuruldu (`pip install -r requirements.txt`)
- [ ] Proje klasörüne gittiniz
- [ ] `python main.py` çalıştırdınız
- [ ] Menüden örnek seçtiniz
- [ ] Pencere 'q' tuşu ile kapandı

---

## 🆘 HATA AYIKLAMA / DEBUGGING

**Hata mesajı alırsanız:**

1. Hatayı okuyun ve not edin
2. Google'da arayın veya README.md kontrol edin
3. Bağımlılıkları yeniden yükleyin:
```bash
pip install --upgrade -r requirements.txt
```

4. Python sürümünü kontrol edin:
```bash
python --version  # 3.7+ olmalı
```

---

## 🎉 BAŞARILI!

Tebrikler! Proje hazır ve çalışmaya başlayabilirsiniz.

Eğer sorun yaşarsanız, README.md veya SETUP.md dosyalarını kontrol edin.

**İyi çalışmalar! / Happy coding!**

---

**Arş. Gör. Ahmet Bestami Köse**  
ISPARTA UYGULAMALI BİLİMLER ÜNİVERSİTESİ
