# SETUP GUIDE / KURULUM KILAVUZU

## Türkçe

### Sistem Gereksinimleri

- Python 3.7 veya daha yeni
- Windows, macOS veya Linux
- Minimum 2GB RAM (4GB önerilir)
- Cam (isteğe bağlı, bazı örnekler canlı video için)

### Kurulum Adımları

#### 1. Python Kurulumu

Windows kullanıyorsanız:
1. [python.org](https://www.python.org/downloads/) adresinden Python indiriniz
2. Kurarken "Add Python to PATH" seçeneğini işaretleyiniz
3. Kurulumu tamamlayınız

macOS/Linux:
```bash
# macOS (Homebrew kullanıyor)
brew install python3

# Ubuntu/Debian
sudo apt-get install python3 python3-pip
```

#### 2. Proje Dosyalarını Hazırlayın

```bash
# Proje klasörüne gidin
cd "Desktop/New folder/image_processing_lab"
```

#### 3. Sanal Ortam Oluşturun (Önerilir)

```bash
# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı etkinleştir
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 4. Bağımlılıkları Yükleyin

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Kurulum sırasında hata alırsanız:

```bash
# Alternatif: Paketleri ayrı ayrı yükleyin
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-image==0.21.0
pip install Pillow==10.0.0
```

#### 5. Programı Çalıştırın

```bash
# Ana menüyü açın
python main.py

# Veya doğrudan bir örneği çalıştırın
python examples/01_background_subtraction.py
python examples/02_hog_descriptor.py
python examples/03_optical_flow.py
python examples/04_integrated_system.py
```

### Sorun Giderme / Troubleshooting

**Hata: "ModuleNotFoundError: No module named 'cv2'"**
```bash
pip install --upgrade opencv-python
```

**Hata: "No camera found" (Kamera bulunamadı)**
- Bazı örnekler otomatik olarak test videoları oluştururlar
- İhtiyaç durumunda `examples/` içinde video çıktıları kullanılır

**Hata: "Permission denied" (İzin reddedildi)**
```bash
# macOS/Linux için
sudo chmod +x main.py
```

**Hata: OpenCV penceresi açılmıyor**
- Ubuntu kullanıyorsanız:
```bash
sudo apt-get install libsm6 libxext6
```

---

## English

### System Requirements

- Python 3.7 or newer
- Windows, macOS, or Linux
- Minimum 2GB RAM (4GB recommended)
- Webcam (optional, required for live video examples)

### Installation Steps

#### 1. Python Installation

For Windows:
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Check "Add Python to PATH" during installation
3. Complete the installation

For macOS/Linux:
```bash
# macOS (using Homebrew)
brew install python3

# Ubuntu/Debian
sudo apt-get install python3 python3-pip
```

#### 2. Prepare Project Files

```bash
# Navigate to project folder
cd "Desktop/New folder/image_processing_lab"
```

#### 3. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If installation errors occur:

```bash
# Alternative: Install packages individually
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-image==0.21.0
pip install Pillow==10.0.0
```

#### 5. Run the Program

```bash
# Open main menu
python main.py

# Or run specific examples
python examples/01_background_subtraction.py
python examples/02_hog_descriptor.py
python examples/03_optical_flow.py
python examples/04_integrated_system.py
```

### Troubleshooting

**Error: "ModuleNotFoundError: No module named 'cv2'"**
```bash
pip install --upgrade opencv-python
```

**Error: "No camera found"**
- Some examples automatically create test videos
- Output videos are saved in `output/` directory

**Error: "Permission denied"**
```bash
# For macOS/Linux
sudo chmod +x main.py
```

**Error: OpenCV window not opening**
- For Ubuntu:
```bash
sudo apt-get install libsm6 libxext6
```

---

## Programı Kullanma / Using the Program

### Main Menu Options / Ana Menü Seçenekleri

```
1. Background Subtraction (Arka Plan Çıkarma)
   - Demonstrates classical background subtraction methods
   - Shows MOG2 algorithm
   - Compares threshold values

2. HOG Descriptor (HOG Tanımlayıcısı)
   - Feature extraction using HOG
   - Scikit-image implementation
   - Cell size comparisons
   - Brightness invariance analysis

3. Optical Flow Analysis (Optik Akış Analizi)
   - Lucas-Kanade sparse optical flow
   - Farneback dense optical flow
   - HSV color visualization
   - Method comparison and performance analysis

4. Integrated Warehouse System (Entegre Depo Sistemi)
   - Complete object detection system
   - Real-time monitoring
   - Motion statistics
   - Performance metrics

5. Run All Examples (Tüm Örnekleri Çalıştır)
   - Runs all demonstrations sequentially

6. View Documentation (Belgeleri Görüntüle)
   - Shows quick reference guide

0. Exit (Çık)
```

### Output Files / Çıkış Dosyaları

Programs save outputs to the `output/` directory:

- `sample_video.mp4` - Test video for background subtraction
- `motion_video.mp4` - Test video for optical flow
- `warehouse_video.mp4` - Warehouse monitoring test video
- `hog_features.png` - HOG feature visualization
- `hog_cell_sizes.png` - HOG cell size comparison
- Various temporary analysis images

### Keyboard Shortcuts / Klavye Kısayolları

In visualization windows:
- `q` - Quit current demo
- `ESC` - Close window
- Arrow keys - Navigate (if implemented)

---

## Advanced Configuration / İleri Ayarlar

### Custom Parameters

Edit the example files to modify parameters:

**Background Subtraction:**
```python
# In 01_background_subtraction.py
threshold = 30  # Change threshold value
history = 500   # MOG2 history
```

**HOG Descriptor:**
```python
# In 02_hog_descriptor.py
cell_size = 8        # Cell size in pixels
nbins = 9           # Number of orientation bins
```

**Optical Flow:**
```python
# In 03_optical_flow.py
max_corners = 100        # Lucas-Kanade max corners
num_levels = 3          # Farneback pyramid levels
window_size = 15        # Farneback window size
```

### Video Input/Output

To use your own video files:

```python
# Edit example file
video_path = "path/to/your/video.mp4"
```

Supported formats:
- MP4, AVI, MOV, WMV, FLV, MKV

---

## Quick Reference / Hızlı Referans

### Import Algorithms

```python
# Background Subtraction
from src.algorithms.background_subtraction import (
    BackgroundSubtraction, MOG2BackgroundSubtractor
)

# HOG Descriptor
from src.algorithms.hog_descriptor import (
    HOGDescriptor, extract_hog_features_scikit
)

# Optical Flow
from src.algorithms.optical_flow import (
    OpticalFlowLucasKanade, OpticalFlowFarneback
)

# Utilities
from src.utils.image_handler import ImageHandler
```

### Basic Usage Examples

```python
import cv2

# 1. Background Subtraction
mog2 = MOG2BackgroundSubtractor()
mask = mog2.process_frame(frame)

# 2. HOG Features
hog = HOGDescriptor()
features = hog.extract_features(image)

# 3. Optical Flow (Lucas-Kanade)
lk = OpticalFlowLucasKanade()
vis, vectors = lk.process_frame(frame)

# 4. Optical Flow (Farneback)
farneback = OpticalFlowFarneback()
vis, flow = farneback.process_frame(frame)
```

---

## Performance Tips / Performans İpuçları

1. **Use Virtual Environment** - Prevents conflicts with system packages
2. **Reduce Frame Size** - Resize frames for faster processing
3. **Lucas-Kanade for Real-time** - Faster than dense optical flow
4. **MOG2 for Robustness** - Better than classical background subtraction
5. **GPU Acceleration** - Install opencv-contrib-python for CUDA support

---

## Support / Destek

For issues or questions:
1. Check the README.md file
2. Review example files
3. Check OpenCV documentation: https://docs.opencv.org/

---

## License / Lisans

Educational project for ISPARTA UYGULAMALI BİLİMLER ÜNİVERSİTESİ

---

**Last Updated: 2024**
