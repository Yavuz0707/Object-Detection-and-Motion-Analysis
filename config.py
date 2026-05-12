"""
Configuration file for Image Processing Lab experiments
Deney parametrelerini ve ayarlarını içerir
"""

# Background Subtraction Configuration
BACKGROUND_SUBTRACTION_CONFIG = {
    'classical': {
        'threshold': 30,                    # Piksel farkı eşiği
        'morphological_kernel': 5,          # Morfolojik işlem çekirdeği
        'morphological_iterations': 2,      # Tekrarlama sayısı
        'min_contour_area': 500             # Minimum kontur alanı
    },
    'mog2': {
        'history': 500,                     # Öğrenme kaçaklığı
        'var_threshold': 16,                # Varyans eşiği
        'detect_shadows': True,             # Gölge algılaması
        'shadow_threshold': 0.5             # Gölge eşiği
    }
}

# HOG Configuration
HOG_CONFIG = {
    'cell_size': 8,                         # Hücre boyutu (piksel)
    'block_size': 2,                        # Blok boyutu (hücreler)
    'nbins': 9,                             # Yönlendirme bölmesi sayısı
    'normalization': 'L2',                  # Normalizasyon tipi
    'orientations': 9,                      # Scikit-image için
    'pixels_per_cell': (8, 8),              # Scikit-image için
    'cells_per_block': (2, 2)               # Scikit-image için
}

# Optical Flow Configuration
OPTICAL_FLOW_CONFIG = {
    'lucas_kanade': {
        'max_corners': 100,                 # Maksimum köşe sayısı
        'quality_level': 0.01,              # Kalite eşiği
        'min_distance': 10,                 # Minimum mesafe
        'block_size': 7,                    # Block size for corner detection
        'use_harris': False                 # Harris köşe dedektörü
    },
    'farneback': {
        'num_levels': 3,                    # Piramit seviyeleri
        'window_size': 15,                  # Pencere boyutu
        'num_iters': 3,                     # İterasyon sayısı
        'poly_n': 5,                        # Polinom derecesi
        'poly_sigma': 1.2,                  # Polinom sigma
        'pyr_scale': 0.5                    # Piramit ölçeği
    }
}

# Video Processing Configuration
VIDEO_CONFIG = {
    'default_fps': 30,                      # Varsayılan FPS
    'default_width': 640,                   # Varsayılan genişlik
    'default_height': 480,                  # Varsayılan yükseklik
    'codec': 'mp4v',                        # Video codec
    'sample_video_duration': 5,             # Örnek video süresi (saniye)
    'warehouse_video_duration': 10          # Depo video süresi (saniye)
}

# Display Configuration
DISPLAY_CONFIG = {
    'show_original': True,                  # Orijinal görseli göster
    'show_foreground': True,                # Ön plan maskesini göster
    'show_flow_sparse': True,               # Seyrek akış göster
    'show_flow_dense': True,                # Yoğun akış göster
    'window_width': 640,                    # Pencere genişliği
    'window_height': 480,                   # Pencere yüksekliği
    'font': 'HERSHEY_SIMPLEX',              # Font türü
    'text_color': (0, 255, 0),              # Metin rengi (BGR)
    'thickness': 2                          # Çizgi kalınlığı
}

# Analysis Parameters
ANALYSIS_CONFIG = {
    'threshold_range': [15, 30, 50, 80],   # Test edilecek eşik değerleri
    'cell_sizes': [4, 8, 16, 32],          # Test edilecek hücre boyutları
    'num_frames_analysis': 30,              # Analiz için kare sayısı
    'statistics_window': 10,                # İstatistik penceresi (kare)
    'motion_threshold': 0.1                 # Hareket eşiği
}

# Warehouse System Configuration
WAREHOUSE_CONFIG = {
    'worker_speed_factor': 0.5,             # İşçi hız faktörü
    'forklift_speed_factor': 2.0,           # Forklift hız faktörü
    'object_detection_area_threshold': 500, # Nesne algılama alan eşiği
    'tracking_max_age': 10,                 # İzleme maksimum yaş (kare)
    'confidence_threshold': 0.5             # Güven eşiği
}

# Performance Thresholds
PERFORMANCE_CONFIG = {
    'min_fps_realtime': 25,                 # Minimum FPS (gerçek zamanlı)
    'max_latency_ms': 40,                   # Maksimum gecikme (ms)
    'memory_warning_threshold': 1024,       # Bellek uyarı eşiği (MB)
    'cpu_warning_threshold': 80             # CPU uyarı eşiği (%)
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',                    # Log seviyesi
    'save_logs': True,                      # Logları kaydet
    'log_file': 'logs/experiment.log',      # Log dosyası
    'detailed_logging': False               # Detaylı logging
}

# Experiment Metadata
EXPERIMENT_CONFIG = {
    'experiment_name': 'Deney 9: Nesne Tespiti ve Hareket Analizi',
    'experiment_number': 9,
    'course_name': 'Görüntü İşleme',
    'university': 'ISPARTA UYGULAMALI BİLİMLER ÜNİVERSİTESİ',
    'instructor': 'Arş. Gör. Ahmet Bestami Köse',
    'student_name': 'Öğrenci Adı',  # Replace with actual name
    'date': '2024-01-01',
    'version': '1.0.0'
}

# Results Directory Configuration
RESULTS_CONFIG = {
    'results_dir': 'output/results',        # Sonuçlar dizini
    'save_images': True,                    # Görüntüleri kaydet
    'save_videos': True,                    # Videoları kaydet
    'save_data': True,                      # Veri dosyalarını kaydet
    'image_format': 'png',                  # Görüntü formatı
    'video_format': 'mp4'                   # Video formatı
}

# Helper Functions
def get_config(section):
    """Get configuration section."""
    configs = {
        'background_subtraction': BACKGROUND_SUBTRACTION_CONFIG,
        'hog': HOG_CONFIG,
        'optical_flow': OPTICAL_FLOW_CONFIG,
        'video': VIDEO_CONFIG,
        'display': DISPLAY_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'warehouse': WAREHOUSE_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'logging': LOGGING_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'results': RESULTS_CONFIG
    }
    return configs.get(section, {})


def update_config(section, key, value):
    """Update configuration value."""
    configs = {
        'background_subtraction': BACKGROUND_SUBTRACTION_CONFIG,
        'hog': HOG_CONFIG,
        'optical_flow': OPTICAL_FLOW_CONFIG,
        'video': VIDEO_CONFIG,
        'display': DISPLAY_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'warehouse': WAREHOUSE_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'logging': LOGGING_CONFIG,
        'experiment': EXPERIMENT_CONFIG,
        'results': RESULTS_CONFIG
    }
    
    if section in configs:
        configs[section][key] = value
        return True
    return False


def print_config(section=None):
    """Print configuration."""
    if section:
        config = get_config(section)
        print(f"\n{section.upper()} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        configs = {
            'background_subtraction': BACKGROUND_SUBTRACTION_CONFIG,
            'hog': HOG_CONFIG,
            'optical_flow': OPTICAL_FLOW_CONFIG,
            'video': VIDEO_CONFIG,
            'display': DISPLAY_CONFIG,
            'analysis': ANALYSIS_CONFIG,
            'warehouse': WAREHOUSE_CONFIG,
            'performance': PERFORMANCE_CONFIG,
            'logging': LOGGING_CONFIG,
            'experiment': EXPERIMENT_CONFIG,
            'results': RESULTS_CONFIG
        }
        
        for section_name, section_config in configs.items():
            print(f"\n{section_name.upper()}:")
            for key, value in section_config.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    # Test configuration
    print_config()
    print("\n" + "="*70)
    print("Configuration loaded successfully!")
