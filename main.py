"""
Main entry point for the Image Processing Lab
Experiment 9: Object Detection and Motion Analysis
"""
import os
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def display_menu():
    """Display main menu."""
    print("\n" + "="*70)
    print("ISPARTA UYGULAMALI BİLİMLER ÜNİVERSİTESİ".center(70))
    print("Görüntü İşleme Dersi - Deney 9".center(70))
    print("Nesne Tespiti ve Hareket Analizi".center(70))
    print("="*70)
    print("\nImage Processing Lab - Experiment 9")
    print("Object Detection and Motion Analysis")
    print("="*70)
    print("\nMENU - Menü:\n")
    print("1. Background Subtraction (Arka Plan Çıkarma)")
    print("2. HOG Descriptor (HOG Tanımlayıcısı)")
    print("3. Optical Flow Analysis (Optik Akış Analizi)")
    print("4. Integrated Warehouse System (Entegre Depo Sistemi)")
    print("5. View Documentation (Belgeleri Görüntüle)")
    print("0. Exit (Çık)")
    print("="*70)


def run_example(example_number):
    """Run specific example."""
    examples = {
        1: ("examples/01_background_subtraction.py", "Background Subtraction"),
        2: ("examples/02_hog_descriptor.py", "HOG Descriptor"),
        3: ("examples/03_optical_flow.py", "Optical Flow"),
        4: ("examples/04_integrated_system.py", "Integrated System")
    }
    
    if example_number not in examples:
        print("Invalid selection / Geçersiz seçim")
        return
    
    example_path, example_name = examples[example_number]
    
    print(f"\n{'='*70}")
    print(f"Running: {example_name}".center(70))
    print(f"{'='*70}\n")
    
    try:
        subprocess.run([sys.executable, example_path], check=False)
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


def show_documentation():
    """Show quick reference."""
    print("\n" + "="*70)
    print("QUICK REFERENCE - HIZLI REFERANS".center(70))
    print("="*70)
    
    doc = """
ALGORITHMS / ALGORİTMALAR:

1. Background Subtraction - Arka Plan Çıkarma
   Method: Frame difference and MOG2 (Gaussian Mixture Model)
   Use: Detect moving objects
   
2. HOG Descriptor - HOG Tanımlayıcısı  
   Method: Histogram of Oriented Gradients
   Use: Human detection, feature extraction
   
3. Optical Flow - Optik Akış
   Method: Lucas-Kanade (sparse) and Farneback (dense)
   Use: Motion analysis, object tracking
   
4. Integrated System - Entegre Sistem
   Method: All algorithms combined
   Use: Warehouse monitoring demonstration

QUICK START / HIZLI BAŞLANGIÇ:

from src.algorithms.background_subtraction import MOG2BackgroundSubtractor
mog2 = MOG2BackgroundSubtractor()
mask = mog2.process_frame(frame)

from src.algorithms.hog_descriptor import HOGDescriptor
hog = HOGDescriptor()
features = hog.extract_features(image)

from src.algorithms.optical_flow import OpticalFlowFarneback
flow = OpticalFlowFarneback()
vis, flow_matrix = flow.process_frame(frame)

FILES / DOSYALAR:

- README.md          : Detailed documentation (Türkçe/English)
- SETUP.md           : Installation guide (Kurulum kılavuzu)
- QUICK_START.md     : Quick start guide (Hızlı başlangıç)
- config.py          : Configuration settings (Ayarlar)
"""
    
    print(doc)


def main():
    """Main entry point."""
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    while True:
        display_menu()
        choice = input("\nEnter your choice / Seçiminizi giriniz (0-5): ").strip()
        
        if choice == "0":
            print("\nThank you for using Image Processing Lab!")
            print("Görüntü İşleme Laboratuvarını kullandığınız için teşekkürler!")
            break
        elif choice in ["1", "2", "3", "4"]:
            run_example(int(choice))
            input("\nPress Enter to continue... / Devam etmek için Enter'e basın...")
        elif choice == "5":
            show_documentation()
            input("\nPress Enter to continue... / Devam etmek için Enter'e basın...")
        else:
            print("Invalid choice / Geçersiz seçim. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting... / Program kesintiye uğradı. Çıkılıyor...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
