"""
IMDb Film İnceleme Verileri ile Duygu Analizi ve Görselleştirme

Bu program, IMDb film yorumlarını kullanarak duygu analizi yapar 
ve sonuçları görselleştirir.
"""

import os
import sys
import time
import argparse
from tqdm import tqdm

def create_parser():
    """Komut satırı argümanlarını ayrıştır"""
    parser = argparse.ArgumentParser(
        description='IMDb Film Yorumları ile Duygu Analizi ve Görselleştirme'
    )
    
    parser.add_argument(
        '--skip-preprocessing', action='store_true',
        help='Veri önişleme adımını atla (veriler zaten işlenmişse)'
    )
    
    parser.add_argument(
        '--skip-training', action='store_true',
        help='Model eğitim adımını atla (modeller zaten eğitilmişse)'
    )
    
    parser.add_argument(
        '--skip-visualization', action='store_true',
        help='Görselleştirme adımını atla'
    )
    
    return parser

def ensure_directories():
    """Gerekli dizinlerin var olduğunu kontrol et"""
    directories = ['data', 'models', 'results']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"'{directory}' dizini oluşturuldu.")

def run_step(step_name, skip_flag, module_name, function_name):
    """Belirtilen adımı çalıştır"""
    if skip_flag:
        print(f"{step_name} adımı atlanıyor...")
        return
    
    print(f"\n{step_name} adımı başlatılıyor...")
    progress_bar = tqdm(total=100, desc=f"{step_name}", ncols=100)
    
    try:
        # Modülü dinamik olarak içe aktar.
        module = __import__(module_name)
        # İşlevi çağır.
        getattr(module, function_name)()
        
        progress_bar.update(100)
        print(f"{step_name} adımı tamamlandı.")
    except Exception as e:
        progress_bar.update(100)
        print(f"{step_name} adımı sırasında hata oluştu: {e}")
        sys.exit(1)
    finally:
        progress_bar.close()

def main():
    """Ana program"""
    # Başlangıç zamanını kaydet.
    start_time = time.time()
    
    # Komut satırı argümanlarını ayrıştır.
    parser = create_parser()
    args = parser.parse_args()
    
    # Gerekli dizinleri oluştur.
    ensure_directories()
    
    # Adımları sırasıyla çalıştır.
    steps = [
        ("Veri Önişleme", args.skip_preprocessing, "data_preprocessing", "preprocess_data"),
        ("Model Eğitimi", args.skip_training, "model_training", "train_models"),
        ("Görselleştirme", args.skip_visualization, "visualization", "create_visualizations")
    ]
    
    for step_name, skip_flag, module_name, function_name in steps:
        run_step(step_name, skip_flag, module_name, function_name)
    
    # Toplam süreyi hesapla.
    total_time = time.time() - start_time
    print(f"\nProgram {total_time:.2f} saniyede tamamlandı.")
    print("Sonuçlar 'results' klasöründe bulunabilir.")

if __name__ == "__main__":
    print("IMDb Film İnceleme Verileri ile Duygu Analizi ve Görselleştirme")
    print("=" * 70)
    main() 