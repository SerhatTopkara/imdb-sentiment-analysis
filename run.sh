#!/bin/bash

# IMDb Film İnceleme Verileri ile Duygu Analizi ve Görselleştirme
# Çalıştırma Betiği

echo "IMDb Film İnceleme Verileri ile Duygu Analizi ve Görselleştirme"
echo "==============================================================="

# Gerekli sistem paketlerini kontrol et ve yükle
if ! command -v python3 &> /dev/null; then
    echo "Python3 yüklü değil. Yükleniyor..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv python3-full python3-dev
fi

# Sanal ortam kontrolü ve etkinleştirme
if [ ! -d "venv" ]; then
    echo "Sanal ortam oluşturuluyor..."
    python3 -m venv venv
fi

# Sanal ortamı etkinleştir
source venv/bin/activate

# pip'i güncelle
echo "pip güncelleniyor..."
python3 -m pip install --upgrade pip

# Gerekli kütüphaneleri kur
echo "Gerekli kütüphaneleri kontrol ediliyor ve kuruluyor..."
python3 -m pip install -r requirements.txt

# NLTK verilerini indir
echo "NLTK verileri indiriliyor..."
python3 -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'movie_reviews', 'punkt_tab'])"

# Proje betiğini çalıştır
echo "Program başlatılıyor..."
python3 src/main.py "$@"

# Sanal ortamdan çık
deactivate 