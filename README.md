# IMDb Film İnceleme Verileri ile Duygu Analizi ve Görselleştirme

Bu proje, IMDb film yorumlarını kullanarak makine öğrenmesi ve doğal dil işleme (NLP) teknikleriyle duygu analizi yapmayı amaçlamaktadır. Yorumların olumlu mu yoksa olumsuz mu olduğu belirlenir ve sonuçlar görsel olarak sunulur.

## Proje Açıklaması

Bu projede:
- IMDb film inceleme veri seti üzerinde veri temizleme işlemleri
- TF-IDF kullanarak metin verilerinin sayısallaştırılması
- Makine öğrenmesi algoritmaları ile duygu analizi modeli eğitimi
- Sonuçların görselleştirilmesi

## Özellikler

- Veri önişleme ve temizleme
- Çoklu model karşılaştırması (Lojistik Regresyon, Naive Bayes, Random Forest)
- Kapsamlı sonuç görselleştirmesi
- Etkileşimli komut satırı arayüzü
- Modüler kod yapısı

## Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/SerhatTopkara/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. Gerekli paketleri yükleyin:
```bash
chmod +x run.sh
./run.sh
```

## Kullanım

Projeyi çalıştırın:
```bash
./run.sh
```
## Windows Kullanıcıları İçin:

Aşağıdaki komutu çalıştırarak projeyi başlatabilirsiniz:

```bat
run_Windows.bat

İsteğe bağlı argümanlar:
```bash
./run.sh --skip-preprocessing  # Veri önişleme adımını atla
./run.sh --skip-training       # Model eğitimi adımını atla
./run.sh --skip-visualization  # Görselleştirme adımını atla
```

## Proje Yapısı

- `src/`: Kaynak kodları
  - `data_preprocessing.py`: Veri temizleme ve ön işleme işlevleri
  - `model_training.py`: Model eğitimi ve değerlendirme
  - `visualization.py`: Sonuçların görselleştirilmesi
  - `main.py`: Ana program dosyası
- `data/`: Veri seti ve işlenmiş veriler
- `models/`: Eğitilmiş modeller
- `results/`: Görselleştirmeler ve analiz sonuçları

## Sonuçlar

Farklı algoritmaların performansı:
- Lojistik Regresyon: Doğruluk = ~%82.25
- Naive Bayes: Doğruluk = ~%80.75
- Random Forest: Doğruluk = ~%79.50

## Görselleştirmeler

Proje sonuçları `results/` klasöründe görselleştirilmiştir:
- Olumlu/Olumsuz yorum dağılımı pasta grafiği
- En çok geçen kelimeler için kelime bulutu
- Model karşılaştırma çubuk grafiği
- Her model için karmaşıklık matrisi




