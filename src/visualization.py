import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from collections import Counter

# Görsel stilleri ayarla.
plt.style.use('seaborn-v0_8-whitegrid')
sns.set(font_scale=1.2)
COLORS = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

def load_results():
    """Model sonuçlarını yükle"""
    try:
        with open('models/model_results.pkl', 'rb') as f:
            results = pickle.load(f)
        return results
    except Exception as e:
        print(f"Sonuçlar yüklenirken hata oluştu: {e}")
        return None

def load_processed_data():
    """İşlenmiş veriyi yükle"""
    try:
        df = pd.read_csv('data/processed_reviews.csv')
        return df
    except Exception as e:
        print(f"İşlenmiş veri yüklenirken hata oluştu: {e}")
        return None

def ensure_directory_exists():
    """Sonuçlar dizininin var olduğunu kontrol et"""
    if not os.path.exists('results'):
        os.makedirs('results')

def plot_sentiment_distribution(df):
    """Duygu dağılımını pasta grafiği olarak görselleştir"""
    plt.figure(figsize=(10, 6))
    
    # Duygu dağılımı
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Duygu', 'Sayı']
    sentiment_counts['Duygu'] = sentiment_counts['Duygu'].map({1: 'Pozitif', 0: 'Negatif'})
    
    # Pasta grafiği
    plt.pie(sentiment_counts['Sayı'], labels=sentiment_counts['Duygu'],
            autopct='%1.1f%%', startangle=90, colors=COLORS,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    
    plt.title('Film Yorumlarında Duygu Dağılımı', fontsize=15)
    plt.tight_layout()
    plt.savefig('results/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Duygu dağılımı pasta grafiği kaydedildi.")

def create_wordcloud(df):
    """Kelime bulutu oluştur"""
    # Pozitif ve negatif yorumları ayır
    positive_text = ' '.join(df[df['sentiment'] == 1]['processed_text'])
    negative_text = ' '.join(df[df['sentiment'] == 0]['processed_text'])
    
    # Pozitif kelime bulutu
    plt.figure(figsize=(12, 6))
    
    wc_positive = WordCloud(width=800, height=400, background_color='white',
                            max_words=200, colormap='YlGn',
                            contour_width=1, contour_color='steelblue')
    wc_positive.generate(positive_text)
    
    plt.subplot(1, 2, 1)
    plt.imshow(wc_positive, interpolation='bilinear')
    plt.title('Pozitif Yorumlarda Sık Kullanılan Kelimeler', fontsize=15)
    plt.axis('off')
    
    # Negatif kelime bulutu
    wc_negative = WordCloud(width=800, height=400, background_color='white',
                           max_words=200, colormap='RdPu',
                           contour_width=1, contour_color='firebrick')
    wc_negative.generate(negative_text)
    
    plt.subplot(1, 2, 2)
    plt.imshow(wc_negative, interpolation='bilinear')
    plt.title('Negatif Yorumlarda Sık Kullanılan Kelimeler', fontsize=15)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/word_clouds.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Kelime bulutu görselleştirmesi kaydedildi.")

def plot_common_words(df):
    """En yaygın kelimelerin çubuk grafiğini oluştur"""
    # Pozitif ve negatif yorumları ayır
    positive_words = ' '.join(df[df['sentiment'] == 1]['processed_text']).split()
    negative_words = ' '.join(df[df['sentiment'] == 0]['processed_text']).split()
    
    # En yaygın kelimeleri say
    pos_word_counts = Counter(positive_words).most_common(15)
    neg_word_counts = Counter(negative_words).most_common(15)
    
    plt.figure(figsize=(15, 10))
    
    # Pozitif kelimeler için çubuk grafiği
    plt.subplot(2, 1, 1)
    sns.barplot(x=[word for word, count in pos_word_counts],
                y=[count for word, count in pos_word_counts],
                palette='YlGn')
    plt.title('Pozitif Yorumlarda En Sık Kullanılan 15 Kelime', fontsize=15)
    plt.xlabel('Kelime')
    plt.ylabel('Frekans')
    plt.xticks(rotation=45)
    
    # Negatif kelimeler için çubuk grafiği
    plt.subplot(2, 1, 2)
    sns.barplot(x=[word for word, count in neg_word_counts],
                y=[count for word, count in neg_word_counts],
                palette='RdPu')
    plt.title('Negatif Yorumlarda En Sık Kullanılan 15 Kelime', fontsize=15)
    plt.xlabel('Kelime')
    plt.ylabel('Frekans')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/common_words.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("En sık kullanılan kelimeler grafiği kaydedildi.")

def plot_model_comparison(results):
    """Model performanslarını karşılaştır"""
    plt.figure(figsize=(12, 6))
    
    # Doğruluk değerlerini çıkar.
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Çubuk grafiği
    sns.barplot(x=[results[model]['model_name'] for model in models],
                y=accuracies, palette=COLORS[:len(models)])
    
    # Doğruluk değerlerini ekle.
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=11)
    
    plt.title('Model Doğruluk Karşılaştırması', fontsize=15)
    plt.xlabel('Model')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Model karşılaştırma grafiği kaydedildi.")

def plot_confusion_matrices(results):
    """Tüm modeller için karmaşıklık matrislerini çiz"""
    plt.figure(figsize=(15, 5 * (len(results) // 2 + len(results) % 2)))
    
    for i, model_name in enumerate(results.keys()):
        cm = results[model_name]['confusion_matrix']
        
        plt.subplot(len(results) // 2 + len(results) % 2, 2, i + 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negatif', 'Pozitif'],
                    yticklabels=['Negatif', 'Pozitif'])
        
        plt.title(f'{results[model_name]["model_name"]} Karmaşıklık Matrisi', fontsize=13)
        plt.xlabel('Tahmin Edilen Etiket')
        plt.ylabel('Gerçek Etiket')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Karmaşıklık matrisleri görselleştirmesi kaydedildi.")

def create_visualizations():
    """Tüm görselleştirmeleri oluştur"""
    # Dizin kontrolü
    ensure_directory_exists()
    
    # Verileri ve sonuçları yükle.
    results = load_results()
    df = load_processed_data()
    
    if df is None or results is None:
        print("Görselleştirme oluşturulamadı. Veriler yüklenemedi.")
        return
    
    # Görselleştirmeleri oluştur.
    plot_sentiment_distribution(df)
    create_wordcloud(df)
    plot_common_words(df)
    plot_model_comparison(results)
    plot_confusion_matrices(results)
    
    print("Tüm görselleştirmeler başarıyla oluşturuldu ve kaydedildi.")

if __name__ == "__main__":
    create_visualizations() 