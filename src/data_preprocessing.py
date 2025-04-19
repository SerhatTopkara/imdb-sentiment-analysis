import re
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# NLTK kaynaklarını indir.
def download_nltk_resources():
    """NLTK için gerekli kaynakları indirme"""
    try:
        resources = ['punkt', 'stopwords', 'wordnet', 'movie_reviews']
        for resource in resources:
            nltk.download(resource, quiet=True)
        print("NLTK kaynakları başarıyla indirildi.")
    except Exception as e:
        print(f"NLTK kaynaklarını indirirken hata oluştu: {e}")

# Veri setini yükle.
def load_data():
    """NLTK'dan film yorumlarını yükle"""
    from nltk.corpus import movie_reviews
    
    reviews = []
    labels = []
    
    for category in ['pos', 'neg']:
        for fileid in movie_reviews.fileids(category):
            # Tüm dosya içeriğini birleştir
            text = ' '.join(movie_reviews.words(fileid))
            reviews.append(text)
            # Pozitif yorumlar için 1, negatif yorumlar için 0
            labels.append(1 if category == 'pos' else 0)
    
    # DataFrame oluştur.
    df = pd.DataFrame({
        'text': reviews,
        'sentiment': labels
    })
    
    print(f"Yüklenen veri seti boyutu: {df.shape}")
    return df

# Metin temizleme işlevi.
def clean_text(text):
    """Metni temizle: küçük harfe çevir, alfanumerik olmayan karakterleri kaldır"""
    # Küçük harfe çevir.
    text = text.lower()
    # HTML etiketlerini kaldır.
    text = re.sub(r'<.*?>', '', text)
    # Alfanumerik olmayan karakterleri kaldır (boşluklar hariç)
    text = re.sub(r'[^\w\s]', '', text)
    # Tek karakterli kelimeleri kaldır.
    text = re.sub(r'\b\w\b', '', text)
    # Fazla boşlukları kaldır.
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Stopwords kaldırma ve lemmatization işlemi
def preprocess_text(text):
    """Stopwords kaldır ve lemmatization uygula"""
    # İngilizce stopwords listesi
    stop_words = set(stopwords.words('english'))
    # Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Tokenize et.
    tokens = word_tokenize(text)
    # Stopwords olmayan ve alfanumerik olan tokenleri filtrele.
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                       if token not in stop_words and token.isalnum()]
    
    return ' '.join(filtered_tokens)

# Veri setini hazırlama
def prepare_dataset(df):
    """Veri setini temizle ve önişle"""
    # Metinleri temizle.
    df['cleaned_text'] = df['text'].apply(clean_text)
    # Önişleme uygula.
    df['processed_text'] = df['cleaned_text'].apply(preprocess_text)
    
    return df

# TF-IDF vektörizasyonu
def vectorize_text(df, max_features=5000):
    """Metinleri TF-IDF kullanarak vektörlere dönüştür"""
    # TF-IDF Vektörleme
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    X = tfidf_vectorizer.fit_transform(df['processed_text'])
    y = df['sentiment']
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, tfidf_vectorizer

# Ana işlev
def preprocess_data():
    """Veri ön işleme ve hazırlama işlemini gerçekleştirir"""
    # NLTK kaynaklarını indir.
    download_nltk_resources()
    
    # Veriyi yükle.
    df = load_data()
    
    # Veriyi temizle ve önişle.
    df = prepare_dataset(df)
    
    # Veriyi vektörize et.
    X_train, X_test, y_train, y_test, vectorizer = vectorize_text(df)
    
    # İşlenen veriyi kaydet.
    np.save('data/X_train.npy', X_train.toarray())
    np.save('data/X_test.npy', X_test.toarray())
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)
    
    # Vektörleştiriciyi kaydet.
    import pickle
    with open('models/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Analiz için işlenmiş veriyi kaydet.
    df.to_csv('data/processed_reviews.csv', index=False)
    
    print("Veri önişleme tamamlandı ve dosyalar kaydedildi.")
    
    return X_train, X_test, y_train, y_test, vectorizer, df

if __name__ == "__main__":
    preprocess_data() 