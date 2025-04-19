import numpy as np
import pandas as pd
import pickle
from time import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data():
    """Önişlenmiş verileri yükle"""
    try:
        X_train = np.load('data/X_train.npy')
        X_test = np.load('data/X_test.npy')
        y_train = np.load('data/y_train.npy')
        y_test = np.load('data/y_test.npy')
        
        print("Veriler başarıyla yüklendi.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Veri yüklenirken hata oluştu: {e}")
        return None, None, None, None

def train_logistic_regression(X_train, y_train):
    """Lojistik regresyon modeli eğitimi"""
    print("Lojistik Regresyon modeli eğitiliyor...")
    start_time = time()
    
    # Model oluştur ve eğit.
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time() - start_time
    print(f"Eğitim tamamlandı. Süre: {training_time:.2f} saniye")
    
    return model

def train_naive_bayes(X_train, y_train):
    """Naive Bayes modeli eğitimi"""
    print("Naive Bayes modeli eğitiliyor...")
    start_time = time()
    
    # Model oluştur ve eğit.
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    training_time = time() - start_time
    print(f"Eğitim tamamlandı. Süre: {training_time:.2f} saniye")
    
    return model

def train_random_forest(X_train, y_train):
    """Random Forest modeli eğitimi"""
    print("Random Forest modeli eğitiliyor...")
    start_time = time()
    
    # Model oluştur ve eğit.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time() - start_time
    print(f"Eğitim tamamlandı. Süre: {training_time:.2f} saniye")
    
    return model

def evaluate_model(model, X_test, y_test, model_name):
    """Model performansını değerlendir"""
    # Tahminleri al.
    y_pred = model.predict(X_test)
    
    # Doğruluk hesapla.
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} Doğruluk: {accuracy:.4f}")
    
    # Sınıflandırma raporu
    report = classification_report(y_test, y_pred, target_names=['Negatif', 'Pozitif'])
    print(f"Sınıflandırma Raporu:\n{report}")
    
    # Karmaşıklık Matrisi
    cm = confusion_matrix(y_test, y_pred)
    
    # Sonuçları dictionary olarak döndür.
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results

def save_model(model, model_name):
    """Modeli kaydet"""
    try:
        with open(f'models/{model_name}.pkl', 'wb') as f:
            pickle.dump(model, f)
        print(f"{model_name} modeli başarıyla kaydedildi.")
    except Exception as e:
        print(f"Model kaydedilirken hata oluştu: {e}")

def train_models():
    """Tüm modelleri eğit ve değerlendir"""
    # Verileri yükle.
    X_train, X_test, y_train, y_test = load_data()
    
    if X_train is None:
        return None
    
    # Tüm modeller için sonuçları sakla.
    all_results = {}
    
    # Lojistik Regresyon
    lr_model = train_logistic_regression(X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    save_model(lr_model, "logistic_regression")
    all_results['logistic_regression'] = lr_results
    
    # Naive Bayes
    nb_model = train_naive_bayes(X_train, y_train)
    nb_results = evaluate_model(nb_model, X_test, y_test, "Naive Bayes")
    save_model(nb_model, "naive_bayes")
    all_results['naive_bayes'] = nb_results
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    save_model(rf_model, "random_forest")
    all_results['random_forest'] = rf_results
    
    # En iyi modeli belirle (doğruluk skoruna göre)
    best_model_name = max(all_results, key=lambda k: all_results[k]['accuracy'])
    print(f"\nEn iyi model: {all_results[best_model_name]['model_name']} "
          f"(Doğruluk: {all_results[best_model_name]['accuracy']:.4f})")
    
    # Sonuçları kaydet.
    with open('models/model_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    
    return all_results

if __name__ == "__main__":
    train_models() 