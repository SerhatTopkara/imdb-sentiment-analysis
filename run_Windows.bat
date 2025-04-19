@echo off
echo ============================================
echo IMDb Film İnceleme Verileri ile Duygu Analizi ve Görselleştirme
echo ============================================

:: Python kontrolü
where python >nul 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo Python yüklü değil. Lütfen https://www.python.org/downloads/ adresinden yükleyin.
    pause
    exit /b
)

:: Sanal ortam kontrolü
IF NOT EXIST "venv" (
    echo Sanal ortam olusturuluyor...
    python -m venv venv
)

:: Sanal ortamı etkinleştir
call venv\Scripts\activate.bat

:: pip güncelle
echo pip guncelleniyor...
python -m pip install --upgrade pip

:: Gerekli kutuphaneler
echo Gerekli kutuphaneler yukleniyor...
pip install -r requirements.txt

:: NLTK verilerini indir
echo NLTK verileri indiriliyor...
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'wordnet', 'movie_reviews', 'punkt_tab'])"

:: Ana betik
echo Program baslatiliyor...
python src\main.py %*

:: Sanal ortamdan cik
deactivate
