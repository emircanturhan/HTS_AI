# Adım 1: Kararlı ve temiz bir başlangıç
FROM python:3.11-slim-bookworm

# Adım 2: Çalışma dizini
WORKDIR /app

# Adım 3: Gerekli minimal sistem paketleri (Artık TA-Lib için derleme araçlarına gerek yok!)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Adım 4: Python'a 'src' klasörünü bulmasını söyleyen adres etiketi
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Adım 5: Python kütüphanelerini kur (Ağ hatalarına karşı dayanıklı)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 -r requirements.txt

# Adım 6: NLTK ve Spacy verilerini indir
RUN python -m nltk.downloader punkt stopwords vader_lexicon && \
    python -m spacy download en_core_web_lg

# Adım 7: Uygulama kodunu kopyala
COPY . .

# Adım 8: Gerekli dizinleri oluştur
RUN mkdir -p data logs models web

# Adım 9: Dashboard için portu aç
EXPOSE 8080

# Adım 10: Uygulamayı başlat
CMD ["python", "main.py"]