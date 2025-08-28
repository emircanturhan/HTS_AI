FROM python:3.11-slim-bookworm AS builder

# Gerekli tüm inşa araçlarını kur
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# TA-Lib C kütüphanesini indir ve kur
RUN TA_LIB_VERSION=0.4.0 && \
    wget "http://prdownloads.sourceforge.net/ta-lib/ta-lib-${TA_LIB_VERSION}-src.tar.gz" -O ta-lib.tar.gz && \
    tar -xzf ta-lib.tar.gz && \
    rm ta-lib.tar.gz && \
    cd "ta-lib/" && \
    curl -L 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD' -o './config.guess' && \
    curl -L 'https://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD' -o './config.sub' && \
    chmod +x ./config.guess ./config.sub && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf "ta-lib/"

# Şimdi, Python paketlerini STRATEJİK SIRAYLA paketle (.whl oluştur)
COPY requirements.txt .
# ÖNCE: Sadece numpy'ı kur. Bu, TA-Lib'in doğru versiyonu kullanmasını garantiler.
RUN pip wheel --no-cache-dir --wheel-dir=/wheels numpy==1.26.2
# SONRA: Sadece TA-Lib'i, az önce kurulan numpy ile derle.
RUN pip wheel --no-cache-dir --wheel-dir=/wheels TA-Lib==0.4.28
# EN SON: Geri kalan her şeyi paketle.
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt


FROM python:3.11-slim-bookworm

WORKDIR /app

# Sadece TA-Lib'in çalışması için gereken C kütüphanesini kopyala
COPY --from=builder /usr/lib/libta_lib.so.0 /usr/lib/libta_lib.so.0
COPY --from=builder /usr/lib/libta_lib.so.0.0.0 /usr/lib/libta_lib.so.0.0.0

# Önceki aşamada inşa ettiğimiz ".whl" paketlerini ve requirements dosyasını kopyala
COPY --from=builder /wheels /wheels
COPY requirements.txt .

# Şimdi, derleme yapmak yerine, hazır inşa edilmiş paketlerimizi kur
RUN pip install --no-cache-dir --find-links=/wheels -r requirements.txt

# NLTK verilerini indir
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('vader_lexicon', quiet=True)"

# Uygulama kodunun geri kalanını kopyala
COPY . .

# Dashboard için portu aç
EXPOSE 8080

# Uygulamayı başlat
CMD ["python", "main.py"]