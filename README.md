# Crypto Market Sentiment Analysis

Sosyal medyadaki kripto duyarlılığının fiyat hareketlerine etkisini araştıran proje. Klasik soruyu istatistik diliyle sormaya çalıştım: *"Twitter ve Reddit'te bir coin hakkında pozitif tweet'ler arttığında fiyatı yükseliyor mu — yoksa fiyat yükseldikten sonra mı insanlar pozitif tweet atıyor?"*

## Bu proje neden farklı?

Diğer projelerimden farklı olarak burada **iki ayrı veri kaynağı** birleştiriliyor:
- Zaman serisi fiyat verisi (klasik finansal analiz)
- Doğal dil işleme ile elde edilen sentiment skorları (NLP)

İki tarafı birleştirmek için **Granger nedensellik testi** kullandım — "A, B'yi tahmin etmeye yarıyor mu?" sorusunu istatistiksel olarak cevaplayan bir yöntem. Klasik korelasyondan farkı, **zaman boyutunu** hesaba katması.

## Sorduğum sorular

1. Sentiment skorları ile günlük getiriler arasında lineer korelasyon var mı?
2. Sentiment **fiyatın önünde mi gidiyor** yoksa **arkasından mı**?
3. "Extreme Fear" / "Extreme Greed" dönemleri kâr fırsatı mı yoksa tuzak mı?
4. Coin'ler arası sentiment ne kadar bağımlı? BTC'deki panik diğerlerine bulaşıyor mu?
5. VADER duygu analizi kripto jargonuyla başa çıkabiliyor mu?

## Bulgular

- BTC sentiment ile günlük getiri arasında **lag = 0** korelasyon zayıf (~0.10), ama **lag = 1-2 gün** olduğunda biraz artıyor → zayıf "leading" etki
- Granger nedensellik testleri **istatistiksel olarak anlamlı değil** (p > 0.05) — yani sentiment, fiyat hareketlerini güvenle yordayamıyor
- "Extreme Fear" sonrasındaki 7 gün, ortalama olarak hafif pozitif getiri getiriyor — kontrarian strateji kısmen çalışıyor
- Coin'ler arası sentiment korelasyonu çok düşük (genelde < 0.10) — yani BTC paniğinin ETH'ye direkt yayıldığı söylenemiyor (en azından sentiment seviyesinde)
- VADER, **"moon", "diamond hands", "rugged"** gibi kripto jargonunu yanlış sınıflandırıyor → custom sözlük şart

## Yöntem

### 1. Veri (`generate_data.py`)
- 4 coin (BTC, ETH, SOL, DOGE) × 365 günlük OHLCV
- Coin başına günlük ~80 sentiment-skorlu sosyal medya postu
- Fear & Greed Index zaman serisi

### 2. Fiyat analizi (`price_analysis.py`)
- Günlük getiriler, oynaklık (rolling std)
- Coin'ler arası korelasyon ısı haritası

### 3. Sentiment (`sentiment.py`)
- VADER ile compound skor
- Coin × gün × kaynak (Twitter/Reddit/Telegram) bazında agregasyon

### 4. Korelasyon ve Granger (`correlation_study.py`)
- Stasyonarlik testi (Augmented Dickey-Fuller)
- Lagged korelasyon: -7 ile +7 gün arasında
- Granger nedensellik: maxlag = 5

## Kullandığım araçlar

- pandas, numpy
- **statsmodels** (Granger, ADF testi)
- **NLTK + VADER** (sentiment analizi)
- matplotlib, seaborn

## Çalıştırmak için

```bash
pip install -r requirements.txt
python src/generate_data.py
python src/price_analysis.py
python src/sentiment.py
python src/correlation_study.py
```

## Gerçek veriyle çalışmak

Bu projenin yapısı gerçek API'lerle drop-in çalışacak şekilde:

```python
# Fiyatlar
import yfinance as yf
btc = yf.download("BTC-USD", start="2023-01-01")

# Sosyal medya (Twitter API v2 veya Reddit API)
# - tweepy
# - praw (Python Reddit API Wrapper)

# Fear & Greed
# https://api.alternative.me/fng/  (ücretsiz, key gerektirmiyor)
```

## Sınırlamalar (önemli)

Bu sonuçları **alım/satım tavsiyesi olarak alma**. Birkaç nedeni var:

- **Sentetik veri**: Gerçek piyasada sentiment-fiyat ilişkisi daha karmaşık (whale wallet'lar, makroekonomik haberler, türev piyasası)
- **Geri görüş yanılgısı**: Kontrarian stratejinin "extreme fear sonrası al" şeklinde işlemesi geçmiş veride güzel görünür ama gelecek için garanti değil
- **VADER sınırlamaları**: Kripto jargonuna eğitilmemiş, "moon" gibi pozitif kelimeleri yanlış sınıflandırabilir
- **Granger nedensellik = "neden" değildir**: Sadece "A, B'yi tahmin etmeye yarıyor mu?" sorusunu cevaplar. Gerçek nedensellik için randomize deney gerekir, finansal piyasada ise bu imkansız

## Not

Bu projede en şaşırtıcı sonuç, **Granger testlerinin anlamlı çıkmaması** oldu. Başta "sentiment fiyatı yordar" hipotezimi doğrulayacağıma emindim — sosyal medya bu kadar etkili olduktan sonra elbette yordar diye düşündüm. Veriyi incelediğimde gördüm ki:

1. Sentiment ve fiyat **aynı anda** hareket ediyor (lag = 0)
2. Yani biri diğerine önce gelmiyor → **iki yönlü etki** muhtemel

Bu, beklediğimin tersi bir sonuç ama daha **gerçekçi**. Veri bilimi her zaman hipotezini doğrulamaz — bazen çürütür ve sen de buna dürüstçe sahip çıkarsın. "Sentiment fiyatı yordar" demek satışsal olarak iyi başlık ama veri böyle söylemiyor.

Bu projede en çok zorlandığım kısım **statsmodels'ın Granger output'unu** doğru yorumlamak oldu. P-value'lar her lag için ayrı ve "hangi lag istatistiksel olarak en güvenilir" sorusunu cevaplamak için biraz okumam gerekti.

## Author

Nisa Kaya — [github.com/nisakayaa](https://github.com/nisakayaa)
