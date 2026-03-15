# Akademik Literatür Tarama Otonom Ajanı

Python, LangChain ve LLM API kullanarak geliştirilmiş, akademik literatür taraması yapan otonom bir yapay zeka ajanıdır.

## Özellikler

- **Literatür Taraması**: Verilen araştırma konusu için son 5 yılın akademik makalelerini tarar
- **En Çok Atıf Alan Makaleler**: Alandaki en etkili çalışmaları atıf sayısına göre sıralar
- **Public Veri Setleri**: Konuyla ilgili açık veri setleri ve benchmark'ları bulur
- **Özgün Değer Analizi**: Literatürdeki boşlukları belirler ve potansiyel katkı alanları önerir
- **Rapor Oluşturma**: Sonuçları yapılandırılmış Markdown raporu olarak kaydeder

## Mimari

```
main.py                  # CLI giriş noktası (tek konu / interaktif mod)
src/
├── agent.py             # LangChain agent ve tool tanımları
└── scholar_client.py    # Semantic Scholar API istemcisi
output/                  # Oluşturulan raporlar
```

**Veri Kaynağı**: [Semantic Scholar Academic Graph API](https://api.semanticscholar.org/) (ücretsiz, API key gerektirmez)

**LLM**: OpenAI GPT-4o-mini (varsayılan) - LangChain üzerinden

## Kurulum

```bash
# Bağımlılıkları yükleyin
pip install -r requirements.txt

# .env dosyasını oluşturun
cp .env.example .env
# .env dosyasına OpenAI API anahtarınızı ekleyin
```

## Kullanım

### Tek Konu ile Çalıştırma

```bash
python main.py "transformers in natural language processing"
```

### İnteraktif Mod

```bash
python main.py
# Araştırma konusu girin: large language models for code generation
```

### Farklı Model Kullanımı

`.env` dosyasında `LLM_MODEL` değişkenini ayarlayabilirsiniz:

```
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o
```

## Çıktı

Ajan her tarama sonucunda `output/` klasörüne bir Markdown raporu kaydeder. Rapor şu bölümleri içerir:

1. **Genel Literatür Özeti** - Alanın genel durumu, ana temalar ve trendler
2. **En Çok Atıf Alan Makaleler** - En etkili çalışmalar ve katkıları
3. **Public Veri Setleri ve Benchmark'lar** - Kullanılabilir açık veri kaynakları
4. **Araştırma Boşlukları ve Özgün Değer Önerileri** - Potansiyel katkı alanları

## Teknolojiler

- **Python 3.10+**
- **LangChain** - Agent framework
- **OpenAI API** - LLM (GPT-4o-mini / GPT-4o)
- **Semantic Scholar API** - Akademik makale veritabanı
