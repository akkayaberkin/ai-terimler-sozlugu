# Gün 7: MLOps & Deployment

Bir modeli eğitmek işin sadece başlangıcı. Asıl maraton, o modeli gerçek dünyada kesintisiz, hızlı ve güvenilir şekilde çalıştırmak. Bu bölümde modelleri ürüne dönüştüren altyapıyı, ölçeklendirme stratejilerini ve canlı sistem yönetimini anlatıyorum.

---

### Model Deployment (Model Dağıtımı)
**Ne işe yarar:** Eğitilmiş bir modeli, gerçek kullanıcıların erişebileceği bir sunucu veya uç cihazda çalışır hale getirme süreci.

**Detaylı:** Jupyter Notebook'ta çalışan bir modelin kimseye faydası yok. Deployment, o modeli alıp bir API endpoint'i, bir mobil uygulama veya bir web servisi olarak ayağa kaldırmaktır. Dört ana deployment şekli var: 1) Batch inference — toplu veriyi bir kerede işle, sonuçları kaydet (günlük raporlar için ideal). 2) Real-time / online inference — her istek anında işlenir, milisaniyeler içinde cevap döner (chatbot, öneri sistemi). 3) Edge deployment — model cihazın kendisinde çalışır, internete gerek yok (telefon, IoT). 4) Serverless — altyapıyla uğraşmadan, sadece modeli gönderirsiniz, bulut otomatik ölçeklendirir (AWS SageMaker, GCP Vertex AI). Hangi yöntemi seçeceğiniz tamamen gecikme toleransına, veri hacmine ve maliyete bağlı. Konuşma gibi gerçek zamanlı bir uygulama yapıyorsanız online inference şart. Ama gece çalışacak bir onboarding modeliniz varsa batch inference yeter.

**Örnek:** FastAPI ile model dağıtımı:

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI()

class Metin(BaseModel):
    icerik: str

# Modeli yükle (uygulama başlangıcında bir kere)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
model = AutoModelForSequenceClassification.from_pretrained("./egitilmis-model")
model.eval()

@app.post("/tahmin")
async def tahmin_et(veri: Metin):
    inputs = tokenizer(veri.icerik, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    tahmin = outputs.logits.argmax().item()
    return {"tahmin": tahmin, "metin": veri.icerik}

@app.get("/saglik")
async def saglik_kontrol():
    return {"durum": "sağlıklı", "model": "bert-turkish-sentiment"}
```

**İlişkili:** Model Serving, Inference, Latency, Batching, Model Registry

---

### Inference (Çıkarım)
**Ne işe yarar:** Eğitilmiş bir modelin, verilen bir girdiye (prompt, görsel, sayısal veri) karşılık tahmin veya çıktı üretme işlemi.

**Detaylı:** Training bir modelin öğrenme aşamasıysa, inference o bildiklerini kullanma aşamasıdır. Matematiksel olarak inference, modelin ileri yayılım (forward pass) hesaplarını yapmasıdır — geri yayılım (backward pass) yoktur, sadece öğrenilmiş ağırlıklar kullanılır. Inference'ı hızlandırmak için bir sürü teknik var: model quantization (ağırlıkları 32-bit'ten 8-bit'e düşürmek), pruning (gereksiz nöronları kesmek), KV cache (LLM'lerde daha önce hesaplanmış attention değerlerini yeniden kullanmak), ve Flash Attention gibi optimizasyonlu kernel'ler. Bir modelin inference hızı, batch size'dan input uzunluğuna, GPU belleğinden model büyüklüğüne kadar birçok faktöre bağlı. Örneğin GPT-4 gibi bir modelde tek bir inference işlemi trilyonlarca matematiksel işlem gerektirir. Bu yüzden inference'ı optimize etmek, maliyetin yanında kullanıcı deneyimi için de kritik.

**Örnek:** PyTorch'da inference:

```python
import torch
import torch.nn as nn

class BasitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.katman = nn.Linear(10, 2)
        
    def forward(self, x):
        return self.katman(x)

model = BasitModel()
model.load_state_dict(torch.load("model.pt"))

# Inference modu — dropout, batch norm gibi katmanlar devre dışı
model.eval()

girdi = torch.randn(1, 10)
with torch.no_grad():  # gradient hesaplama yok, daha hızlı ve az bellek
    cikti = model(girdi)
    print(cikti)

# LLM inference'ında generation loop
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

inputs = tokenizer("Türkiye'nin başkenti", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0]))
```

**İlişkili:** Throughput (token/s), Latency, KV Cache, Quantization, Model Serving

---

### Throughput (token/s) (Çıktı Hızı — Token/Saniye)
**Ne işe yarar:** Bir modelin saniyede kaç token (veya tahmin) üretebildiğini ölçen performans metriği.

**Detaylı:** Throughput, bir model ne kadar "verimli" çalışıyor sorusunun cevabıdır. LLM'lerde genelde "token/saniye" olarak ölçülür — model bir saniyede kaç tane token üretebiliyor? Bunu hesaplamak için: üretilen toplam token / geçen süre. 70B parametreli bir model tek GPU'da 10-20 token/s civarında çalışırken, quantize edilmiş 7B bir model 100+ token/s'ye ulaşabilir. Throughput'u etkileyen faktörler: GPU tipi ve sayısı, model büyüklüğü (parametre sayısı), batch size, input/output token uzunluğu, quantization seviyesi, ve kullanılan serving framework'ü. Batch size'ı artırmak throughput'u yükseltir (daha fazla paralel işlem) ama tek bir kullanıcının latency'sini de artırır. Bu yüzden throughput ve latency arasında bir trade-off var. Özellikle batch API'lerinde throughput metriği kritiktir — saniyede kaç müşteriye hizmet verebileceğinizi belirler.

**Örnek:** Throughput ölçümü:

```python
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

inputs = tokenizer("Uzun bir hikaye anlat", return_tensors="pt")

baslangic = time.time()
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True)
bitis = time.time()

uretilen_token = outputs.shape[1] - inputs.input_ids.shape[1]
gecen_sure = bitis - baslangic
throughput = uretilen_token / gecen_sure

print(f"Üretilen token: {uretilen_token}")
print(f"Süre: {gecen_sure:.2f}s")
print(f"Throughput: {throughput:.2f} token/s")

# Batch throughput
def batch_throughput(model, tokenizer, promptlar, batch_size=8):
    inputs = tokenizer(promptlar, padding=True, return_tensors="pt")
    baslangic = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50)
    bitis = time.time()
    
    toplam_token = outputs.shape[0] * outputs.shape[1] - inputs.input_ids.shape[1]
    return toplam_token / (bitis - baslangic)

# Tek tek vs batch karşılaştırması
print(f"Batch throughput: {batch_throughput(model, tokenizer, ['Merhaba'] * 8):.2f} token/s")
```

**İlişkili:** Inference, Latency, Batching, Model Serving, GPU Utilization

---

### Latency (Gecikme Süresi)
**Ne işe yarar:** Bir isteğin modele ulaşmasından, modelin cevap üretmesine kadar geçen toplam süre.

**Detaylı:** Latency, kullanıcının "bekleme" süresidir. İki ana bileşeni var: Time-to-First-Token (TTFT) — ilk token ne kadar sürede geldi — ve Inter-Token Latency — sonraki token'lar arasındaki süre. TTFT özellikle sohbet uygulamalarında kritik: kullanıcı "Selam" yazdıktan sonra 3 saniye siyah ekran görürse deneyim berbat olur. İdeal TTFT 500ms altı, kabul edilebilir 1-2 saniye. Inter-token latency ise modelin "akıcı" hissettirip hissettirmediğini belirler — yavaş bir model 5-6 token/s üretir ve kullanıcı harf harf yazıyormuş gibi hisseder. Latency'i düşürmek için: modeli küçültün (distillation), input'u kısaltın, streaming kullanın (token üretildikçe gönderin, toplu beklemeden), GPU'yu soğutun (thermal throttling latency'i artırır) ve doğru serving framework'ünü seçin. vLLM'nin PagedAttention'ı, TTFT'yi önemli ölçüde düşüren bir optimizasyondur.

**Örnek:** Latency ölçümü:

```python
import time
import numpy as np

def olc_latency(model, tokenizer, prompt, olcum_sayisi=10):
    TTFT_sureleri = []
    token_araliklari = []
    
    for _ in range(olcum_sayisi):
        inputs = tokenizer(prompt, return_tensors="pt")
        
        basla = time.perf_counter()
        ilk_token = True
        onceki_zaman = basla
        
        # Token üretme döngüsü
        for token_id in model.generate(**inputs, max_new_tokens=50):
            simdi = time.perf_counter()
            
            if ilk_token:
                TTFT_sureleri.append((simdi - basla) * 1000)  # ms
                ilk_token = False
            else:
                token_araliklari.append((simdi - onceki_zaman) * 1000)
            
            onceki_zaman = simdi
    
    return {
        "TTFT (ms)": f"{np.mean(TTFT_sureleri):.1f} ± {np.std(TTFT_sureleri):.1f}",
        "Token Aralığı (ms)": f"{np.mean(token_araliklari):.1f} ± {np.std(token_araliklari):.1f}",
        "Token/s": f"{1000 / np.mean(token_araliklari):.1f}"
    }

# Percentile metrikleri (P50, P95, P99)
latency_olcumleri = [150, 152, 148, 310, 155, 149, 151, 450, 153]  # ms
print(f"P50: {np.percentile(latency_olcumleri, 50):.0f}ms")
print(f"P95: {np.percentile(latency_olcumleri, 95):.0f}ms")  # outlier'ları gör
print(f"P99: {np.percentile(latency_olcumleri, 99):.0f}ms")
```

**İlişkili:** Inference, Throughput (token/s), Streaming, Model Serving, Batching

---

### Batching (Statik / Dinamik / Continuous) (Toplu İşleme)
**Ne işe yarar:** Birden çok isteği aynı anda işleyerek GPU'nun verimini artıran ve throughput'u yükselten optimizasyon tekniği.

**Detaylı:** GPU'lar paralel işlemcide canavar gibidir. Tek bir istekle GPU'yu kullanmak, Ferrari'yle bakkala gitmek gibidir. Batching, birden çok isteği bir araya toplayıp aynı anda işleterek GPU'nun tüm çekirdeklerini verimli kullanır. Üç çeşit batching stratejisi var. Static batching: istekler toplanır, belli bir sayıya ulaşınca veya süre dolunca toplu işlem başlar. Basit ama verimsizdir çünkü bir hızlı istek, bir yavaş isteği beklemek zorunda kalır. Dynamic batching: timeout mekanizması vardır — "en fazla 200ms bekle, sonra ne varsa gönder". Veya "en fazla 32 istek topla, erken dolarsa hemen gönder". Continuous batching (veya iteration-level batching): en gelişmiş yöntem. Bir batch halindeki istekler aynı anda bitmek zorunda değildir — erken biten istek batch'ten çıkarılır, yeni istek eklenir. Böylece her GPU döngüsünde maksimum doluluk sağlanır. vLLM'nin en büyük yeniliklerinden biri continuous batching desteğidir.

**Örnek:** Batching stratejilerinin karşılaştırması:

```python
import asyncio
import time

# STATIC BATCHING — tüm istekler toplanır, toplu işlenir
class StaticBatching:
    def __init__(self, batch_size=4, max_wait=0.5):
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.kuyruk = []
        self.son_islem = time.time()
    
    async def istek_ekle(self, prompt):
        self.kuyruk.append(prompt)
        
        # Batch dolu veya süre dolduysa işle
        if len(self.kuyruk) >= self.batch_size or \
           (time.time() - self.son_islem) > self.max_wait:
            return await self.isle()
        return None  # Diğer isteklerle birlikte işlenir
    
    async def isle(self):
        if not self.kuyruk:
            return
        batch = self.kuyruk[:]
        self.kuyruk = []
        self.son_islem = time.time()
        
        print(f"[StaticBatch] {len(batch)} istek işleniyor...")
        # Simüle model inference — en yavaş isteğe göre tamamlanır
        await asyncio.sleep(max(len(p) * 0.01 for p in batch))
        return [f"Sonuç_{i}" for i in range(len(batch))]

# CONTINUOUS BATCHING — her adımda biten çıkar, yeni eklenir
class ContinuousBatching:
    def __init__(self, max_concurrent=4):
        self.max_concurrent = max_concurrent
        self.aktif = {}  # {id: (prompt, ilerleme)}
    
    async def istek_ekle(self, prompt):
        istek_id = id(prompt)
        self.aktif[istek_id] = {"prompt": prompt, "ilerleme": 0}
        
        while istek_id in self.aktif:
            # Her döngüde tüm aktif istekleri 1 adım ilerlet
            for uid, istek in list(self.aktif.items()):
                istek["ilerleme"] += 1
                if istek["ilerleme"] >= len(istek["prompt"]):
                    # Bu istek bitti, batch'ten çıkar
                    del self.aktif[uid]
                    return f"Sonuc_{uid}"
            
            await asyncio.sleep(0.01)

# Avantaj: hızlı istekler yavaşları beklemez
```

**İlişkili:** Throughput (token/s), Latency, Model Serving, vLLM, GPU Utilization

---

### Model Serving (vLLM / TGI / SGLang) (Model Sunumu)
**Ne işe yarar:** Büyük dil modellerini verimli şekilde canlıya almak ve API üzerinden sunmak için özel olarak geliştirilmiş framework'ler.

**Detaylı:** Bir modeli raw PyTorch ile deploy ederseniz, hem yavaş olur hem de bellek yönetimi elle yapmanız gerekir. İşte serving framework'leri bu dertleri çözer. vLLM: açık kaynak, en popüler seçenek. PagedAttention ile KV cache yönetimini yapar — tıpkı işletim sistemlerindeki sayfalama (paging) gibi, cache'i küçük bloklara böler ve ihtiyaç duyulmayanları disk'e atar. Continuous batching desteği vardır. TGI (Text Generation Inference): Hugging Face'in framework'ü. HF ekosistemiyle en iyi entegrasyona sahiptir, watermarking ve token streaming gibi özellikler built-in gelir. SGLang en yeni oyunculardan — hem serving hem de dil yapıları (structured generation) konusunda güçlü. JSON Schema, regular expression gibi yapılarla çıktıyı kısıtlamaya izin verir, bu da özellikle function calling'de çok işe yarar. Kısacası: vLLM genel amaçlı en iyisi, TGI HF ekosistemindeyseniz ideal, SGLang structured output'a ihtiyacınız varsa öne çıkar.

**Örnek:** vLLM ile model serving:

```python
# Sunucu tarafı — vLLM'yi ayağa kaldır
# Terminal: vllm serve mistralai/Mistral-7B-v0.1 --port 8000

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="bos"  # vLLM anahtar istemez
)

# OpenAI API uyumlu, direkt kullan
cevap = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[{"role": "user", "content": "Türkiye'de görülmesi gereken 3 yer?"}],
    max_tokens=100,
    temperature=0.7
)

print(cevap.choices[0].message.content)
```

TGI ile:
```python
# Terminal: text-generation-launcher --model-id mistralai/Mistral-7B-v0.1

import requests

response = requests.post(
    "http://localhost:8080/generate",
    json={
        "inputs": "Türkiye'nin başkenti",
        "parameters": {"max_new_tokens": 20}
    }
)
print(response.json()["generated_text"])
```

SGLang ile structured generation:
```python
# Terminal: python -m sglang.launch_server --model mistralai/Mistral-7B-v0.1

import sglang as sgl

@sgl.function
def json_yaz(s, prompt):
    s += prompt
    s += "Şu JSON formatında cevap ver: {'sehir': str, 'nufus': int}\n"
    s += sgl.gen("cevap", max_tokens=100)

state = json_yaz.run(prompt="Ankara hakkında bilgi", stream=False)
print(state["cevap"])
```

**İlişkili:** Model Deployment, Batching, Inference, Throughput (token/s), Latency

---

### A/B Testing (A/B Testi)
**Ne işe yarar:** İki farklı model veya model versiyonunu gerçek kullanıcı trafiğinde karşılaştırarak hangisinin daha iyi olduğunu istatistiksel olarak belirleme yöntemi.

**Detaylı:** A/B testi, "yeni model eski modelden gerçekten daha iyi mi?" sorusunun objektif cevabıdır. Şöyle işler: kullanıcıların bir kısmına A modelini (mevcut/control), bir kısmına B modelini (yeni/treatment) gösterirsiniz. Hangi kullanıcının hangi gruba düştüğü rastgeledir. Ardından bir metrik belirlersiniz — örneğin kullanıcı memnuniyeti, tıklama oranı, dönüşüm oranı, veya hata sayısı. Yeterli veri toplandıktan sonra (genelde binlerce istek), istatistiksel testlerle (t-test, chi-square) iki grup arasında anlamlı fark var mı diye bakarsınız. LLM dünyasında A/B testi yapmak daha karmaşıktır çünkü çıktılar yaratıcıdır ve objektif metrik bulmak zordur. Bazen LLM-as-judge kullanılır — bir üçüncü model iki çıktıyı karşılaştırır. Diğer zorluklar: aynı kullanıcı aynı prompt'u farklı modellerde deneyince tutarsızlık hisseder (consistency issue), ve model çıktıları sabitlenmiş bir doğru cevap olmadığı için metrik tasarımı çetrefillidir.

**Örnek:** A/B test altyapısı:

```python
import random
import hashlib

class ABTest:
    def __init__(self, model_a, model_b, trafik_orani=0.5):
        self.model_a = model_a  # Control (mevcut)
        self.model_b = model_b  # Treatment (yeni)
        self.trafik_orani = trafik_orani
        self.metrikler = {"A": [], "B": []}
    
    def hangi_grup(self, kullanici_id: str) -> str:
        """Kullanıcı ID'sine göre deterministik grup ataması."""
        hash_deger = hashlib.md5(kullanici_id.encode()).hexdigest()
        return "A" if int(hash_deger, 16) / (2**128) > self.trafik_orani else "B"
    
    def tahmin_et(self, kullanici_id: str, prompt: str):
        grup = self.hangi_grup(kullanici_id)
        
        if grup == "A":
            cevap = self.model_a(prompt)
        else:
            cevap = self.model_b(prompt)
        
        return cevap, grup
    
    def metrik_kaydet(self, grup: str, skor: float):
        self.metrikler[grup].append(skor)
    
    def rapor(self):
        import numpy as np
        for grup in ["A", "B"]:
            veri = self.metrikler[grup]
            if veri:
                print(f"Grup {grup}: n={len(veri)}, ortalama={np.mean(veri):.3f}, std={np.std(veri):.3f}")

# Kullanım
test = ABTest(model_a=eski_model, model_b=yeni_model)
cevap, grup = test.tahmin_et("kullanici_123", "Bugün hava nasıl?")

# Kullanıcı beğendi mi diye ölç
kullanici_memnuniyeti = 4.5 / 5.0  # Skorla
test.metrik_kaydet(grup, kullanici_memnuniyeti)
```

Gerçek hayatta LLM-AS-Judge ile:
```python
judge_prompt = """
Aşağıdaki iki çıktıyı karşılaştır. Hangisi daha kaliteli?
- Kapsamlılık
- Doğruluk
- Kullanıcıya uygunluk

A: {model_a_cevap}
B: {model_b_cevap}

Karar: [A daha iyi / B daha iyi / Eşit]
"""
```

**İlişkili:** Canary Deployment, Model Registry, Evaluation, LLM-as-Judge

---

### Canary Deployment (Kanarya Dağıtımı)
**Ne işe yarar:** Yeni bir model versiyonunu önce küçük bir kullanıcı grubuna açarak, olası sorunları büyük kitleye yayılmadan tespit etme stratejisi.

**Detaylı:** İsmi madenlerde kullanılan kanarya kuşundan gelir — maden işçileri zehirli gazı fark etmek için yanlarında kanarya taşırdı, kuş ölünce kaçarlardı. Aynı mantık: yeni modeli %100 kullanıcıya açmadan önce küçük bir kısma verip "patlarsa az kişi etkilensin" dersiniz. Canary deployment şu şekilde işler: yeni modeli alırsınız, trafiğin %1'ini bu modele yönlendirirsiniz. Bir süre (örneğin 1 saat) metrikleri izlersiniz — latency yükseldi mi? Hata oranı arttı mı? Kullanıcı şikayetleri geldi mi? Sorun yoksa %5'e, sonra %20'ye, %50'ye, %100'e çıkartırsınız. Herhangi bir aşamada sorun görürseniz anında geri alırsınız (rollback) ve eski modele dönersiniz. A/B testing'den farkı: Canary deployment bir release stratejisidir, A/B testi ise bir deney. Canary'de amaç tüm kullanıcılara ulaşmaktır; A/B'de amaç karşılaştırma yapmaktır. En sevdiğim özelliklerinden biri: otomatik rollback. Eğer hata oranı eşiği geçerse, deployment pipeline kendiliğinden durur ve eski modele döner.

**Örnek:** Canary deployment mantığı:

```python
import random

class CanaryDeployment:
    def __init__(self, model_eski, model_yeni):
        self.model_eski = model_eski
        self.model_yeni = model_yeni
        self.canary_orani = 0.01  # %1 ile başla
        self.metrikler = {
            "eski": {"hata": 0, "toplam": 0, "latency": []},
            "yeni": {"hata": 0, "toplam": 0, "latency": []}
        }
        self.hata_esigi = 0.05  # %5 hata → rollback
    
    def deploy(self, istek):
        # Canary oranına göre yönlendir
        if random.random() < self.canary_orani:
            model = "yeni"
            try:
                sonuc = self.model_yeni(istek)
                self.metrikler["yeni"]["toplam"] += 1
            except:
                self.metrikler["yeni"]["hata"] += 1
                sonuc = self.model_eski(istek)  # Fallback
        else:
            model = "eski"
            sonuc = self.model_eski(istek)
            self.metrikler["eski"]["toplam"] += 1
        
        return sonuc, model
    
    def kontrol_et(self):
        yeni = self.metrikler["yeni"]
        hata_orani = yeni["hata"] / max(yeni["toplam"], 1)
        
        if hata_orani > self.hata_esigi:
            print(f"🚨 Hata oranı {hata_orani:.2%} — ROLLBACK!")
            self.canary_orani = 0  # Tüm trafiği eski modele yönlendir
            return False
        
        # Her şey iyiyse oranı artır
        if yeni["toplam"] > 100 and hata_orani < 0.01:
            self.canary_orani = min(self.canary_orani * 5, 1.0)
            print(f"✅ Canary oranı arttırıldı: %{self.canary_orani*100:.0f}")
        
        return True
```

Kubernetes ile doğal canary:
```yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: model-servis
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-servis-yeni
  service:
    port: 8080
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
      - name: hata-orani
        thresholdRange:
          max: 5
      - name: latency-p99
        thresholdRange:
          max: 500  # 500ms üzeri alarm
```

**İlişkili:** A/B Testing, Model Registry, Rollback, Blue-Green Deployment, Model Monitoring

---

### Model Registry (Model Kaydı / Versiyon Yönetimi)
**Ne işe yarar:** Eğitilen her model versiyonunu, onu oluşturan veri, parametreler ve metriklerle birlikte kataloglayan merkezi sistem.

**Detaylı:** Bir süre sonra projede o kadar çok model dosyası birikir ki "hangi model canlıda?", "bu model hangi veriyle eğitildi?", "doğruluk skoru neydi?" gibi soruların cevabı kaybolur. Model Registry bu kaosu önler. Her model kaydedildiğinde şu bilgileri tutar: versiyon numarası, eğitim tarihi, kullanılan veri seti ID'si, hiperparametreler, değerlendirme metrikleri (accuracy, F1, vs.), modelin bulunduğu depo yolu (S3, Hugging Face Hub), ve deployment durumu (staging/production/archived). MLflow, DVC, Hugging Face Hub, ve Neptune.ai gibi araçlar bu işi yapar. En pratik özellik: "Promote to Production" butonu. Bir model test ortamında doğrulandıktan sonra tek tıkla production'a alınır, ve registry hemen eski versiyonu arşivler. Ayrıca lineage tracking sayesinde — bir model kötü sonuç verirse, hangi veriyle eğitildiğine kadar iz sürebilirsiniz. Audit log'u da cabası: kim, ne zaman, hangi modeli deploy etti belli olur.

**Örnek:** MLflow ile model registry:

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("metin-siniflandirma")

with mlflow.start_run() as run:
    # Modeli eğit
    model = RandomForestClassifier(n_estimators=100)
    X, y = make_classification(n_samples=1000)
    model.fit(X, y)
    
    # Metrikleri kaydet
    accuracy = model.score(X, y)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("veri_seti", "musteri-yorumlari-v3")
    
    # Modeli registry'e kaydet
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="metin-siniflandirici"
    )
    
    print(f"Run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.4f}")
    
# Registry'den model çekme
import mlflow.pyfunc

# Belirli bir versiyonu yükle
model = mlflow.pyfunc.load_model(
    model_uri="models:/metin-siniflandirici/3"  # version 3
)

# Stage'e göre yükle
model_staging = mlflow.pyfunc.load_model(
    model_uri="models:/metin-siniflandirici/Staging"  # staging'deki en son
)

model_prod = mlflow.pyfunc.load_model(
    model_uri="models:/metin-siniflandirici/Production"
)

# Registry'deki tüm versiyonları listele
from mlflow.tracking import MlflowClient
client = MlflowClient()
for v in client.search_model_versions("name='metin-siniflandirici'"):
    print(f"Versiyon {v.version}: stage={v.current_stage}, accuracy={v.tags.get('accuracy')}")

**İlişkili:** Model Deployment, A/B Testing, Canary Deployment, Experiment Tracking, Version Control

---

### Feature Store (Öznitelik Deposu)
**Ne işe yarar:** Model eğitimi ve inference'ı için kullanılan özniteliklerin (feature) merkezi olarak tanımlandığı, hesaplandığı ve sunulduğu veri katmanı.

**Detaylı:** Her modelin eğitim ve inference aşamasında aynı feature'ları kullanması gerekir, yoksa model çıktıları tutarsız olur. Feature Store bu sorunu çözer: bir kere feature tanımlarsınız, hem eğitimde hem canlıda aynı feature kullanılır. Online feature'lar (hızlı erişim, Redis gibi veritabanında) ve offline feature'lar (büyük hacimli, parquet/Delta Lake'te) olarak ikiye ayrılır. Mesela bir e-ticaret modeli için "son 7 gündeki ortalama sepet tutarı" feature'ını düşünün. Eğitim sırasında bu feature tarihsel veriden hesaplanır. Canlıda ise aynı feature anlık veriden hesaplanıp Redis'te tutulur. Feature Store aynı hesaplama kodunu kullandığı için training-serving skew (eğitimle canlı arasında feature farkı) oluşmaz. Popüler açık kaynak çözümler: Feast, Tecton (enterprise), Hopsworks. Feature Store aynı zamanda point-in-time join denilen bir işlem yapar — bir feature'ı hesaplarken "o tarihte o feature neydi?" diye sorar, böylece future data leakage (gelecekteki verinin sızdırılması) önlenir.

**Örnek:** Feast ile feature store:

```python
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64
from datetime import timedelta

# Feature tanımı (feature_store.yaml ile birlikte)
musteri = Entity(name="musteri_id", join_keys=["musteri_id"])

musteri_feature_view = FeatureView(
    name="musteri_ozellikleri",
    entities=[musteri],
    ttl=timedelta(days=1),
    schema=[
        Field(name="son_7_gun_harcama", dtype=Float32),
        Field(name="toplam_siparis", dtype=Int64),
        Field(name="ortalama_puan", dtype=Float32),
    ],
    source="musteri_data_source"
)

# Feature Store'dan veri çekme
store = FeatureStore(repo_path="./feature_repo")

# Online inference — anlık veri
online_features = store.get_online_features(
    features=["musteri_ozellikleri:son_7_gun_harcama"],
    entity_rows=[{"musteri_id": 12345}]
).to_dict()

print(f"Son 7 gün harcama: {online_features['son_7_gun_harcama'][0]} TL")

# Offline training — tarihsel veri
from datetime import datetime
eğitim_verisi = store.get_historical_features(
    features=["musteri_ozellikleri:son_7_gun_harcama"],
    entity_df="SELECT musteri_id, event_timestamp FROM egitim_verileri"
).to_df()
```

Feature Store olmadan yaşanan klasik sorun:
```python
# EĞİTİM: feature'ı elle hesapla
import pandas as pd
df["harcama_7_gun"] = df.groupby("musteri")["tutar"].rolling(7).sum()

# CANLI: aynı feature'ı yeniden hesapla (belki farklı hesaplar!)
def get_harcama(musteri_id):
    sorgu = "SELECT SUM(tutar) FROM siparisler WHERE musteri_id=? AND tarih > NOW() - 7"
    return db.query(sorgu, musteri_id)
# Yukarıdaki iki kod farklı sonuç verebilir → training-serving skew!

# Feature Store ile bu sorun ortadan kalkar
**İlişkili:** Model Registry, Model Deployment, Online/Offline Inference, Data Leakage, Feast
