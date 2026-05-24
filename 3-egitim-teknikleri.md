# Gün 3: Eğitim Teknikleri

Modelleri eğitmek kadar, onları doğru tekniklerle eğitmek de ayrı bir sanat. Bu bölümde bir modelin sıfırdan nasıl hazırlandığını, ince ayar yapılırken hangi yöntemlerin kullanıldığını ve kaynakları verimli tüketmek için neler yapıldığını anlatıyorum.

---

### Pretraining (Ön Eğitim)
**Ne işe yarar:** Devasa miktarda etiketsiz veriyle bir modeli sıfırdan eğiterek dilin temel yapısını, gramerini ve genel bilgiyi öğretme süreci.

**Detaylı:** Pretraining, bir bebeğin etrafındaki her şeyi dinleyerek dili öğrenmesine benziyor. Modele internetten toplanmış milyarlarca cümle gösteriliyor ve "Bir sonraki kelimeyi tahmin et" deniyor. Bu işlem haftalarca sürebiliyor, yüzlerce GPU/TPU saatlik hesaplama gücü gerektiriyor. Sonuçta ortaya "foundation model" çıkıyor — yani henüz özel bir işe yönlendirilmemiş, sadece dilin genel dinamiklerini öğrenmiş bir model. GPT, LLaMA, BERT gibi modellerin ilk halleri hep pretraining ile oluşturuluyor. Maliyeti oldukça yüksek — LLaMA 3 70B'yi eğitmek için milyonlarca dolar GPU zamanı harcanmıştı.

**Örnek:** 1 trilyon tokenlık veriyle model eğitmek isterseniz, yaklaşık 8 x A100 80GB GPU ile 3-4 hafta beklersiniz. Yani evde denenecek bir şey değil maalesef.

```
# Hugging Face Trainer ile basit bir pretraining
# Gerçekte çok daha büyük ölçekte olur
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer

model = AutoModelForCausalLM.from_pretrained("gpt2")
# Veri: milyarlarca token büyüklüğünde
# Kaynak: 8+ GPU, haftalarca süre
```

**İlişkili:** Foundation Model, Fine-tuning, Causal LM, Masked LM

---

### Supervised Fine-Tuning (SFT / Denetimli İnce Ayar)
**Ne işe yarar:** Pretrained modeli, elle hazırlanmış etiketli örneklerle belirli bir göreve veya davranış biçimine uyarlama süreci.

**Detaylı:** Pretrained model çok şey biliyor ama nasıl cevap vereceğini tam bilmiyor. SFT'de insanların yazdığı soru-cevap çiftleriyle modele "Bak, sorulara böyle cevap ver, şöyle yardımcı ol" diye öğretiyoruz. Genelde birkaç bin ile birkaç yüz bin arasında el yapımı örnek kullanılıyor. Loss fonksiyonu olarak cross-entropy kullanılır ve model sadece cevap kısmındaki tokenlar için loss hesaplar (soru kısmı maskelenir). SFT tek başına modeli yeterince iyi hizalamayabilir, bu yüzden genelde ardından RLHF gibi yöntemler eklenir.

**Örnek:** "3 tane yemek tarifi söyler misin?" sorusuna modelin "Elbette! İşte 3 nefis tarif..." diye başlayıp liste halinde vermesi SFT ile öğretilen bir davranış.

```python
# SFT için dataset formatı genelde şöyle
{
  "instruction": "Türkiye'nin başkenti neresidir?",
  "output": "Türkiye'nin başkenti Ankara'dır."
}
# Hugging Face ile:
from trl import SFTTrainer

trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    max_seq_length=2048,
)
trainer.train()
```

**İlişkili:** RLHF, Instruction Tuning, DPO, Pretraining

---

### RLHF (Reinforcement Learning from Human Feedback / İnsan Geri Bildirimiyle Pekiştirmeli Öğrenme)
**Ne işe yarar:** İnsan tercihlerini kullanarak modelin çıktılarını hizalamak, yani modeli daha faydalı, dürüst ve zararsız hale getirmek.

**Detaylı:** RLHF üç aşamadan oluşuyor. İlk aşamada bir SFT yapılıyor (yukarıda anlattım). İkinci aşamada bir "reward model" (ödül modeli) eğitiliyor — insanlar aynı soruya verilen iki farklı cevabı karşılaştırıp hangisinin daha iyi olduğunu söylüyor, model de bunu öğreniyor. Son aşamada ise asıl model PPO algoritmasıyla eğitiliyor: model cevap üretiyor, reward model puan veriyor, model de puanı yükseltmek için kendini güncelliyor. Arada bir divergence penalty (KL divergence) ekleniyor ki model çok sapmasın. ChatGPT'nin ardındaki sihir tam olarak bu.

**Örnek:** "Bana bomba yapımını anlat" sorusuna model ya çok detaylı anlatır (kötü) ya da direkt reddeder (aşırı). RLHF ile model "Bu konuda yardımcı olamam, başka bir konuda yardımcı olabilir miyim?" gibi dengeli bir cevap vermeyi öğrenir.

```
# RLHF aşamaları çok karmaşık, özet mantık:
# Aşama 1: SFT — modele doğru cevap formatını öğret
# Aşama 2: Reward Model — insan tercihlerinden puanlamayı öğren
# Aşama 3: PPO — modeli reward model'a göre optimize et
#             + KL penalty ekle ki çok uçmasın
```

**İlişkili:** PPO, Reward Model, DPO, SFT, Alignment

---

### DPO (Direct Preference Optimization / Doğrudan Tercih Optimizasyonu)
**Ne işe yarar:** RLHF'in karmaşıklığını ortadan kaldıran, doğrudan insan tercihleriyle modeli optimize eden daha basit bir yöntem.

**Detaylı:** RLHF'te ayrı bir reward model eğitmeniz ve PPO algoritması kurmanız gerekiyor — bu stabil olmayan, ince ayar gerektiren bir süreç. DPO ise bunu tek adımda çözüyor. Matematiksel bir dönüşüm sayesinde reward model'i ayrıca eğitmeye gerek kalmıyor, doğrudan tercih verisi (hangi cevap daha iyi) üzerinden policy loss hesaplanıyor. Yani DPO, "İnsanlar şu cevabı daha çok beğendi, gel modeli o yöne çek" diyor. Daha hızlı, daha stabil ve çoğu durumda RLHF kadar iyi sonuç veriyor. Son dönemde neredeyse herkes DPO'ya geçmiş durumda.

**Örnek:** Tercih verisi şöyle: (soru, iyi_cevap, kötü_cevap). DPO bu üçlüden direkt loss hesaplar. RLHF'teki reward model + PPO karmaşası olmaz.

```python
from trl import DPOTrainer

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,  # SFT'li model (referans)
    dataset=preference_data,
    beta=0.1,  # KL düzenleme katsayısı
)
dpo_trainer.train()
```

**İlişkili:** RLHF, PPO, Reward Model, Alignment

---

### LoRA (Low-Rank Adaptation / Düşük Rütbeli Uyarlama)
**Ne işe yarar:** Büyük bir modelin tamamını değil, çok küçük bir kısmını güncelleyerek ince ayar yapmayı sağlayan verimli bir yöntem.

**Detaylı:** Mikroişlemcilerde register dosyası gibi düşünün. LoRA'nın fikri şu: Bir ağırlık matrisinde yapılacak güncellemenin düşük rütbeli olduğu varsayımından yola çıkıyor. Yani W matrisine ekleyeceğimiz ΔW'yi A×B şeklinde iki küçük matrisin çarpımı olarak temsil ediyoruz. A (r x k) ve B (d x r) — burada r, full rank'ten çok daha küçük bir sayı (genelde 8 veya 16). Bu sayede full fine-tuning'de 340 milyon parametre güncellenirken, LoRA ile sadece 0.5 milyon parametre güncelleniyor. Modelin orijinal ağırlıkları donuk kalıyor, sadece bu ek matrisler eğitiliyor. Inference'da ise bu iki matristen ΔW = AB hesaplanıp orijinal ağırlıklara ekleniyor.

**Örnek:** 7B param'lik bir modeli full fine-tune etmek için ~140GB GPU belleği gerekirken, LoRA ile ~16GB yeterli. Yani bir tane RTX 4090 bile iş görüyor.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,            # LoRA rank'i (ne kadar düşük o kadar az parametre)
    lora_alpha=32,  # Ölçekleme katsayısı
    target_modules=["q_proj", "v_proj"],  # Hangi katmanlara LoRA uygulansın
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(base_model, lora_config)
# Toplam parametre sayısı hala 7B
# Ama eğitilebilir parametre sayısı: ~4.2M (sadece %0.06!)
```

**İlişkili:** QLoRA, Fine-tuning, PEFT, Parameter Efficiency

---

### QLoRA (Quantized LoRA / Nicelenmiş LoRA)
**Ne işe yarar:** LoRA'nın bir adım ötesi — model ağırlıklarını 4-bit'e düşürüp LoRA ile ince ayar yapmayı mümkün kılan, hafıza kullanımını iyice minimize eden yöntem.

**Detaylı:** LoRA zaten hafif bir yöntemdi, QLoRA ise bunu bir kat daha ileri taşıyor. Modelin ağırlıklarını 4-bit normal float'a (NF4) dönüştürüyor ve bunları donuk tutuyor. LoRA katmanları ise 16-bit veya 32-bit'te eğitiliyor. Ayrıca "double quantization" ile quantization sabitlerini de niceliyor — ekstra hafıza kazancı sağlıyor. Normalde 65 milyar param'lik bir modeli fine-tune etmek için 780 GB GPU gerekirken, QLoRA ile sadece 48 GB yetiyor. İşte bu devrim niteliğinde bir gelişme. Tek bir A100 ile 65B model ince ayarı yapmak mümkün hale geliyor.

**Örnek:** 7B boyutundaki LLaMA-3 modeli full fine-tuning'de 56 GB isterken, QLoRA ile sadece 6-8 GB GPU RAM yeterli oluyor. Neredeyse herkesin ev bilgisayarında çalıştırabileceği bir hale geliyor.

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 4-bit'te yükle, LoRA ekle
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=bnb_config
)
model = get_peft_model(model, lora_config)
model.train()
```

**İlişkili:** LoRA, Quantization, PEFT, BitsAndBytes

---

### Quantization (Niceleme — FP16 / INT8 / INT4)
**Ne işe yarar:** Model ağırlıklarını daha düşük hassasiyette saklayarak bellek kullanımını ve hızı optimize etme tekniği. Modele daha az yer kaplama ve hızlı çalışma imkanı verir.

**Detaylı:** Normalde modeller 32-bit kayan noktalı (FP32) sayılarla çalışır. Quantization bunu 16-bit, 8-bit hatta 4-bit'e indiriyor. Mesela FP32'den INT8'e geçince model boyutu %75 küçülüyor, ama çoğu durumda kalite kaybı %1-2'yi geçmiyor. FP16 (yarı hassasiyet) genelde eğitim için kullanılırken, INT8/INT4 daha çok inference içindir. GPTQ, AWQ, GGUF gibi farklı quantization yöntemleri var — herbirinin kalite ve hız arasında farklı bir dengesi var. GGUF özellikle CPU'da çalıştırmak için ideal. Önemli bir nokta: quantization sonrası modelin ağırlıkları artık ondalıklı sayılar yerine tam sayılara dönüştüğü için işlemler çok daha hızlı yapılabiliyor.

**Örnek:** Mistral 7B modeli FP32'de 28 GB, FP16'da 14 GB, INT8'de 7 GB, INT4'te ise sadece 3.5 GB yer kaplıyor. Aynı modeli 4090'da INT8'de çalıştırırken, INT4 ile 8 GB RAM'li bir MacBook'ta bile çalıştırabiliyorsunuz.

```
# Quantization seviyelerinin karşılaştırması
# FP32  → 4 bytes/param  → 7B model: 28 GB
# FP16  → 2 bytes/param  → 7B model: 14 GB  
# INT8  → 1 byte/param   → 7B model:  7 GB
# INT4  → 0.5 byte/param → 7B model:  3.5 GB
# NF4   → ~0.5 byte/param (normal float, QLoRA'da kullanılır)
```

**İlişkili:** QLoRA, GPTQ, AWQ, GGUF, Inference Optimization

---

### Knowledge Distillation (Bilgi Damıtma)
**Ne işe yarar:** Büyük ve yetenekli bir modelin (öğretmen) bilgisini daha küçük bir modele (öğrenci) aktarma tekniği. Büyük modelin kalitesini korurken küçük modelin hızından faydalanmayı sağlar.

**Detaylı:** Büyük modeller yavaş ve pahalıdır. Knowledge distillation'da büyük model (teacher) bir veri kümesi üzerinde çalıştırılır ve çıktı olasılıkları (soft labels — sadece en yüksek değil, tüm sınıfların olasılıkları) kaydedilir. Küçük model (student) ise hem gerçek etiketleri hem de teacher'ın olasılıklarını taklit ederek eğitilir. Teacher'ın "soft" çıktıları çok değerlidir çünkü sınıflar arası ilişkileri içerir — mesela "kedi" için teacher "kedi %90, köpek %5, aslan %3" derse, öğrenci sadece "kedi" etiketini değil, bu benzerlik ilişkilerini de öğrenir. Temperature hyperparameter'ı bu olasılıkların ne kadar "yumuşak" olacağını kontrol eder.

**Örnek:** Google'ın BERT modelini 12 katman yerine 6 katmanlı küçük bir model (DistilBERT) olarak damıtması — BERT'in accuracy'sinin %97'sini korurken boyutunu yarıya indiriyor ve 2 kat hızlanma sağlıyor.

```python
def distillation_loss(student_logits, teacher_logits, labels, T=4.0):
    """
    T: temperature — yüksek değerler daha yumuşak olasılıklar
    """
    soft_loss = KL_divergence(
        softmax(teacher_logits / T),
        softmax(student_logits / T)
    ) * (T ** 2)
    
    hard_loss = cross_entropy(student_logits, labels)
    
    return alpha * soft_loss + (1 - alpha) * hard_loss
```

**İlişkili:** Model Compression, Quantization, Pruning, DistilBERT

---

### Gradient Checkpointing (Gradyant Kontrol Noktalama)
**Ne işe yarar:** Eğitim sırasında ara katman aktivasyonlarını bellekte tutmak yerine ihtiyaç halinde yeniden hesaplayarak bellek kullanımını azaltan bir teknik.

**Detaylı:** Normal eğitimde forward pass sırasında her katmanın çıktısı bellekte tutulur ki backward pass'te gradyan hesaplanabilsin. 32 katmanlı bir modelde bu 32 katmanın çıktısı demek — çok fazla bellek. Gradient checkpointing'te ise aktivasyonların tamamı değil, sadece belirli aralıklarla (örneğin her 4 katmanda bir) kaydedilir. Backward sırasında kaydedilmeyen katmanların aktivasyonu, en yakın checkpoint'tan forward pass tekrar çalıştırılarak hesaplanır. Bellek kullanımı %50-70 azalır ama %15-30 arası ek süre kaybı olur. Yani zamanla bellek arasında bir trade-off var. Büyük modeller eğitilirken neredeyse her zaman kullanılıyor.

**Örnek:** 12 katmanlı bir modeli batch size 4 ile eğitirken OOM (out of memory) hatası alıyorsanız gradient checkpointing açın, batch size'ı 8'e çıkarıp daha verimli eğitim yapabilirsiniz.

```python
import torch

model = LargeTransformer()
# Tüm katmanlarda gradient checkpointing aktif
model.gradient_checkpointing_enable()

# Veya Hugging Face ile:
model.config.use_cache = False  # (gradient checkpointing ile uyumsuz)
model.gradient_checkpointing_enable()

# Bellek: ~%50 tasarruf
# Hız: ~%20 yavaşlama
# Ama batch size 2 katına çıkabilir → net hızlı
```

**İlişkili:** Mixed Precision, Bellek Optimizasyonu, Activation Memory

---

### Mixed Precision Training (Karışık Hassasiyetli Eğitim)
**Ne işe yarar:** Modeli hem 16-bit hem 32-bit kullanarak eğitme tekniği. Büyük modellerin eğitimini hızlandırırken bellek tüketimini azaltır, ama kalite kaybı yaşanmaz.

**Detaylı:** İşin sırrı şu: Modelin çoğu hesaplaması FP16'da (yarı hassasiyet) yapılır çünkü FP16 işlemleri FP32'den 2-4 kat daha hızlıdır. Ama bazı kritik işlemler (gradyan birikimi, loss scaling, normalization gibi) FP32'de tutulur. Bu karışım sayesinde FP32 kadar kaliteli ama FP16 kadar hızlı. Modern GPU'ların (A100, H100, RTX 3090+) Tensor Core'ları FP16 hesaplamak için optimize edilmiştir. AMP (Automatic Mixed Precision) bu işi otomatik yapar. Loss scaling özellikle önemli — küçük gradyanlar FP16'da sıfıra yuvarlanmasın diye loss büyütülür, backward'dan sonra eski haline döndürülür.

**Örnek:** 70B'lik bir modeli FP32'de eğitmeye çalışırsanız 280 GB GPU, FP16'da 140 GB gerekir. Mixed precision ile FP16'nın hızı + FP32'nin kararlılığı birleşmiş olur.

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # Otomatik loss scaling

for batch in dataloader:
    with autocast():  # Otomatik FP16/FP32 karışımı
        outputs = model(batch)
        loss = loss_fn(outputs, batch["labels"])
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

**İlişkili:** Quantization, Gradient Checkpointing, AMP, Tensor Core

---

### Curriculum Learning (Müfredat Öğrenmesi)
**Ne işe yarar:** Modele konuları kolaydan zora doğru sıralayarak öğretme stratejisi. Tıpkı bir okulda toplamadan integrale gitmek gibi.

**Detaylı:** Normal eğitimde model bütün örnekleri rastgele görür. Curriculum learning'de ise veri sıralanır — önce kolay, sonra orta, en son zor örnekler. Kolay örnekler modelin temel kalıpları hızlıca öğrenmesini sağlarken, zor örnekler ince detaylara odaklanmasını sağlıyor. "Difficulty" (zorluk) metriği çeşitli şekillerde tanımlanabilir: cümle uzunluğu, kelime çeşitliliği, modelin o anki loss değeri vs. Özellikle makine çevirisi, duygu analizi ve görsel alanlarda işe yarıyor. Her durumda işe yaramasa da, doğru uygulandığında daha hızlı yakınsama ve daha yüksek final accuracy sağlıyor.

**Örnek:** GPT-2'yi eğitirken önce kısa cümleler (5-10 token), sonra paragraflar (50-100 token), sonra sayfalar (1000+ token) şeklinde sıralamak. Veya çeviri modelini önce basit cümlelerle, sonra karmaşık yapılarla eğitmek.

```python
def curriculum_sampler(dataset, epoch, total_epochs):
    """Örnekleri epoch ilerledikçe zorlaştır"""
    # Kolay (kısa) örnekler başta, zor (uzun) örnekler sonda
    dataset = sorted(dataset, key=lambda x: len(x["text"]))
    progress = epoch / total_epochs
    
    # İlk %20 zorluğa karşılık gelen örnekler
    cutoff = int(len(dataset) * (0.2 + 0.8 * progress))
    return dataset[:cutoff]
```

**İlişkili:** Active Learning, Training Strategies, Difficulty Scoring

---

### Few-shot Learning (Az Örnekle Öğrenme)
**Ne işe yarar:** Modele sadece birkaç örnek göstererek yeni bir görevi yapmasını sağlama tekniği. Ek eğitime gerek kalmadan, sadece prompt içinde örnek vererek çalışır.

**Detaylı:** Büyük dil modellerinin en etkileyici özelliklerinden biri. Modele bir görevi anlatmak için yeni bir model eğitmek veya fine-tuning yapmak gerekmiyor. Sadece prompt'un içine 2-3 örnek koyuyorsunuz, model de aynı kalıbı takip ederek cevap veriyor. Zero-shot (hiç örnek yok), one-shot (1 örnek), few-shot (2-5 örnek) olarak derecelendiriliyor. Bu yetenek özellikle GPT-3 ile ortaya çıktı ve model boyutu büyüdükçe bu yeteneğin de arttığı görüldü (scaling laws). Aslında model bu sırada yeni bir şey öğrenmiyor — zaten var olan bilgiyi prompt'taki kalıba göre şekillendiriyor. In-context learning olarak da geçer.

**Örnek:** Duygu analizi yapmak istiyorsunuz, ek bir model eğitmenize gerek yok:

```
Metin: "Bugün hava çok güzel"
Duygu: Pozitif

Metin: "Yemeğin tadı berabattı"
Duygu: Negatif

Metin: "Toplantı saat 3'te"
Duygu: Nötr

Metin: "Bu film inanılmazdı, herkes izlemeli"
Duygu: [model burayı tamamlar → "Pozitif"]
```

**Örnek (Python ile):**
```python
def few_shot_sentiment(text, examples):
    prompt = ""
    for example in examples:
        prompt += f"Metin: \"{example['text']}\"\n"
        prompt += f"Duygu: {example['sentiment']}\n\n"
    
    prompt += f"Metin: \"{text}\"\n"
    prompt += "Duygu: "
    # Modeli çağır ve prompt'u tamamla
    return call_model(prompt)
```

**İlişkili:** Zero-shot Learning, In-context Learning, Prompt Engineering, Scaling Laws
