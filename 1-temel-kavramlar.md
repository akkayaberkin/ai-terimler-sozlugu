# Gün 1: Temel Kavramlar

## İçindekiler

| # | Terim | İngilizce |
|---|-------|-----------|
| 1 | [Tokenization](#1-tokenization) | Tokenization |
| 2 | [Gömme Vektörü (Embedding)](#2-embedding) | Embedding |
| 3 | [Sözcük Dağarcığı (Vocabulary)](#3-vocabulary) | Vocabulary |
| 4 | [Bağlam Penceresi (Context Window)](#4-context-window) | Context Window |
| 5 | [Sıcaklık (Temperature)](#5-temperature) | Temperature |
| 6 | [Top-p (Nucleus Sampling)](#6-top-p-nucleus-sampling) | Top-p / Nucleus Sampling |
| 7 | [Top-k](#7-top-k) | Top-k |
| 8 | [Logits](#8-logits) | Logits |
| 9 | [Softmax](#9-softmax) | Softmax |
| 10 | [One-hot Kodlama](#10-one-hot-encoding) | One-hot Encoding |
| 11 | [Dizi Uzunluğu (Sequence Length)](#11-sequence-length) | Sequence Length |
| 12 | [Batch Boyutu (Batch Size)](#12-batch-size) | Batch Size |

---

### 1. Tokenization

**Ne işe yarar:** Ham metni, modelin anlayabileceği küçük parçalara (token'lara) böler.

**Detaylı Açıklama:** Bir cümleyi modele göndermeden önce onu sayılara çevirmen gerekir. Tokenization, metni kelimelere, hecelere veya karakterlere ayırarak her bir parçaya bir ID atar. "Merhaba dünya" cümlesi iki token olabilir veya ["Mer", "haba", "dün", "ya"] şeklinde daha küçük parçalara ayrılabilir. Modern modeller (GPT, Llama gibi) genelde BPE veya SentencePiece gibi alt kelime (subword) tokenizer'lar kullanır.

**Örnek:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer.tokenize("Tokenization nedir?")
print(tokens)  # ['token', '##ization', 'nedir', '?']
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)  # [19204, 3567, 21490, 1029]
```

**İlişkili Terimler:** Vocabulary, Sequence Length, Embedding

---

### 2. Embedding

**Ne işe yarar:** Token'ları veya kelimeleri anlamsal ilişkileri koruyan sayısal vektörlere dönüştürür.

**Detaylı Açıklama:** Embedding, her token'ı yüzlerce veya binlerce boyutlu bir vektör olarak temsil eder. Bu vektörler öyle hesaplanır ki anlamca benzer kelimeler birbirine yakın, zıt anlamlılar uzak konumlanır. Örneğin "kral" ve "kraliçe" vektörleri arasındaki fark, "adam" ve "kadın" arasındaki farka benzer olur. Word2Vec, GloVe gibi klasik yöntemler de vardır ama bugün modeller kendi embedding katmanlarını eğitimin bir parçası olarak öğrenir.

**Örnek:**
```python
# Basit bir embedding katmanı
import torch
embedding = torch.nn.Embedding(num_embeddings=1000, embedding_dim=128)
token_id = torch.tensor([42])
vector = embedding(token_id)
print(vector.shape)  # torch.Size([1, 128])
```

**İlişkili Terimler:** Tokenization, Vocabulary, Vector Database

---

### 3. Vocabulary

**Ne işe yarar:** Modelin tanıdığı tüm token'ların kümesidir.

**Detaylı Açıklama:** Vocabulary, modelin eğitildiği veri kümesinden çıkarılan ve modelin üretebileceği ya da anlayabileceği tüm token'ları içeren bir sözlüktür. Her token'ın bir ID'si vardır (0'dan vocabulary_size - 1'e kadar). Küçük modeller 10-30K token kullanırken, GPT-4 gibi büyük modeller 100K+ token'a sahiptir. Vocabulary'nin dışındaki kelimeler "bilinmeyen token" (UNK) ile temsil edilir.

**Örnek:**
```python
# Bir tokenizer'ın vocabulary büyüklüğünü görmek
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(len(tokenizer))  # 50257 — GPT-2'nin vocabulary boyutu
```

**İlişkili Terimler:** Tokenization, One-hot Encoding, Embedding

---

### 4. Context Window

**Ne işe yarar:** Modelin bir kerede işleyebileceği maksimum token sayısını belirler.

**Detaylı Açıklama:** Context window (bağlam penceresi), modelin girdi olarak alabileceği token sayısının üst sınırıdır. Bu sınıra ulaşıldığında, model en eski token'ları "hatırlamayı" bırakır. Klasik modellerde 512-2048 token civarındayken, günümüzde 128K, 200K hatta 1M token destekleyen modeller var (Claude, Gemini, GPT-4 Turbo gibi). Ne kadar büyük olursa, o kadar uzun metinleri veya tüm dokümanları tek seferde işleyebilirsin.

**Örnek:**
```python
# Token sayısını hesaplama
text = "Uzun bir metin..." * 1000
tokens = tokenizer.encode(text)
print(f"Token sayısı: {len(tokens)}")
print(f"Context window aşıldı mı? {len(tokens) > model_max_length}")
```

**İlişkili Terimler:** Sequence Length, Attention Mechanism, Transformer

---

### 5. Temperature

**Ne işe yarar:** Modelin ürettiği çıktıların ne kadar "rastgele" veya "yaratıcı" olacağını kontrol eder.

**Detaylı Açıklama:** Temperature, softmax çıktısının olasılık dağılımını yumuşatmak veya keskinleştirmek için kullanılan bir parametredir. Değer 1.0'a yakın veya üstünde olduğunda düşük olasılıklı token'lar da seçilme şansı bulur, bu da daha yaratıcı ama tutarsız çıktılar demektir. 0'a yaklaştıkça dağılım keskinleşir, model neredeyse her zaman en yüksek olasılıklı token'ı seçer — yani daha deterministik ve tekrarlayıcı olur. Genelde 0.1 ile 1.5 arası değerler kullanılır.

**Örnek:**
```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1, 0.5])

def apply_temperature(logits, temp):
    return F.softmax(logits / temp, dim=-1)

print(apply_temperature(logits, 0.5))  # Keskin dağılım, yüksek olan daha yüksek
print(apply_temperature(logits, 1.5))  # Yumuşak dağılım, düşük olan da şanslı
```

**İlişkili Terimler:** Softmax, Logits, Top-p, Top-k

---

### 6. Top-p (Nucleus Sampling)

**Ne işe yarar:** Olasılık dağılımını kırparak sadece en olası token'ların bir kısmını seçime dahil eder.

**Detaylı Açıklama:** Top-p (veya nucleus sampling), kümülatif olasılığı p değerine kadar olan en küçük token kümesini seçer ve geri kalanı eler. Örneğin p=0.9 ise, en yüksek olasılıklı token'lardan başlayarak toplam olasılıkları 0.9'a ulaşana kadar token'ları toplar, diğerlerini atar. Temperature'dan farkı, sabit bir sayı (k) kullanmak yerine dinamik bir eşik belirlemesidir. Çok eşit dağılımlarda daha fazla token seçer, çok keskin dağılımlarda sadece birkaç token bırakır.

**Örnek:**
```python
def top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Kümülatif olasılık top_p'i geçen token'ları sil
    sorted_indices_to_remove = cum_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    return sorted_indices_to_remove
```

**İlişkili Terimler:** Temperature, Top-k, Logits, Softmax

---

### 7. Top-k

**Ne işe yarar:** Modelin bir sonraki token'ı seçerken sadece en olası k kadar token arasından seçim yapmasını sağlar.

**Detaylı Açıklama:** Top-k, olasılık sıralamasında ilk k token dışındaki tüm token'ların olasılığını sıfırlayarak çalışır. k=40 popüler bir varsayılan değerdir, yani her adımda model sadece en olası 40 token arasından seçim yapar. Düşük k değerleri daha güvenli ve tahmin edilebilir çıktılar üretirken, yüksek k değerleri daha çeşitli çıktılar sağlar. Top-p ile birlikte de kullanılabilir: önce top-k uygulanır, sonra top-p filtresi geçirilir.

**Örnek:**
```python
def top_k_filtering(logits, k=40):
    if k == 0:
        return logits
    # k'ıncı en büyük değerin altındaki her şeyi -inf yap
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(-1)
    return torch.where(logits < min_values, float('-inf'), logits)
```

**İlişkili Terimler:** Top-p, Temperature, Logits, Softmax

---

### 8. Logits

**Ne işe yarar:** Modelin son katmanındaki ham, normalize edilmemiş skorlardır ve bir sonraki token'ın ne olacağını belirler.

**Detaylı Açıklama:** Logits, modelin çıktı katmanından gelen, her vocabulary token'ı için bir skor içeren ham değerlerdir. Bu değerler pozitif veya negatif olabilir, herhangi bir aralıkla sınırlı değildir. Anlamlı olasılıklara dönüştürülmek için softmax fonksiyonundan geçirilmeleri gerekir. Modelin "kafasındaki" en olası token'ı görmek istiyorsan argmax(logits) yapabilirsin — bu en yüksek logit değerine sahip token'ı verir.

**Örnek:**
```python
import torch

# Modelin çıktısı — vocabulary'deki her kelime için bir skor
logits = torch.tensor([-2.3, 5.1, 0.7, -1.2, 3.8, -0.5])
predicted_idx = torch.argmax(logits).item()
print(f"En yüksek logit: {logits[predicted_idx].item()}")  # 5.1
```

**İlişkili Terimler:** Softmax, Temperature, Cross-Entropy Loss

---

### 9. Softmax

**Ne işe yarar:** Ham logit değerlerini 0 ile 1 arasında, toplamları 1 olan olasılıklara dönüştürür.

**Detaylı Açıklama:** Softmax, bir vektördeki her bir elemanı üstel (exponential) fonksiyondan geçirip tüm elemanların toplamına bölerek normalize eder. Matematiksel olarak: softmax(x_i) = exp(x_i) / sum(exp(x_j)). Bu sayede büyük logit değerleri daha da büyür, küçükler iyice küçülür — yani farkları belirginleştirir. Modelin son katmanında olasılık dağılımı çıkarmak için kullanılır ve classification problemlerinin temel taşıdır.

**Örnek:**
```python
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1])
probs = F.softmax(logits, dim=-1)
print(probs)  # tensor([0.6590, 0.2424, 0.0986])
# Toplam 1.0 — gerçek olasılık
print(probs.sum())  # tensor(1.0)
```

**İlişkili Terimler:** Logits, Temperature, Cross-Entropy Loss, Classification

---

### 10. One-hot Encoding

**Ne işe yarar:** Kategorik verileri (kelimeler, etiketler) modele verilebilecek sayısal vektörlere dönüştürür.

**Detaylı Açıklama:** One-hot encoding, bir kategorik değişkeni, sadece bir elemanı 1 diğerleri 0 olan bir vektörle temsil eder. Kelime dağarcığındaki her kelime için bir indeks belirlenir ve o indeksteki değer 1, diğerleri 0 olur. Örneğin vocabulary'de 10.000 kelime varsa, her kelime 9.999 tane 0 ve bir tane 1 içeren 10.000 boyutlu bir vektörle temsil edilir. Pratikte bu çok verimsiz olduğu için daha çok embedding katmanlarının girişinde başlangıç adımı olarak kullanılır.

**Örnek:**
```python
import torch.nn.functional as F

# 5 kelimelik bir vocabulary, 3. kelimeyi (index 2) one-hot olarak temsil et
vocab_size = 5
index = 2
one_hot = F.one_hot(torch.tensor(index), num_classes=vocab_size)
print(one_hot)  # tensor([0, 0, 1, 0, 0])
```

**İlişkili Terimler:** Embedding, Vocabulary, Tokenization

---

### 11. Sequence Length

**Ne işe yarar:** Modele aynı anda gönderilen girdi dizisindeki token sayısını tanımlar.

**Detaylı Açıklama:** Sequence length (dizi uzunluğu), modele bir forward pass'te verilen token dizisinin uzunluğudur. Tüm örneklerin aynı uzunlukta olması için genelde padding (doldurma) yapılır, yani kısa diziler özel bir PAD token'ı ile istenen uzunluğa tamamlanır. Uzun sequence length'ler daha fazla bağlam demektir ama aynı zamanda daha fazla bellek ve işlem gücü gerektirir. Transformer modellerinde bellek kullanımı sequence length'in karesiyle orantılıdır (O(n²)).

**Örnek:**
```python
# Farklı uzunluktaki dizileri padding ile eşitleme
sequences = ["merhaba", "bugün hava çok güzel", "nasılsın"]
tokenized = [tokenizer.encode(s) for s in sequences]
max_len = max(len(t) for t in tokenized)
# Kısa olanları PAD token (0) ile doldur
padded = [t + [0] * (max_len - len(t)) for t in tokenized]
print(padded)
```

**İlişkili Terimler:** Context Window, Batch Size, Tokenization, Attention

---

### 12. Batch Size

**Ne işe yarar:** Modelin bir adımda (one forward/backward pass) aynı anda işlediği örnek sayısını belirler.

**Detaylı Açıklama:** Batch size, eğitim veya inference sırasında aynı anda modele verilen örnek sayısıdır. Batch size büyüdükçe GPU'dan daha verimli yararlanılır çünkü paralel hesaplama avantajı devreye girer. Ama çok büyük batch size'lar daha fazla GPU belleği (VRAM) gerektirir ve generalization'ı (genelleme) olumsuz etkileyebilir. Küçük batch size'lar (genelde 1-32 arası) daha stabil eğitim sağlarken eğitim süresini uzatır. Büyük batch size'lar (64-512+) hızlıdır ama learning rate'in de ayarlanması gerekir.

**Örnek:**
```python
# Farklı batch size'larla bellek kullanımı
import torch

batch_size = 32
seq_len = 512
vocab_size = 50000

# Tek batch'in bellekte kapladığı yer (float32)
tokens_per_batch = batch_size * seq_len
memory_mb = (tokens_per_batch * 4) / (1024 * 1024)  # 4 bytes per float32
print(f"Batch basına token: {tokens_per_batch}")
print(f"Yaklaşık bellek: {memory_mb:.2f} MB (sadece girdi)")
```

**İlişkili Terimler:** Sequence Length, GPU Memory, Training Loop, Gradient Accumulation

---

*Sözlük devam ediyor — sonraki gün: Model Mimarisi*
