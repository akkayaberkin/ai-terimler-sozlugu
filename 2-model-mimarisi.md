# Gün 2: Model Mimarisi

Bu bölümde yapay zeka modellerinin iç yapısını oluşturan temel mimari bileşenleri ele alıyoruz.

---

### Transformer (Transformer Mimarisi)
**Ne işe yarar:** Diziler arasındaki uzun mesafeli ilişkileri paralel olarak işleyebilen sinir ağı mimarisi. Günümüzdeki neredeyse tüm büyük dil modellerinin temelini oluşturur.

**Detaylı:** 2017'de Google'ın "Attention Is All You Need" makalesiyle hayatımıza giren bu mimari, RNN'lerin sıralı yapısını tamamen terk ediyor. RNN'ler adım adım ilerlerken, Transformer tüm tokenları aynı anda işleyip aralarındaki ilişkiyi attention mekanizmasıyla öğreniyor. Bu sayede hem çok daha hızlı eğitim hem de çok daha uzun bağlamlar mümkün hale geliyor. En temel haliyle Encoder-Decoder yapısında çalışır, ama GPT gibi modeller sadece decoder kullanırken BERT sadece encoder kullanıyor.

**Örnek:** Basitçe düşün — "Ali topu at, o da onu yakaladı" cümlesinde "o" ve "onun" kim olduğunu bulmak için Transformer tüm kelimelere aynı anda bakar. RNN bunu sırayla yapmak zorunda kalırdı.

```
# PyTorch ile minimal bir Transformer katmanı
import torch
import torch.nn as nn

transformer_layer = nn.TransformerEncoderLayer(
    d_model=512,      # Gömmelerin boyutu
    nhead=8,          # Çok başlı attention'daki kafa sayısı
    dim_feedforward=2048,
    dropout=0.1,
    activation='relu'
)
encoder = nn.TransformerEncoder(transformer_layer, num_layers=6)
```

**İlişkili:** Attention, Self-Attention, Encoder-Decoder, BERT, GPT

---

### Attention (Dikkat Mekanizması)
**Ne işe yarar:** Modelin girdi dizisinin hangi kısımlarına odaklanması gerektiğini belirleyen mekanizma. Transformer mimarisinin kalbi diyebiliriz.

**Detaylı:** Attention sayesinde model bir çıktı üretirken girdideki her token'a farklı ağırlıklar veriyor. Mesela "kedi matın üzerinde" derken "kedi" kelimesini üretirken model "kedi" girdisine daha çok bakıyor, "mat" kısmının ağırlığı daha düşük kalıyor. Bu mekanizma sayesinde model, girdinin herhangi bir noktasına doğrudan erişebiliyor — aradaki mesafe önemli değil. "Scaled Dot-Product Attention" en yaygın kullanılan türüdür.

**Örnek:** Çeviri yaparken "The cat sat on the mat" → "Kedi matın üzerinde oturuyor" çevirisinde, "kedi" çıktısı üretilirken "cat" kelimesine en yüksek attention ağırlığı verilir.

```python
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """Basit Scaled Dot-Product Attention"""
    scores = torch.matmul(Q, K.transpose(-2, -1))
    scores = scores / (K.size(-1) ** 0.5)  # Ölçeklendirme
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)
```

**İlişkili:** Multi-Head Attention, Self-Attention, Cross-Attention, Transformer

---

### Self-Attention (Öz Dikkat)
**Ne işe yarar:** Bir dizideki her elemanın diğer elemanlarla ilişkisini hesaplayan attention türü. Modelin cümle içindeki bağlamı anlamasını sağlar.

**Detaylı:** Normal attention'da model bir diziyi (girdi) başka bir diziyle (çıktı) karşılaştırır. Self-attention'da ise aynı dizi hem sorgu hem anahtar hem de değer olarak kullanılıyor. Yani her kelime diğer tüm kelimelere "Ben sana nasıl bağlıyım?" diye soruyor. "Banka" kelimesi bir cümlede "para çektim" ile geçiyorsa finansal, "oturdum" ile geçiyorsa fiziksel banka anlamını alıyor. İşte self-attention bu ayrımı yapıyor.

**Örnek:** "O bankaya gitti ve parasını çekti" cümlesinde self-attention sayesinde "O" → "bankaya" → "parasını" arasındaki bağlantılar kurulur ve model "banka"nın finansal bağlamda kullanıldığını anlar.

```
# Q, K, V aynı kaynaktan geliyor — işte self-attention budur
self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=8)
x = torch.randn(10, 32, 512)  # (seq_len, batch, embed_dim)
output, weights = self_attention(x, x, x)  # Q=K=V=x
```

**İlişkili:** Attention, Multi-Head Attention, Transformer

---

### Multi-Head Attention (Çok Başlı Dikkat)
**Ne işe yarar:** Attention hesaplamasını birden çok kafaya bölüp her birinin farklı ilişki türlerini öğrenmesini sağlayan mekanizma.

**Detaylı:** Tek bir attention hesaplamak yerine, model aynı anda birden çok attention hesaplaması yapıyor. Her "kafa" farklı bir şeye odaklanabiliyor — biri dilbilgisel ilişkilere bakarken diğeri anlamsal bağlantıları yakalıyor, bir başkası uzak mesafedeki tokenları ilişkilendiriyor. Bu hesaplamalar paralel yapıldığı için performans kaybı da olmuyor. Sonuçta tüm kafaların çıktıları birleştirilip tek bir temsile dönüştürülüyor.

**Örnek:** 8 kafalı bir Multi-Head Attention'da 1. kafa "o" zamirinin öncülünü bulmaya, 2. kafa fiil-nesne ilişkisine, 3. kafa zaman uyumuna bakabilir.

```python
multihead_attn = nn.MultiheadAttention(
    embed_dim=512, 
    num_heads=8,
    dropout=0.1
)
# embed_dim num_heads'e tam bölünmeli: 512 / 8 = 64 boyut/kafa
```

**İlişkili:** Attention, Self-Attention, Transformer

---

### Encoder (Kodlayıcı)
**Ne işe yarar:** Girdiyi işleyip zengin bir iç temsile dönüştüren Transformer bileşeni. Girdinin tamamını görür ve bağlamı anlamaya çalışır.

**Detaylı:** Encoder katmanı, girdi dizisini alır ve her bir token için bağlamsal bir gömmeye dönüştürür. BERT gibi modeller sadece encoder kullanır. Encoder'da her katman iki ana bloktan oluşur: self-attention ve feed-forward sinir ağı. Her bloğun ardından residual bağlantı ve layer normalization var. Encoder'ın güzel tarafı, çift yönlü (bidirectional) olması — yani bir kelimeyi anlamak için sağındaki ve solundaki tüm kelimelere bakabiliyor.

**Örnek:** "Yıkanmaz" kelimesindeki "yıka-" fiil kökünü ve "-n-" edilgen ekini ayırt etmek gibi düşünün. Encoder hem "yıka"ya hem "-n"e hem "-maz"a aynı anda bakar.

```python
encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# Girdi tensörü: (seq_len, batch, feature)
encoder_output = encoder(input_embeddings)
```

**İlişkili:** Decoder, Transformer, BERT, Encoder-Decoder

---

### Decoder (Kod Çözücü)
**Ne işe yarar:** Encoder'ın ürettiği temsili kullanarak adım adım çıktı üreten Transformer bileşeni. Otoregresif yapıda çalışır — yani her adımda bir önceki ürettiğini de girdi olarak alır.

**Detaylı:** Decoder'da iki tane attention katmanı var. İlki masked self-attention — bu katman sadece daha önce üretilmiş tokenları görebiliyor, gelecekteki tokenlara bakamıyor (çünkü bunlar henüz üretilmedi). İkincisi cross-attention — encoder'ın çıktısına bakarak girdiyle bağlantı kuruyor. GPT serisi decoder-only modellerdir, yani encoder'ları yoktur. Onlarda cross-attention da bulunmaz, sadece masked self-attention vardır.

**Örnek:** "Merhaba, nasılsın?" çıktısını üreten bir decoder önce "Merhaba"yı, sonra "Merhaba," yı, sonra "Merhaba, nasılsın?"ı üretir. Her adımda kendi ürettiklerini de görür.

```python
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

# memory encoder'dan gelen çıktı
output = decoder(tgt_embeddings, memory=encoder_output)
```

**İlişkili:** Encoder, Cross-Attention, Masked Self-Attention, GPT

---

### Cross-Attention (Çapraz Dikkat)
**Ne işe yarar:** Decoder'ın encoder'ın çıktısına odaklanmasını sağlayan attention türü. İki farklı dizi arasındaki ilişkiyi kurar.

**Detaylı:** Encoder-Decoder mimarisinde decoder, kendi içinde self-attention yaptıktan sonra cross-attention ile encoder'ın son katman çıktısına bakıyor. Burada sorgular (Q) decoder'dan, anahtarlar (K) ve değerler (V) encoder'dan geliyor. Yani decoder "Şimdiye kadar şunları ürettim, girdiye göre sıradaki en uygun token ne olmalı?" sorusuna cevap arıyor. Çeviri modellerinde bu çok kritik — Türkçe çıktı üretirken İngilizce girdinin hangi kısmına bakacağını cross-attention belirliyor.

**Örnek:** "I love you" girdisi için decoder "Seni seviyorum" üretirken cross-attention "Seni" → "you", "seviyorum" → "love" bağlantısını kurar.

```python
cross_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8)
# Q: decoder'dan, K,V: encoder'dan
decoder_state = cross_attn(
    query=decoder_output,  
    key=encoder_output,    
    value=encoder_output   
)
```

**İlişkili:** Self-Attention, Encoder, Decoder, Transformer

---

### Positional Encoding (Konumsal Kodlama)
**Ne işe yarar:** Transformer'a kelimelerin sırasını öğreten mekanizma. Self-attention tek başına sırayı bilmediği için bu bilginin eklenmesi gerekir.

**Detaylı:** Self-attention tüm tokenları aynı anda gördüğü için "Köpek kediyi kovaladı" ile "Kedi köpeği kovaladı" arasındaki farkı anlayamaz — ikisinde de aynı kelimeler var. Positional encoding tam bu noktada devreye giriyor. Her token'a konumunu belirten bir sinüs/kosinüs sinyali veya öğrenilebilir bir vektör ekleniyor. Bu sayede model kelimelerin cümle içindeki sırasını da biliyor. Modern modellerde genellikle RoPE (Rotary Position Embedding) gibi daha gelişmiş yöntemler kullanılıyor.

**Örnek:** "Ali Ayşe'yi gördü" ile "Ayşe Ali'yi gördü" arasındaki farkı anlamak — positional encoding olmadan bu iki cümle model için aynı olurdu.

```python
import math

def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angles = pos / (10000 ** (2 * i / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe
```

**İlişkili:** Transformer, Self-Attention, RoPE

---

### Feed-Forward Network (FFN / İleri Beslemeli Ağ)
**Ne işe yarar:** Attention katmanından sonra gelen, token bazında işlem yapan sinir ağı katmanı. Modele doğrusal olmayan dönüşümler ve ek öğrenme kapasitesi kazandırır.

**Detaylı:** Her attention katmanının ardından her token'a ayrı ayrı uygulanan iki katmanlı bir sinir ağı. Genelde d_model → 4*d_model → d_model şeklinde genişleyip daralan bir yapıdadır. Aradaki ReLU veya GELU aktivasyon fonksiyonu doğrusal olmayanlığı sağlıyor. Yani her token bağımsız olarak bu ağdan geçiyor — tokenlar arasında bilgi paylaşımı olmuyor, o işi attention katmanı yapmış oluyor. FFN'nin boyutu (genelde 4 kat) modelin parametre sayısını büyük ölçüde belirliyor.

**Örnek:** GPT-3'te d_model=12288 ve FFN boyutu 4*12288=49152 — yani parametrelerin çoğu FFN katmanlarında.

```python
ffn = nn.Sequential(
    nn.Linear(512, 2048),  # Genişlet
    nn.GELU(),              # Aktivasyon
    nn.Linear(2048, 512),   # Daralt
    nn.Dropout(0.1)
)
```

**İlişkili:** Transformer, Attention, Aktivasyon Fonksiyonları

---

### Residual Connection (Artık Bağlantı)
**Ne işe yarar:** Katmanlar arasında bilginin doğrudan akmasını sağlayarak derin ağların eğitimini kolaylaştıran teknik. Vanishing gradient sorununu büyük ölçüde çözer.

**Detaylı:** Her Transformer katmanında attention veya FFN çıktısı, katmanın girdisine eklenir: `output = layer(x) + x`. Yani katman sadece "farkı" öğrenmek zorunda kalıyor. Bu sayede 100 katmanlı bir model bile sorunsuz eğitilebiliyor. Ayrıca gradyanlar geri yayılırken artık bağlantı sayesinde kısa yol bulup doğrudan ilk katmanlara ulaşabiliyor. Transformer'da her alt katmandan sonra residual + layer normalization uygulanır.

**Örnek:** 50 katlı bir binada asansör bozulsa bile merdivenler var — residual connection o merdiven gibi. Gradyanlar bu kısa yoldan geçip kaybolmadan ilk katmanlara ulaşabiliyor.

```python
def transformer_sub_layer(x, sub_layer):
    # Residual bağlantı: çıktıya girdiyi ekle
    return x + sub_layer(x)

# Normalde sub_layer attention veya FFN oluyor
# Ardından layer normalization geliyor
```

**İlişkili:** Layer Normalization, Transformer, Vanishing Gradient

---

### Layer Normalization (Katman Normalizasyonu)
**Ne işe yarar:** Her bir token'ın temsilini ölçeklendirip normalize ederek eğitimi kararlı hale getiren teknik.

**Detaylı:** Batch normalization'dan farklı olarak layer normalization, batch boyunca değil token bazında çalışıyor. Yani her token için tüm öznitelik boyutları boyunca ortalama ve varyans hesaplanıyor. Bu sayede batch boyutundan bağımsız çalışabiliyor. Transformer'larda her alt katmanın (attention veya FFN) ardından residual + layer normalization şeklinde uygulanıyor. Modern modellerde "Pre-LN" yani ön normalizasyon daha yaygın — normalizasyon önce yapılıyor, sonra attention/FFN'ye giriliyor.

**Örnek:** Bir token'ın 512 boyutlu vektöründe her boyut farklı ölçeklerde olabilir. Layer normalization bunları standart normal dağılıma çeker.

```python
ln = nn.LayerNorm(512)  # 512 = feature dimension
normalized = ln(token_embedding)
# Her token bağımsız normalize edilir: ort=0, varyans=1
```

**İlişkili:** Residual Connection, Batch Normalization, Transformer

---

### Gating Mechanism (Geçit Mekanizması)
**Ne işe yarar:** Bilginin ne kadarının geçeceğini, ne kadarının unutulacağını kontrol eden mekanizma. LSTM'lerden Transformer'daki MoE'ye kadar birçok yapıda kullanılır.

**Detaylı:** En basit haliyle bir sigmoid fonksiyonundan geçen ağırlıklar sayesinde çalışıyor — çıktı 0-1 arasında olduğu için "bilgiyi ne kadar geçireyim?" sorusuna cevap veriyor. LSTM'lerde input, forget, output olmak üzere üç geçit var. Transformer tabanlı modellerde ise gating mekanizmaları daha çok MoE (Mixture of Experts) yapılarında karşımıza çıkıyor — her token sadece ilgili uzman ağlara yönlendiriliyor.

**Örnek:** LSTM'de "Cümle uzadı, baştaki özneyi unutma" dediğimiz yer forget gate.

```python
# LSTM'deki forget gate
forget_gate = torch.sigmoid(
    torch.mm(hidden_state, W_f) + torch.mm(input, U_f) + b_f
)
# 0'a yakınsa unut, 1'e yakınsa hatırla
```

**İlişkili:** LSTM, MoE, Attention, Transformer
