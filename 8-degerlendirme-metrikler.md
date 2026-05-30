# Gün 8: Değerlendirme & Metrikler

Bir model ne kadar iyi? Bunu ölçmek için metrikler var. Doğruluk oranı yetmez — bazen hangi hatayı yaptığın, nasıl yanlış yaptığından daha önemli. Bu bölümde modelleri objektif değerlendirmek için kullanılan temel metrikleri ve kavramları anlatıyorum.

---

### Accuracy (Doğruluk)
**Ne işe yarar:** Modelin toplam tahminlerinin ne kadarının doğru olduğunu ölçen temel metrik.

**Detaylı:** Accuracy = (Doğru Tahminler) / (Toplam Tahminler). Basit ve anlaşılır ama tek başına yanıltıcı olabilir. Mesela %99'u negatif olan bir spam verisinde, model her şeye "negatif" derse accuracy %99 çıkar — ama bir tane spam maili bile yakalamamıştır. Bu yüzden dengesiz veri setlerinde (imbalanced dataset) accuracy tek başına güvenilmez. Precision, Recall ve F1-Score ile birlikte değerlendirilmesi gerekir. İş hayatında "modelin accuracy'si kaç?" diye sorulduğunda, "genel accuracy %92 ama spesifik olarak X sınıfında recall düşük" gibi bir cevap daha anlamlı olur.

**Örnek:**
```python
from sklearn.metrics import accuracy_score

gercek = [0, 1, 0, 0, 1, 1, 0, 1, 0, 0]
tahmin = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]

acc = accuracy_score(gercek, tahmin)
print(f"Accuracy: {acc:.2f}")  # Accuracy: 0.80
```

**İlişkili:** Precision, Recall, F1-Score, Confusion Matrix, Balanced Accuracy

---

### Precision (Kesinlik)
**Ne işe yarar:** Modelin "pozitif" dediklerinin gerçekten ne kadarının pozitif olduğunu ölçer.

**Detaylı:** Precision = True Positive / (True Positive + False Positive). Yanlış alarmlardan (false positive) ne kadar kaçındığınızı gösterir. Mesela bir e-posta spam filtresinde precision yüksekse, spam olmayan bir maili spam olarak işaretleme olasılığınız düşüktür. Yanlış pozitiflerin maliyetli olduğu durumlarda (örneğin tıbbi teşhis, dolandırıcılık tespiti) precision önem kazanır. Model çok agresifse precision düşer — her şeye "spam" derse bir sürü önemli maili kaybedersiniz.

**Örnek:**
```python
from sklearn.metrics import precision_score

gercek = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
tahmin = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

prec = precision_score(gercek, tahmin)
print(f"Precision: {prec:.2f}")  # Precision: 0.80
```

**İlişkili:** Recall, F1-Score, False Positive, Spam Filtreleme

---

### Recall (Duyarlılık / Geri Çağırma)
**Ne işe yarar:** Gerçek pozitiflerin ne kadarını modelin doğru tespit ettiğini ölçer.

**Detaylı:** Recall = True Positive / (True Positive + False Negative). Kaçırdığınız pozitif örneklerin oranını gösterir. Mesela bir kanser tarama modelinde recall düşükse, hasta olan birini "sağlıklı" diye kaçırıyorsunuz demektir ki bu hayati bir hata. Yanlış negatiflerin (false negative) maliyeti yüksekse recall değerini yüksek tutmak isteriz. Ama recall'ı artırmak için modeli daha "hassas" yaparsanız bu sefer precision düşer. İşte bu ikilem precision-recall trade-off olarak bilinir.

**Örnek:**
```python
from sklearn.metrics import recall_score

gercek = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
tahmin = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

rec = recall_score(gercek, tahmin)
print(f"Recall: {rec:.2f}")  # Recall: 0.80
```

**İlişkili:** Precision, F1-Score, False Negative, Sensitivity, Tıbbi Teşhis

---

### F1-Score
**Ne işe yarar:** Precision ve Recall değerlerinin harmonik ortalamasını alarak tek bir skor üretir.

**Detaylı:** F1 = 2 * (Precision * Recall) / (Precision + Recall). Dengesiz veri setlerinde accuracy'den çok daha güvenilir bir metriktir. Hem precision'ı hem recall'ı aynı anda optimize etmek istediğinizde kullanılır. Örneğin bir metin sınıflandırma modelinde precision %90, recall %60 ise F1-Score %72 olur — bu model bazı pozitifleri kaçırıyor ama kaçırmadıklarında doğru tahmin ediyor demektir. F1 yüksekse, precision ve recall arasında sağlıklı bir denge var demektir.

**Örnek:**
```python
from sklearn.metrics import f1_score

gercek = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
tahmin = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

f1 = f1_score(gercek, tahmin)
print(f"F1-Score: {f1:.2f}")  # F1-Score: 0.80
```

**İlişkili:** Precision, Recall, Accuracy, Macro F1, Weighted F1

---

### Confusion Matrix (Karmaşıklık Matrisi)
**Ne işe yarar:** Modelin hangi sınıflarda ne kadar doğru/yanlış yaptığını tablo halinde gösterir.

**Detaylı:** Confusion matrix, gerçek değerlerle tahmin edilen değerleri karşılaştıran bir tablodur. İkili sınıflandırmada 4 hücresi vardır: True Positive (doğru pozitif), True Negative (doğru negatif), False Positive (yanlış alarm), False Negative (kaçırılan). Modelin sadece hangi oranda hata yaptığını değil, hatanın türünü de anlamanızı sağlar. Mesela bir modelin accuracy'si %90 olabilir ama confusion matrix'e baktığınızda tüm hataların "pozitifleri negatif sanma" yönünde olduğunu görebilirsiniz. Bu da size modeli iyileştirme stratejisi hakkında fikir verir.

**Örnek:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

gercek = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
tahmin = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

cm = confusion_matrix(gercek, tahmin)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.show()
"""
Çıktı:
        Tahmin: 0  1
Gerçek: 0   [4  1]
Gerçek: 1   [1  4]
"""
```

**İlişkili:** True Positive, False Positive, False Negative, Accuracy, Classification Report

---

### AUC-ROC
**Ne işe yarar:** Modelin sınıfları ayırma başarısını, eşik değerinden bağımsız olarak ölçen bir metriktir.

**Detaylı:** ROC eğrisi (Receiver Operating Characteristic), farklı eşik değerlerinde True Positive Rate ile False Positive Rate arasındaki ilişkiyi gösterir. AUC (Area Under Curve) ise bu eğrinin altında kalan alandır. 0.5 = rastgele tahmin (çöp model), 1.0 = mükemmel sınıflandırma. Genelde 0.8+ kabul edilebilir, 0.9+ mükemmel kabul edilir. AUC'nin güzel yanı, accuracy gibi bir eşik değerine bağlı olmamasıdır. Modelin çıktısı olasılık olduğu için herhangi bir threshold'da ne kadar iyi olduğunu ölçebilirsiniz.

**Örnek:**
```python
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

gercek = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
olasilik = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9, 0.4, 0.6, 0.85, 0.15])

auc = roc_auc_score(gercek, olasilik)
print(f"AUC: {auc:.3f}")  # AUC: 0.929
```

**İlişkili:** ROC Curve, True Positive Rate, False Positive Rate, Threshold, Binary Classification

---

### Perplexity
**Ne işe yarar:** Dil modellerinin ne kadar "şaşırdığını" ölçer — düşük perplexity, modelin bir sonraki tokeni daha iyi tahmin ettiği anlamına gelir.

**Detaylı:** Perplexity (PPL), bir dil modelinin bir metin dizisini ne kadar iyi modellediğini ölçen bir metriktir. Matematiksel olarak, her bir token'ın olasılığının geometrik ortalamasının tersidir. Basitçe söylemek gerekirse: model bir sonraki kelimeyi tahmin etmekte ne kadar "şaşkınsa" (düşük olasılık), perplexity o kadar yüksek olur. GPT-2'nin perplexity'si ~35 civarındayken, GPT-3'te bu ~20'ye düşmüştü. Perplexity her ne kadar model başarımını değerlendirmede yaygın bir metrik olsa da, tek başına modelin ne kadar "kullanışlı" olduğunu söylemez — ezber yapan bir model de düşük perplexity alabilir.

**Örnek:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

metin = "Yapay zeka günümüzde birçok alanda kullanılıyor."
inputs = tokenizer(metin, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    ppl = math.exp(loss)

print(f"Perplexity: {ppl:.2f}")
```

**İlişkili:** Cross-Entropy Loss, Language Modeling, Token Prediction, Entropy

---

### BLEU (Bilingual Evaluation Understudy)
**Ne işe yarar:** Makine çevirisi ve metin üretimi modellerinde, üretilen metnin referans metne ne kadar benzediğini ölçer.

**Detaylı:** BLEU skoru, modelin ürettiği metinle referans metin arasındaki n-gram (1-gram, 2-gram, 3-gram, 4-gram) örtüşmesine bakar. 0 ile 1 arasında bir değer alır. 0.3+ iyi, 0.5+ çok iyi, 0.7+ insan seviyesine yakın kabul edilir. BLEU'nun en büyük sorunu, anlamsal benzerliği değil, yüzeysel örtüşmeyi ölçmesidir. "Kedi matın üstünde" ile "Matın üstünde kedi var" cümleleri aynı anlamda olmasına rağmen BLEU skoru düşük çıkabilir. Ayrıca kısa cümlelerde cezalandırma (brevity penalty) mekanizması var ki bu da bazı durumlarda adil olmayan skorlara yol açabilir.

**Örnek:**
```python
from nltk.translate.bleu_score import sentence_bleu

referans = ["kedi matın üzerinde uyuyor".split()]
kaynak = "kedi matın üstünde uyuyor".split()

bleu = sentence_bleu(referans, kaynak)
print(f"BLEU: {bleu:.4f}")  # BLEU: 0.7636
```

**İlişkili:** ROUGE, METEOR, Machine Translation, n-gram Overlap

---

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
**Ne işe yarar:** Özetleme modellerini değerlendirmek için kullanılan, recall odaklı bir metriktir.

**Detaylı:** ROUGE, BLEU'nun aksine recall odaklıdır — yani referans özetteki kelimelerin ne kadarının modelin ürettiği özette geçtiğine bakar. Çeşitli türleri vardır: ROUGE-N (n-gram örtüşmesi), ROUGE-L (en uzun ortak alt dizi), ROUGE-S (skip-bigram). Özetleme sistemlerinde en yaygın kullanılan ROUGE-L'dir çünkü cümle yapısını da bir nebze dikkate alır. BLEU gibi ROUGE da yüzeysel bir metriktir — anlamı, özgünlüğü veya akıcılığı ölçmez. Bu yüzden özetleme modellerini değerlendirirken ROUGE yanında insan değerlendirmesi de şarttır.

**Örnek:**
```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
referans = "yapay zeka sağlık sektöründe teşhis koymaya yardımcı oluyor"
kaynak = "yapay zeka sağlıkta teşhis için kullanılıyor"

skorlar = scorer.score(referans, kaynak)
print(f"ROUGE-1: {skorlar['rouge1'].fmeasure:.3f}")  # ROUGE-1: 0.571
print(f"ROUGE-L: {skorlar['rougeL'].fmeasure:.3f}")  # ROUGE-L: 0.571
```

**İlişkili:** BLEU, Summarization, n-gram, ROUGE-1, ROUGE-L, ROUGE-N

---

### Hallucination Rate (Halüsinasyon Oranı)
**Ne işe yarar:** Modelin gerçekle uyuşmayan, hayali veya yanlış bilgi üretme oranını ölçer.

**Detaylı:** Hallucination, LLM'lerin var olmayan bilgileri sanki gerçekmiş gibi sunmasıdır. "2024 Fransa başbakanı şudur" diye sorduğunuzda, var olmayan birini çok kararlı bir dille anlatmaya başlaması klasik bir örnektir. Halüsinasyon oranını ölçmek için model çıktıları güvenilir bir kaynakla (Wikipedia, resmi dokümanlar) karşılaştırılır. Son araştırmalarda GPT-4'ün halüsinasyon oranı ~%15-20 civarında, GPT-3.5'te bu oran ~%30-40 seviyesinde. RAG (Retrieval-Augmented Generation) sistemleri halüsinasyonu önemli ölçüde azaltır çünkü model kaynak göstererek cevap verir. Halüsinasyon oranını sıfıra indirmek günümüz teknolojisiyle mümkün değil — LLM'ler doğası gereği "yaratıcı" modeller.

**Örnek:**
```python
# Basit bir halüsinasyon tespiti kontrolü
def hallucination_check(model_output, source_text):
    model_keywords = set(model_output.lower().split())
    source_keywords = set(source_text.lower().split())
    
    overlap = len(model_keywords & source_keywords)
    total = len(model_keywords)
    
    hallucination_ratio = 1 - (overlap / total) if total > 0 else 1.0
    return {
        "hallucination_orani": round(hallucination_ratio, 2),
        "ortak_kelime": len(model_keywords & source_keywords),
        "toplam_kelime": total,
        "durum": "⚠️ Muhtemel halüsinasyon var" if hallucination_ratio > 0.5 else "✅ Tutarlı"
    }

model_cikti = "Einstein 1945'te atom bombasını icat etti."
kaynak = "Einstein 1905'te görelilik teorisini yayınladı."
print(hallucination_check(model_cikti, kaynak))
# {'hallucination_orani': 0.6, 'ortak_kelime': 2, ...}
```

**İlişkili:** RAG, Factuality, Grounding, Model Confidence, Verifiability

---

### Mean Average Precision (mAP)
**Ne işe yarar:** Nesne tespiti (object detection) modellerinde, tespit edilen nesnelerin hem doğruluk hem sıralama başarımını tek bir skorda birleştirir.

**Detaylı:** mAP, özellikle COCO, PASCAL VOC gibi nesne tespiti verisetlerinde standart metrik haline gelmiştir. Önce her sınıf için precision-recall eğrisinin altındaki alan (Average Precision / AP) hesaplanır, sonra tüm sınıfların ortalaması alınır. mAP@0.5 (IoU eşiği 0.5) ve mAP@0.5:0.95 (farklı IoU eşiklerinin ortalaması) en yaygın kullanılan varyantlardır. mAP@0.5:0.95 daha katıdır çünkü bounding box'ın referansa tam oturmasını gerektirir. Modern modellerde mAP@0.5:0.95'in 0.5+ olması iyi kabul edilir.

**Örnek:**
```python
# mAP hesaplaması için torchmetrics kullanımı
from torchmetrics.detection import MeanAveragePrecision
import torch

# Örnek tahminler (format: [x1, y1, x2, y2, confidence, label])
preds = [dict(
    boxes=torch.tensor([[100, 150, 200, 300], [50, 50, 150, 200]]),
    scores=torch.tensor([0.95, 0.75]),
    labels=torch.tensor([0, 1]),
)]
# Gerçek değerler
target = [dict(
    boxes=torch.tensor([[110, 140, 210, 310], [45, 55, 145, 195]]),
    labels=torch.tensor([0, 1]),
)]

metrik = MeanAveragePrecision(iou_type="bbox")
metrik.update(preds, target)
sonuc = metrik.compute()
print(f"mAP@0.5:0.95: {sonuc['map']:.3f}")
print(f"mAP@0.5: {sonuc['map_50']:.3f}")
```

**İlişkili:** IoU, Object Detection, Precision-Recall Curve, COCO Metrics, Bounding Box
