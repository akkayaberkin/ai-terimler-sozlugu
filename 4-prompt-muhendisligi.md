# Gün 4: Prompt Mühendisliği

Sormak, bir modelden en iyi cevabı almanın yarısıdır. Bu bölümde doğru soruyu sormanın, modeli yönlendirmenin ve en iyi çıktıyı almak için kullanılan teknikleri anlatıyorum.

---

### Prompt Engineering (Prompt Mühendisliği)
**Ne işe yarar:** Bir dil modeline istediğimiz cevabı almak için soruyu, talimatı ve bağlamı en verimli şekilde tasarlama sanatı.

**Detaylı:** Modelin kafasının içinde dev bir bilgi kütüphanesi var ama bu kütüphaneye nasıl soru sorduğunuz çok önemli. Prompt engineering, bir soruyu sormanın en doğru yolunu bulmakla ilgili. Mesela "Şunu yap" demekle "Şöyle bir senaryoda, şu rolde, bu formatta yap" demek arasında dağlar kadar fark var. Doğru prompt yazmak deneme yanılma gerektiriyor — kimi zaman bir kelimenin yeri bile sonucu değiştirebiliyor. Bu alanda uzmanlaşmak aslında bir çeşit iletişim becerisi: modele ne kadar net ve detaylı talimat verirseniz, o kadar iyi sonuç alıyorsunuz. Son zamanlarda "prompt engineer" diye ayrı bir iş pozisyonu bile var, hatta buralarda maaşlar 150K USD'ye kadar çıkıyor.

**Örnek:** "Bir hikaye yaz" ile "Polisiye türünde, İstanbul'da geçen, 30 yaşında bir dedektifin ilk vakasını anlatan, 500 kelimelik bir hikaye yaz" arasındaki farkı tahmin edersiniz.

```
# Kötü prompt:
"Bana Python öğret"

# İyi prompt:
"Ben hiç kod yazmamış biriyim. Bana Python'da değişken tanımlamayı,
for döngüsünü ve liste işlemlerini adım adım, en basit örneklerle anlat.
Her konsept için 1-2 satırlık çalışan kod örneği ver."
```

**İlişkili:** System Prompt, Few-shot Prompting, Zero-shot Prompting, Prompt Chaining

---

### System Prompt (Sistem Promptu)
**Ne işe yarar:** Modele bir kişilik, rol veya davranış kuralları seti vermek için kullanılan, kullanıcı mesajlarından ayrı olarak tanımlanan başlangıç talimatı.

**Detaylı:** ChatGPT ve benzeri uygulamalarda arkada çalışan sistem promptu, kullanıcının görmediği ama modelin her cevabını etkileyen bir gizli talimat gibidir. Bu prompt modelin nasıl davranacağını belirler: hangi dilde konuşacağı, ne kadar detaylı cevap vereceği, hangi konularda konuşup konuşamayacağı gibi pek çok şey sistem promptuyla belirlenir. Örneğin bir müşteri hizmetleri botu için "Her zaman kibar ol, asla müşteriyle tartışma, sorunu çözmek için en fazla 3 adım öner" şeklinde bir sistem promptu yazılabilir. Sistem promptu kullanıcı mesajından önce geldiği için modelin dikkatini ve önceliklerini belirlemede çok etkilidir. İyi yazılmış bir sistem promptu, modelin binlerce satır kod gibi karmaşık bir işi yaparken tutarlı kalmasını sağlar.

**Örnek:** Bir yemek tarifi asistanı düşünün:

```
Sistem: Sen profesyonel bir şef asistanısın.
- Her zaman Türkçe konuş
- Malzemeleri önce liste halinde yaz
- Adımları numaralandır
- Porsiyon sayısını belirt
- Sağlık uyarısı gerekiyorsa ekle
- Asla alkollü tarif önerme
```

```
# API ile kullanım
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Sen bir seyahat danışmanısın. Önce bütçeyi sor, sonra öneri yap."},
        {"role": "user", "content": "Tatil planlamak istiyorum"}
    ]
)
```

**İlişkili:** Prompt Engineering, Role Prompting, Context Window, Instruction Tuning

---

### Few-shot Prompting (Az Örnekle Yönlendirme)
**Ne işe yarar:** İstediğiniz çıktının formatını modele birkaç örnek göstererek öğretme tekniği. Ek eğitime gerek yok, hepsi prompt'un içinde.

**Detaylı:** Modelin ne istediğinizi anlaması bazen sadece talimatla olmuyor. Few-shot prompting'de talimatın yanına 2-5 arası örnek koyuyorsunuz, model de bu kalıbı görüp aynı şekilde devam ediyor. Bu özellikle çıktı formatının çok spesifik olduğu durumlarda işe yarıyor — mesela JSON formatı, tablo, liste, şiir gibi özel yapılar istediğinizde. Örneklerin kalitesi çok önemli: kötü örnekler modeli yanıltırken, iyi seçilmiş örnekler neredeyse fine-tuning kalitesinde sonuç verebiliyor. Ayrıca örnek sayısı arttıkça prompt'un token sayısı da artıyor, yani maliyet ve gecikme artıyor. Bu yüzden en az örnekle en iyi sonucu almak bir beceri haline geliyor.

**Örnek:** İngilizce cümleleri Türkçe'ye çeviren bir prompt:

```
İngilizce: "The cat is on the table"
Türkçe: "Kedi masanın üstünde"

İngilizce: "I have been waiting for hours"
Türkçe: "Saatlerdir bekliyorum"

İngilizce: "She will arrive tomorrow morning"
Türkçe: "Yarın sabah varacak"

İngilizce: "They were playing football when it started raining"
Türkçe:
```

```python
def few_shot_translate(text, examples):
    """Verilen örneklerle çeviri yap"""
    prompt = "İngilizce cümleleri Türkçe'ye çevir:\n\n"
    for ing, tr in examples:
        prompt += f"İngilizce: \"{ing}\"\nTürkçe: \"{tr}\"\n\n"
    prompt += f"İngilizce: \"{text}\"\nTürkçe: "
    return prompt
```

**İlişkili:** Zero-shot Prompting, In-context Learning, Prompt Engineering, Few-shot Learning

---

### Zero-shot Prompting (Sıfır Örnekle Yönlendirme)
**Ne işe yarar:** Hiçbir örnek vermeden, sadece talimatla modelden istenen görevi yapmasını isteme tekniği.

**Detaylı:** Few-shot'ın aksine burada hiç örnek yok. Sadece "Bunu yap" diyorsunuz, model de zaten eğitildiği veriler sayesinde ne istediğinizi anlıyor. Modern büyük dil modelleri (GPT-4, Claude 3, Gemini gibi) zero-shot'ta şaşırtıcı derecede başarılı. Bunun sebebi bu modellerin o kadar büyük ve kapsamlı veriyle eğitilmiş olması ki, neredeyse her talimat formatını görmüş olmaları. Yine de zero-shot her zaman işe yaramayabilir — çok spesifik formatlar veya domain bilgisi gerektiren görevlerde few-shot daha iyi sonuç veriyor. Pratikte önce zero-shot'u denemek, olmazsa few-shot'a geçmek en akıllıca yol.

**Örnek:**

```
Şu metni özetle: "Yapay zeka son yıllarda inanılmaz bir hızla gelişiyor.
Özellikle büyük dil modelleri sayesinde makineler artık insanlarla
doğal dilde iletişim kurabiliyor. Bu gelişmeler bir yandan heyecan
vericiyken diğer yandan etik soruları da beraberinde getiriyor."
```

Model şöyle bir özet çıkarabilir: "Yapay zeka hızla gelişiyor, dil modelleri insanlarla iletişimi mümkün kılıyor ancak etik sorular da doğuruyor."

```python
def zero_shot_summarize(text):
    prompt = f"Şu metni 1 cümlede özetle:\n\n{text}"
    return call_model(prompt)

# Çalışması için hiç örneğe gerek yok
```

**İlişkili:** Few-shot Prompting, Zero-shot Learning, In-context Learning

---

### Chain-of-Thought (CoT / Düşünce Zinciri)
**Ne işe yarar:** Modeli adım adım düşünmeye zorlayarak, özellikle mantık, matematik ve çok adımlı problemlerde doğruluk oranını artıran bir teknik.

**Detaylı:** Normalde bir modele "300 elma var, 120'sini sattım, kalanları 5 kasaya eşit koydum, her kasada kaç elma var?" diye sorarsanız direkt cevap vermeye çalışır ve çoğu zaman yanlış yapar. Ama "Adım adım düşünelim..." derseniz, model önce "300 - 120 = 180 elma kaldı", sonra "180 / 5 = 36" diye gider ve doğru cevaba ulaşır. CoT, bu adım adım düşünme sürecini prompt'a ekleyerek modelin de aynı mantığı izlemesini sağlıyor. 2022'de Google'daki araştırmacıların yayınladığı bir makaleyle popüler oldu. O kadar etkili ki, matematik problemlerinde doğruluk %18'den %79'a fırlayabiliyor. CoT özellikle büyük modellerde (100B+ parametre) işe yarıyor, küçük modellerde pek faydası olmuyor.

**Örnek:** Klasik bir mantık sorusu:

```
Soru: Bir çiftlikte tavşanlar ve tavuklar var. Toplam 20 hayvan ve 56 bacak var.
Kaç tavşan ve kaç tavuk var?

Adım adım düşünelim:
1. Her tavşan 4 bacaklı, her tavuk 2 bacaklı.
2. Toplam hayvan sayısı: T + K = 20
3. Toplam bacak sayısı: 4T + 2K = 56
4. İlk denklemden K = 20 - T
5. İkinci denklemde yerine koy: 4T + 2(20 - T) = 56
6. 4T + 40 - 2T = 56
7. 2T = 16
8. T = 8 tavşan
9. K = 20 - 8 = 12 tavuk
```

```python
# CoT prompt'u otomatik ekleme
def chain_of_thought_prompt(question):
    return f"""Soru: {question}

Adım adım düşünelim:
"""
# Model bu prompt ile çağrılınca CoT yapar
```

**İlişkili:** Tree-of-Thoughts, ReAct, Self-Consistency, Reasoning

---

### Tree-of-Thoughts (ToT / Düşünce Ağacı)
**Ne işe yarar:** Modelin tek bir düşünce zinciri yerine, birden fazla olasılığı paralel olarak değerlendirip en iyi yolu seçmesini sağlayan ileri düzey bir akıl yürütme tekniği.

**Detaylı:** CoT'de model tek bir çizgide düşünür — bir yanlış adım tüm çözümü batırır. ToT ise bir satranç oyuncusu gibi çalışır: her adımda birkaç olası hamle üretir, her hamlenin sonucunu değerlendirir ve en umut verici yoldan devam eder. Model adeta bir ağaç yapısı oluşturur: her düğüm bir düşünce, her dal farklı bir çözüm yolu. BFS (genişlik öncelikli arama) veya DFS (derinlik öncelikli arama) ile bu ağaçta gezinebilir. Her düğümde model kendine "Bu mantıklı mı?" diye sorar ve gereksiz dalları budar. ToT, karmaşık planlama gerektiren görevlerde, matematik problemlerinde ve yaratıcı yazımda CoT'den çok daha iyi sonuç veriyor. Tabii maliyeti de yüksek — her adımda modeli defalarca çağırmanız gerekiyor.

**Örnek:** 24 oyunu (4 sayıyla 24'e ulaşma) gibi bir problemde ToT şöyle çalışır:

```
Başlangıç: [3, 5, 7, 9]

Adım 1 - Olası işlemler:
├─ 3 + 5 = 8 → kalan: [7, 8, 9]  (Değerlendirme: umut verici)
├─ 3 × 5 = 15 → kalan: [7, 9, 15]  (Değerlendirme: umut verici)
├─ 9 - 7 = 2 → kalan: [2, 3, 5]  (Değerlendirme: çok umut verici)
├─ ...

Adım 2 - En iyi dal: 9 - 7 = 2 [2, 3, 5]
├─ 2 × 3 = 6 → kalan: [5, 6]  (bir adım kaldı!)
├─ 5 - 2 = 3 → kalan: [3, 3]  (devam eder)

Adım 3 - 6 × 5 = 30 olmaz... 6 × 5 - 6 = 24 ✗
Devam: 2 × 3 = 6, kalan [5, 6]
5 × 6 - 6 = 24 ✗
Hmm diğer dallara dön...
```

```
# ToT'nin basitleştirilmiş mantığı:
# Her adımda N tane olası düşünce üret, değerlendir, en iyilerini seç
for step in range(max_steps):
    thoughts = generate_thoughts(current_state, N=3)
    scores = evaluate_thoughts(thoughts)
    best_thought = select_best(thoughts, scores, k=2)
    current_state = expand(best_thought)
```

**İlişkili:** Chain-of-Thought, ReAct, BFS, DFS, Arama Algoritmaları

---

### ReAct (Reasoning + Acting / Akıl Yürütme + Eylem)
**Ne işe yarar:** Modelin hem düşünüp (reasoning) hem de dış araçları kullanarak (acting) adım adım bir problemi çözmesini sağlayan framework.

**Detaylı:** ReAct aslında iki yeteneği birleştiriyor: CoT'den gelen adım adım düşünme ile bir ajan gibi dış dünyayla etkileşim. Model önce "Şimdi ne yapmalıyım?" diye düşünüyor, sonra bir eylem yapıyor (örneğin bir API çağrısı, Wikipedia sorgusu, hesap makinesi kullanımı), gelen sonucu gözlemliyor ve tekrar düşünüyor. Bu döngü sayesinde model sadece hafızasındaki bilgiyle yetinmiyor, gerektiğinde dış kaynaklara başvurabiliyor. En bilinen örneği LangChain ajanları. Kendi başına çalışan bir model yanlış cevap verebilir ama ReAct ile model "Dur, bu bilgiyi internette arayayım" diyerek hatasını düzeltebiliyor. Özellikle güncel bilgi gerektiren sorular ve çok adımlı görevler için birebir.

**Örnek:**

```
Soru: Elon Musk'ın doğduğu yıl Türkiye'deki cumhurbaşkanı kimdi?

Düşünce: Önce Elon Musk'ın doğum yılını öğrenmeliyim.
Eylem: wikipedia['Elon Musk']
Gözlem: Elon Musk 28 Haziran 1971'de doğdu.

Düşünce: 1971'de Türkiye'de cumhurbaşkanı kimmiş bakalım.
Eylem: wikipedia['Türkiye cumhurbaşkanları listesi']
Gözlem: 1971 yılında Türkiye Cumhurbaşkanı Cevdet Sunay'dı.

Düşünce: Şimdi cevabı verebilirim.
Cevap: Elon Musk'ın doğduğu yıl Türkiye Cumhurbaşkanı Cevdet Sunay'dı.
```

```python
# ReAct döngüsü basitleştirilmiş hali
def react_loop(question):
    max_steps = 5
    context = f"Soru: {question}\n"
    
    for _ in range(max_steps):
        # Modelden düşünce veya eylem üretmesini iste
        response = model.generate(context)
        
        if "Eylem:" in response:
            # Eylemi parse et ve çalıştır
            action = parse_action(response)
            result = execute_action(action)
            context += f"\nGözlem: {result}\n"
        elif "Cevap:" in response:
            return parse_answer(response)
    
    return "Çözülemedi"
```

**İlişkili:** Chain-of-Thought, Tool Use, Agent, LangChain, Tree-of-Thoughts

---

### Structured Output (Yapılandırılmış Çıktı)
**Ne işe yarar:** Modelin serbest metin yerine önceden belirlenmiş bir formatta (JSON, XML, tablo, CSV) çıktı üretmesini sağlayan teknik.

**Detaylı:** Dil modelleri doğal olarak serbest metin üretir. Ama bir uygulamada bu çıktıyı kullanmak istiyorsanız — mesela bir JSON'ı parse edip veritabanına yazacaksanız — modelin tutarlı bir formatta çıktı vermesi gerekir. Structured output bunun için var. Modele çıktının şemasını (schema) veriyorsunuz, o da birebir o şemaya uygun çıktı üretiyor. En son modellerde (GPT-4o, Claude 3.5 gibi) artık native JSON mode var — yani modelin geçersiz JSON üretmesi neredeyse imkansız. Eskiden regex veya başka yöntemlerle çıktıyı zorlamak gerekiyordu. Bazı kütüphaneler (Outlines, Jsonformer, Instructor) model çıktısını anında kontrol edip hatalıysa yeniden ürettiriyor. Structured output özellikle API development, veri çıkarma, form doldurma gibi senaryolarda hayat kurtarıcı.

**Örnek:** Bir faturadan bilgi çıkarma:

```
Şu faturadan bilgileri JSON olarak çıkar:
"ABC Şirketi, 15 Mart 2024, Ürün: Laptop (25,000 TL), Masa (3,500 TL),
KDV %20, Toplam: 34,200 TL"

{
  "firma": "ABC Şirketi",
  "tarih": "2024-03-15",
  "kalemler": [
    {"urun": "Laptop", "birim_fiyat": 25000, "adet": 1},
    {"urun": "Masa", "birim_fiyat": 3500, "adet": 1}
  ],
  "kdv_orani": 20,
  "kdv_tutari": 5700,
  "toplam": 34200
}
```

```python
import json
from pydantic import BaseModel

# Önce çıktı şemasını tanımla
class Fatura(BaseModel):
    firma: str
    tarih: str
    kalemler: list
    toplam: float

prompt = f"""Şu metni verilen JSON şemasına göre çıkar:
{metin}

Şu formatta JSON:
{json.dumps(Fatura.model_json_schema(), indent=2)}

JSON:"""
```

**İlişkili:** JSON Mode, Function Calling, Schema, Pydantic

---

### Self-Consistency (Kendi Kendine Tutarlılık)
**Ne işe yarar:** Aynı soruyu modele birden fazla kez sorup (biraz randomness ile) en çok tekrar eden cevabı seçerek doğruluk oranını artırma tekniği.

**Detaylı:** Bir problemin çözümünde model bazen doğru, bazen yanlış cevap verebiliyor. Self-consistency'nin fikri çok basit: aynı soruyu modele 5-10 kere soruyorsunuz (temperature'ı 0.3-0.7 arası bir değere ayarlayarak), aldığınız cevapları grupluyorsunuz ve en sık tekrar eden cevabı seçiyorsunuz. Bu, bir sınavda aynı soruyu 10 kişiye sorup en çok verilen cevabı doğru kabul etmek gibi bir şey. CoT ile birlikte kullanıldığında çok etkili oluyor — her bir CoT zinciri farklı bir yol izleyebilir, ama doğru cevap genelde aynı yere çıkar. Araştırmalara göre self-consistency, matematik problemlerinde CoT'yi tek başına kullanmaktan %15-20 daha iyi sonuç veriyor. Tabii maliyeti var: her soru için N kat fazla API çağrısı yapmanız gerekiyor.

**Örnek:**

```
Soru: Bir otobüste 15 yolcu var. İlk durakta 3 kişi bindi, 5 kişi indi.
İkinci durakta 7 kişi bindi, 2 kişi indi. Şu anda kaç yolcu var?

Deneme 1 (CoT ile): 15 + 3 - 5 + 7 - 2 = 18 ✅
Deneme 2 (CoT ile): 15 + 3 = 18, 18 - 5 = 13, 13 + 7 = 20, 20 - 2 = 18 ✅
Deneme 3 (CoT ile): 15 - 5 = 10, 10 + 3 = 13, 13 + 7 = 20, 20 - 2 = 18 ✅
Deneme 4 (CoT ile): 15 + 3 - 5 + 7 - 2 = 20 ❌ (hesap hatası)

Sonuç: 18 (4 denemeden 3'ü 18 dedi → doğru kabul edilir)
```

```python
import statistics
from collections import Counter

def self_consistency(question, n_samples=5, temperature=0.5):
    answers = []
    for _ in range(n_samples):
        response = call_model(question, temperature=temperature)
        answer = extract_answer(response)
        answers.append(answer)
    
    # En sık tekrar eden cevabı seç
    most_common = Counter(answers).most_common(1)[0][0]
    return most_common

# CoT ile birleştirince:
# Soruyu 5 kere sor, her seferinde adım adım düşünmesini iste
# En çok tekrar eden final cevabı seç
```

**İlişkili:** Chain-of-Thought, Temperature Sampling, Majority Voting, Ensemble

---

### Prompt Chaining (Prompt Zincirleme)
**Ne işe yarar:** Karmaşık bir görevi, her biri bir önceki adımın çıktısını alan birden fazla küçük prompt'a bölerek çözme tekniği.

**Detaylı:** Bazen bir görev o kadar karmaşıktır ki tek bir prompt'a sığdırmak imkansızdır — ya context window aşılır, ya da modelin dikkati dağılır. Prompt chaining'de büyük görevi küçük, yönetilebilir adımlara bölüyorsunuz. Her adımda modeli ayrı ayrı çağırıyorsunuz, bir adımın çıktısı bir sonraki adımın girdisi oluyor. Örneğin bir blog yazısı yazdırmak için: (1) Konuyu analiz et → (2) Ana başlıkları çıkar → (3) Her başlık için içerik yaz → (4) Yazıyı düzenle/formatla → (5) Özet çıkar. Her adım ayrı bir prompt ve model çağrısı. Bu yaklaşım sayesinde her adımda modele daha odaklı talimat verebiliyor, hataları daha kolay tespit edip düzeltebiliyorsunuz. LangChain, LlamaIndex gibi framework'ler bu işi otomatikleştiriyor.

**Örnek:** Bir müşteri yorumundan aksiyon maddeleri çıkarma:

```
Adım 1 - Duygu Analizi:
Prompt: "Şu yorumun duygusunu belirle (Pozitif/Negatif/Nötr):
'Ürün güzel ama kargo çok geç geldi, ambalaj da yırtıktı.'"

Adım 2 - Kategorize Et:
Prompt: "Yorumun kategorisi nedir? (Ürün Kalitesi / Kargo / Müşteri Hizmetleri):
'Negatif: Ürün güzel ama kargo sorunu var.'"

Adım 3 - Aksiyon Üret:
Prompt: "Bu kategori ve duyguya göre hangi aksiyon alınmalı?
Kategori: Kargo, Duygu: Negatif
Önerilen aksiyon nedir? (İade teklif et, özür dile, kargo ücretini iade et vs.)"
```

```python
def prompt_chain_pipeline(review):
    # Adım 1: Duygu analizi
    sentiment = call_model(f"Şu yorumun duygusu nedir:\n{review}")
    
    # Adım 2: Kategori
    category = call_model(
        f"Yorum kategorisi?\nYorum: {review}\nDuygu: {sentiment}"
    )
    
    # Adım 3: Aksiyon önerisi
    action = call_model(
        f"Duygu: {sentiment}\nKategori: {category}\nÖnerilen aksiyon:"
    )
    
    return {"sentiment": sentiment, "category": category, "action": action}

# Her adım ayrı model çağrısı, ayrı prompt
```

**İlişkili:** Prompt Engineering, LangChain, Pipeline, Agent, Workflow
