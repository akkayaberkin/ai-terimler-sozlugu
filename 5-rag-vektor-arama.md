# Gün 5: RAG & Vektör Arama

Bilgiye erişmek kadar, o bilgiyi modele doğru anda ve doğru formda sunmak da büyük bir mesele. Bu bölümde modellerin eğitim verisine ihtiyaç duymadan güncel bilgiye nasıl ulaştığını, bunun için kullanılan indeksleme ve arama yöntemlerini anlatıyorum.

---

### RAG (Retrieval-Augmented Generation)
**Ne işe yarar:** Bir dil modeline, eğitilmediği konularda veya güncel bilgilerle ilgili soruları cevaplaması için harici bir kaynaktan ilgili bilgi parçalarını getirip bağlam olarak verme yöntemi.

**Detaylı:** Şöyle düşünün: bir dil modelinin bildikleri eğitim tarihinde donuyor. Örneğin 2023'te eğitilmiş bir modele 2025'teki bir olayı sorarsanız bilemez. RAG işte bu sorunu çözüyor. Önce soruyla ilgili belgeleri (PDF, web sayfası, veritabanı kaydı) bir vektör veritabanında arıyorsunuz, sonra bulduğunuz en alakalı parçaları modele "şu anda cevap verirken bunları da kullan" diyerek ekliyorsunuz. RAG'ın güzel tarafı, modeli yeniden eğitmek zorunda kalmamanız. Sadece belgeleri güncelleyerek cevapların kalitesini anında artırabiliyorsunuz. Halüsinasyonu da azaltıyor çünkü model artık havadan cevap uydurmak yerine size verdiğiniz kaynağa dayanarak konuşuyor. Kurumsal ortamlarda PDF, Slack, e-posta gibi birçok kaynağa bağlanarak şirket içi bilgiyi modele entegre etmek için en yaygın kullanılan yöntem bu.

**Örnek:** En basit haliyle LangChain üzerinden bir RAG zinciri şöyle kuruluyor:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Belgeleri vektör veritabanına yükle
vektor_db = FAISS.load_local("sirket_dokumanlari", OpenAIEmbeddings())

# RAG zinciri
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=vektor_db.as_retriever(search_kwargs={"k": 3}),
)

# Soru sor
cevap = qa_chain.invoke("Şirket izin politikası nedir?")
```

**İlişkili:** Embedding, Vector Database, Retriever, Reranker, Chunking

---

### Chunking (Parçalama / Belge Bölümleme)
**Ne işe yarar:** Uzun belgeleri, RAG sisteminin işleyebileceği ve anlamlı bir şekilde indeksleyebileceği küçük parçalara ayırma işlemi.

**Detaylı:** Bir romanın tamamını alıp embedding çıkarmaya kalksanız hem teknik limitlere takılırsınız (çoğu modelin token limiti var) hem de bir soru sorduğunuzda tüm kitabın embedding'i o soruyla ilgili olmaz. İşte chunking burada devreye giriyor. Belgeyi paragraflara, cümlelere veya anlamlı bölümlere ayırıyorsunuz. Ama bu iş sandığınız kadar basit değil — çok küçük chunk'lar bağlamı kaybettiriyor, çok büyük chunk'lar da gereksiz bilgi getiriyor. En yaygın yöntem sabit boyutlu chunk'ları overlap (üst üste binme) ile bölmek. Bir de anlamlı bölünme var: cümle ortasında kesmemek için nokta, paragraf sonu gibi doğal sınırları kullanmak. Semantic chunking'de ise konu değişimini yakalayıp ona göre bölüyorsunuz. Doğru chunking stratejisi, RAG sisteminizin başarısını ciddi ölçüde etkiliyor.

**Örnek:** LangChain'de farklı chunking yöntemleri:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Klasik recursive splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, # her parça yaklaşık 500 karakter
    chunk_overlap=50, # %10 overlap
    separators=["\n\n", "\n", ". ", " ", ""]
)

parcalar = splitter.split_text(uzun_belge)

# Semantik bölünme
semantic_splitter = SentenceTransformersTokenTextSplitter(
    chunk_size=256,
    chunk_overlap=32
)
```

**İlişkili:** RAG, Embedding Vector, Retriever, Vector Database

---

### Embedding Vector (Gömme Vektörü)
**Ne işe yarar:** Metin, resim veya ses gibi verileri, anlamsal benzerliklerini ölçebileceğimiz sayı dizilerine (vektörlere) dönüştürme işlemi.

**Detaylı:** Diyelim ki "kedi" ile "köpek" kelimelerini bir bilgisayara anlatmak istiyorsunuz. Harf olarak bakınca hiç benzemiyorlar. Ama anlam olarak ikisi de evcil hayvan. İşte embedding modelleri her kelimeyi veya cümleyi, o anlamı temsil eden bir sayı vektörüne çeviriyor. Bu vektörler öyle bir uzayda duruyor ki, benzer anlamlı kelimeler birbirine yakın, zıt anlamlılar uzak oluyor. Mesela "mutlu" ile "neşeli" vektörleri birbirine yakınken "mutlu" ile "üzgün" uzakta kalıyor. Modern embedding'ler 768 veya 1536 boyut gibi çok yüksek boyutlu oluyor. OpenAI'nin text-embedding-ada-002'si, Cohere'ın embed'i veya açık kaynak olarak BGE, E5 gibi modeller var. Embedding'ler RAG'ın temel taşı — çünkü bir soruyu aynı vektör uzayına gömüp, en yakın belgeleri bulmak işte bu sayede mümkün oluyor.

**Örnek:** Basit bir embedding çıkarımı:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# İki cümle arasındaki benzerliği hesapla
metin1 = "Bugün hava çok güzel"
metin2 = "Harika bir gün"

emb1 = client.embeddings.create(input=metin1, model="text-embedding-ada-002").data[0].embedding
emb2 = client.embeddings.create(input=metin2, model="text-embedding-ada-002").data[0].embedding

# Cosine similarity
benzerlik = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
print(f"Benzerlik: {benzerlik:.2f}")  # 1'e yakınsa anlamca yakın
```

Açık kaynak alternatifi (sentence-transformers):
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-large-v1.5')
vektor = model.encode("Bu cümle vektöre dönüşüyor")
print(len(vektor))  # 1024 boyutlu bir vektör
```

**İlişkili:** Cosine Similarity, Vector Database, RAG, ANN

---

### Vector Database (Vektör Veritabanı)
**Ne işe yarar:** Embedding vektörlerini depolayıp, bir sorgu vektörüne en yakın vektörleri çok hızlı bir şekilde bulan veritabanı türü.

**Detaylı:** Normal bir SQL veritabanı size "adı Ali olan kayıtları bul" dediğinizde çalışır ama "anlamca buna en yakın 5 belgeyi bul" dediğinizde çuvallar. Vektör veritabanları işte bu sorunu çözmek için var. Her vektörün yanında orijinal metni de saklarlar, böylece en yakın vektörleri bulduktan sonra onların metinlerini size geri verirler. Popüler vektör veritabanları arasında Pinecone (bulut tabanlı, yönetilen), Weaviate, Qdrant, Milvus ve bir de Chroma var (hafif, local'de çalışır). PostgreSQL'e pgvector eklentisiyle vektör desteği eklemek de mümkün — zaten çoğu sistemde belgelerin kendisi normal bir veritabanında dururken, embedding'ler ayrı bir sütunda tutuluyor. Vektör veritabanları genelde ANN (Approximate Nearest Neighbor) algoritmaları kullanarak milyonlarca vektör arasında milisaniyeler içinde arama yapabiliyor. Seçim yaparken dikkat edilmesi gerekenler: hız, doğruluk, ölçeklenebilirlik ve maliyet.

**Örnek:** Chroma ile vektör veritabanına yazma ve arama:

```python
import chromadb
from chromadb.utils import embedding_functions

# Vektör DB'ye bağlan
client = chromadb.PersistentClient(path="./veritabanim")
koleksiyon = client.create_collection(
    name="urun_aciklamalari",
    embedding_function=embedding_functions.DefaultEmbeddingFunction()
)

# Veri ekle
koleksiyon.add(
    documents=["Kırmızı deri cüzdan", "Mavi kot pantolon", "Beyaz gömlek"],
    metadatas=[{"kategori": "aksesuar"}, {"kategori": "giyim"}, {"kategori": "giyim"}],
    ids=["urun1", "urun2", "urun3"]
)

# Ara
sonuclar = koleksiyon.query(
    query_texts=["şık bir cüzdan arıyorum"],
    n_results=2
)
print(sonuclar['documents'])  # [['Kırmızı deri cüzdan', ...]]
```

**İlişkili:** Embedding Vector, ANN, HNSW, IVFFlat, Cosine Similarity

---

### Cosine Similarity (Kosinüs Benzerliği)
**Ne işe yarar:** İki vektör arasındaki yön (açı) benzerliğini ölçerek ne kadar "anlamca" yakın olduklarını hesaplama yöntemi.

**Detaylı:** Cosine similarity, adı üstünde, iki vektör arasındaki açının kosinüsünü alıyor. İki vektör aynı yönü gösteriyorsa kosinüs 1 (çok benzer), tam zıt yöndeyse -1 (çok farklı), dik ise 0 (alakasız) çıkıyor. Embedding'lerle çalışırken bu metriğin güzel tarafı, vektörlerin boyutundan (uzunluğundan) etkilenmemesi. Yani bir metin çok uzun, diğeri kısa olabilir — önemli olan anlamca hangi yöne baktıkları. Özellikle metin benzerliğinde en yaygın kullanılan metrik budur. Aynı vektör veritabanlarında da default similarity metrik olarak karşınıza çıkar. L2 (Euclidean) mesafesi bazen hata verebilir çünkü uzun metinlerin vektörlerinin boyu daha büyük olur, cosine similarity ise bu sorunu aşar.

**Örnek:** Python'da sıfırdan:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Üç kelimenin embedding'leri hayali olarak
mutlu = np.array([0.9, 0.2, 0.5])
neseli = np.array([0.85, 0.3, 0.4])
uzgun = np.array([-0.5, 0.1, 0.8])

print(f"Mutlu - Neşeli: {cosine_similarity(mutlu, neseli):.3f}")  # 0.97
print(f"Mutlu - Üzgün: {cosine_similarity(mutlu, uZgun):.3f}")   # 0.12
```

**İlişkili:** Embedding Vector, Dot Product, Vector Database

---

### Dot Product (Nokta Çarpımı)
**Ne işe yarar:** İki vektörün benzerliğini, yönlerinin yanı sıra büyüklüklerini de hesaba katarak hesaplama yöntemi.

**Detaylı:** Dot product, cosine similarity'e çok benzer ama bir fark var: vektörlerin boyunu (normunu) da hesaba katar. Yani aynı yöne bakan iki vektörden hangisi daha "büyük"se onun skoru daha yüksek olur. Bu bazen iyi bir şey değil — uzun bir metni, kısa ama tam isabetli bir metnin önüne koyabilir. Ama normalize edilmiş vektörlerle (hepsi aynı boyda) çalışıyorsanız dot product ile cosine similarity aynı sonucu verir. Gerçek hayatta birçok vektör veritabanı dot product'ı cosine similarity'den daha hızlı hesaplar, bu yüzden embedding'leri normalize edip dot product kullanmak yaygın bir optimizasyondur. Hesaplama açısından da daha basit: sadece çarp ve topla.

**Örnek:**

```python
import numpy as np

a = np.array([0.5, 0.8, 0.2])
b = np.array([0.4, 0.9, 0.1])

nokta_carpim = np.dot(a, b)
print(f"Dot product: {nokta_carpim:.3f}")

# Normalize edilmiş vektörlerde dot product = cosine similarity
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
print(f"Normalize dot: {np.dot(a_norm, b_norm):.3f}")  # cosine ile aynı
```

**İlişkili:** Cosine Similarity, Embedding Vector, ANN

---

### ANN (Approximate Nearest Neighbor — Yaklaşık En Yakın Komşu)
**Ne işe yarar:** Milyonlarca vektör arasında, bir sorgu vektörüne en yakın olanları mükemmel doğruluk yerine çok yüksek hızla bularak çalışan arama algoritmalarının genel adı.

**Detaylı:** Tam en yakın komşu (exact nearest neighbor) bulmak, her vektörü sorguyla tek tek karşılaştırmak demek. Bir milyon vektörünüz varsa, bu bir milyon karşılaştırma demek — yavaş. ANN ise biraz doğruluktan feragat edip hız kazanır. Mesela en yakın 10 vektörün 8-9'unu doğru bulur, 1-2'sini kaçırabilir ama işlemi saniyeler yerine milisaniyelerde bitirir. Çoğu uygulama için bu yeterlidir — bir veya iki alakasız belge gelmesi, saniyelerce beklemekten iyidir. ANN algoritmaları çeşitli stratejiler kullanır: ağaç yapıları, hash tabanlı yöntemler veya graflar. Popüler ANN algoritmaları arasında HNSW, IVF, PQ (Product Quantization), LSH (Locality-Sensitive Hashing) var. RAG sistemleri neredeyse her zaman ANN kullanır çünkü kullanıcıya 30 saniye bekletmektense %95 doğrulukla 100ms'de cevap vermek çok daha iyidir.

**Örnek:** FAISS ile ANN arama:

```python
import faiss
import numpy as np

# 100 bin tane 768 boyutlu rastgele vektör
veri = np.random.random((100000, 768)).astype('float32')

# IVF (Inverted File) indeksi oluştur
nlist = 100  # küme sayısı
quantizer = faiss.IndexFlatIP(768)  # iç çarpım metriği
index = faiss.IndexIVFFlat(quantizer, 768, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(veri)
index.add(veri)

# Sorgu
sorgu = np.random.random((1, 768)).astype('float32')
distances, indices = index.search(sorgu, k=5)

print(f"En yakın 5 vektörün indeksleri: {indices}")
```

**İlişkili:** HNSW, IVFFlat, Vector Database, RAG

---

### HNSW (Hierarchical Navigable Small World — Hiyerarşik Gezinilebilir Küçük Dünya)
**Ne işe yarar:** Vektörleri katmanlı bir graf yapısında organize ederek, milyarlarca vektör arasında bile çok hızlı ANN araması yapabilen bir algoritma.

**Detaylı:** HNSW'nin arkasındaki fikir aslında sezgisel: bir şehirde bir yer ararken önce haritaya tepeden bakarsınız (en üst katman), sonra sokak seviyesine inersiniz (alt katman). HNSW de aynen böyle çalışıyor. Birden çok katmanlı bir graf kuruyorsunuz. En üst katmanda sadece birkaç düğüm var — kabaca bir fikir edinmek için. Aşağı indikçe daha fazla düğüm ve daha hassas arama yapıyorsunuz. Arama yaparken en üst katmandan başlayıp en yakın düğümü buluyor, sonra bir alt katmana inip orada devam ediyorsunuz. HNSW şu an ANN algoritmaları arasında hız-doğruluk dengesi açısından en popüler olanı. Pinecone, Qdrant, Weaviate gibi birçok vektör veritabanı default olarak HNSW kullanıyor. Tek dezavantajı indeksi bellekte tutması ve ekleme yaparken biraz yavaş olması.

**Örnek:** FAISS ile HNSW:

```python
import faiss
import numpy as np

d = 768  # vektör boyutu
veri = np.random.random((50000, d)).astype('float32')

# HNSW indeksi
index = faiss.IndexHNSWFlat(d, 32)  # 32 = her düğümün bağlantı sayısı
index.hnsw.efConstruction = 200  # indeks kalitesi
index.add(veri)

# Arama
index.hnsw.efSearch = 64  # arama derinliği

sorgu = np.random.random((1, d)).astype('float32')
D, I = index.search(sorgu, k=10)
```

**İlişkili:** ANN, IVFFlat, Vector Database, Embedding Vector

---

### IVFFlat (Inverted File with Flat — Ters Dosya ile Düz İndeks)
**Ne işe yarar:** Vektörleri kümelere ayırarak arama yapılacak alanı daraltan, ANN algoritmalarının en temel ve anlaşılır örneklerinden biri.

**Detaylı:** IVFFlat'ın mantığı çok basit: milyonlarca vektörü birkaç yüz kümeye bölüyorsunuz (k-means ile). Sonra sorgu geldiğinde, önce sorgunun hangi kümeye yakın olduğunu buluyor, sadece o kümenin içindeki vektörlere bakıyorsunuz. "Flat" kısmı, küme içinde tam arama yapılması anlamına geliyor — yani küçük bir alanda kusursuz arama yapıp hız kazanıyorsunuz. Ayarı da kolay: küme sayısı (nlist) ve aramada kaç kümeye bakılacağı (nprobe). nprobe'u artırınca doğruluk artıyor ama hız düşüyor. IVFFlat, HNSW'ye göre daha az bellek kullanır ve indeks oluşturması daha hızlıdır. Ama çok yüksek boyutlu verilerde (768+) HNSW genelde daha iyi sonuç verir.

**Örnek:**

```python
import faiss
import numpy as np

d = 768
veri = np.random.random((100000, d)).astype('float32')

nlist = 100  # k-means ile 100 küme
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

index.train(veri)
index.add(veri)

# nprobe: kaç kümeye bakacağız
index.nprobe = 10

sorgu = np.random.random((1, d)).astype('float32')
D, I = index.search(sorgu, k=5)
```

**İlişkili:** ANN, HNSW, Vector Database, Cosine Similarity

---

### Retriever (Getirici / Alıcı)
**Ne işe yarar:** Bir RAG sisteminde, kullanıcının sorusuna en uygun belge parçalarını vektör veritabanından bulup getiren bileşen.

**Detaylı:** Retriever, RAG sisteminin "arama motoru" kısmıdır. Kullanıcı bir soru sorduğunda retriever devreye girer: soruyu embedding'e çevirir, vektör veritabanında en yakın belgeleri arar ve bulduklarını bir sonraki aşamaya (generation) iletir. Ne kadar iyi retriever'ınız varsa, RAG sisteminiz o kadar iyidir. Zayıf bir retriever ya alakasız belgeler getirir ya da doğru belgeleri kaçırır. Retriever'lar ikiye ayrılır: sparse (seyrek) ve dense (yoğun). Sparse retriever'lar (BM25 gibi) anahtar kelime eşleşmesine dayanır — "makine öğrenmesi" diye ararsanız içinde bu kelimeleri barındıran belgeleri getirir. Dense retriever'lar ise embedding tabanlıdır — anlamı yakalar. Modern RAG sistemleri ikisini birden kullanıp sonuçları birleştirir (hybrid search). Retriever'ın döndürdüğü sonuç sayısı da önemlidir. Çok az getirirseniz model yeterli bilgi alamaz, çok fazla getirirseniz token limitini aşar veya dikkatin dağılmasına yol açarsınız. Genelde 3-5 arası iyi bir sayıdır.

**Örnek:** LangChain'de farklı retriever türleri:

```python
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Dense retriever (embedding tabanlı)
vektor_db = FAISS.load_local("vektorler", embeddings)
dense_retriever = vektor_db.as_retriever(search_kwargs={"k": 3})

# Sparse retriever (anahtar kelime tabanlı)
bm25_retriever = BM25Retriever.from_documents(belgeler)
bm25_retriever.k = 3

# Hybrid: ikisini birden kullan
ensemble = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
sonuclar = ensemble.invoke("deri cüzdan fiyatları")
```

**İlişkili:** RAG, Reranker, BM25, Embedding Vector, Vector Database

---

### Reranker (Yeniden Sıralayıcı)
**Ne işe yarar:** Retriever'ın getirdiği aday belgeleri daha hassas bir modelle tekrar puanlayarak en alakalı olanları başa alan bileşen.

**Detaylı:** Retriever hızlıdır ama bazen tam isabetli değildir. Mesela embedding tabanlı bir retriever "elma" ile "armut"u anlamca yakın bulabilir çünkü ikisi de meyve. Ama siz aslında sadece elma ile ilgili belgeler istiyorsunuzdur. İşte reranker burada devreye giriyor. Retriever'ın getirdiği (diyelim 20) belgeyi alır, daha güçlü bir modelle — genelde cross-encoder denilen bir yapıyla — bir kere daha değerlendirir. Cross-encoder, sorgu ile belgeyi aynı anda görür ve aralarındaki ilişkiyi daha derinlemesine analiz eder. Retriever gibi embedding çıkarmak yerine direkt sorgu-belge çiftini işler, bu yüzden daha yavaştır ama çok daha doğrudur. Bu yüzden önce hızlı retriever ile adayları 20-50'ye indirir, sonra reranker ile en iyi 3-5'ini seçersiniz. Cross-encoder modelleri arasında Cohere Rerank, BGE Reranker ve MonoBERT gibi seçenekler var. Reranker eklemek RAG sisteminizin cevap kalitesini gözle görülür şekilde artırır.

**Örnek:** Cohere'ın reranker'ı ile:

```python
import cohere

co = cohere.Client("API_ANAHTARI")

sorgu = "Kırmızı iPhone 15 fiyatı ne kadar?"
belgeler = [
    "iPhone 15 Pro Max 256GB Siyah 60.000 TL",
    "Kırmızı kılıf iPhone 15 uyumlu 500 TL",
    "iPhone 15 kırmızı renk seçeneği Türkiye'ye geliyor",
    "Samsung Galaxy S24 kırmızı 40.000 TL"
]

sonuc = co.rerank(
    query=sorgu,
    documents=belgeler,
    model="rerank-english-v3.0",
    top_n=2
)

for r in sonuc.results:
    print(f"Skor: {r.relevance_score:.3f} - {belgeler[r.index]}")
# En üstte ilgili belgeler, altta Samsung gibi alakasızlar
```

**İlişkili:** Retriever, RAG, Embedding Vector, Cosine Similarity

---

### BM25 (Best Matching 25)
**Ne işe yarar:** Anahtar kelime eşleşmesine dayanan, eski ama hala çok etkili bir metin arama (retrieval) algoritması.

**Detaylı:** BM25 aslında 1970'lerden beri var olan TF-IDF'nin gelişmiş bir versiyonu diyebilirim. Şöyle çalışıyor: bir belgede aradığınız kelime ne kadar sık geçiyorsa o belge o kadar relevant (TF kısmı). Ama o kelime çok yaygınsa (mesela "ve", "bir" gibi) onun ağırlığı düşüyor (IDF kısmı). BM25'in TF-IDF'den farkı, kelimenin belgede aşırı sık geçmesinin getirisini sınırlaması — bir kelime 100 kere geçiyorsa 50 kere geçmesinden çok daha iyi değildir, BM25 bunu dengeliyor. Modern RAG sistemlerinde embedding tabanlı arama genelde daha iyi sonuç verse de BM25 hala çok değerli, özellikle teknik dokümanlarda tam eşleşme gerektiğinde. Mesela "Python 3.12 async/await syntax" diye arıyorsanız, BM25 bu spesifik terimlerin geçtiği belgeleri embedding'den daha iyi bulur. Bu yüzden hybrid search'te BM25 + embedding ikilisi sıkça kullanılır.

**Örnek:** rank_bm25 kütüphanesi ile:

```python
from rank_bm25 import BM25Okapi

belgeler = [
    "Kırmızı elma çok lezzetli",
    "Yeşil elma daha ekşi olur",
    "Muz potasyum deposudur"
]

tokenize_belgeler = [doc.split() for doc in belgeler]
bm25 = BM25Okapi(tokenize_belgeler)

sorgu = "kırmızı elma".split()
skorlar = bm25.get_scores(sorgu)

for i, skor in enumerate(skorlar):
    print(f"{belgeler[i]}: {skor:.2f}")
# Çıktı:
# Kırmızı elma çok lezzetli: 1.86
# Yeşil elma daha ekşi olur: 0.54
# Muz potasyum deposudur: 0.00
```

**İlişkili:** Retriever, RAG, Cosine Similarity, Embedding Vector
