# Gün 9: Donanım & Altyapı

LLM eğitmek için grafik kartı lazım — o kadar. Bir modeli eğitmek artık sadece algoritma bilmek değil, GPU belleğinden termal throttle'a, interconnect hızından power budget'a kadar donanım mühendisliği bilmeyi de gerektiriyor. Bu bölümde AI çalışmalarının bel kemiği olan donanım terimlerini ve altyapı kavramlarını anlatıyorum.

---

### VRAM (Video Random Access Memory)
**Ne işe yarar:** GPU üzerinde modellerin ağırlıklarını, aktivasyon tensörlerini ve ara hesaplamaları geçici olarak saklayan bellektir.

**Detaylı:** VRAM, modelinizi eğitirken veya inference yaparken kullanacağınız en kritik kaynaklardan biridir. Model ağırlıkları, optimizer state'leri, gradient'ler ve forward pass sırasında hesaplanan aktivasyon tensörlerinin tamamı VRAM'de tutulur. 7B parametreli bir modeli FP16 hassasiyette yüklemek yaklaşık ~14 GB VRAM isterken, aynı modeli BF16'da eğitmek aktivasyon ve gradient'lerle birlikte ~50-60 GB'ı bulabilir. RTX 4090'da 24 GB, A100'de 40/80 GB, H100'de 80 GB VRAM bulunur. Modelinizi tek GPU'ya sığdıramıyorsanız model parallelism, pipeline parallelism veya CPU offloading gibi tekniklere başvurmanız gerekir — ki bu da beraberinde iletişim gecikmesi getirir. Kısacası VRAM ne kadar büyükse, o kadar büyük model tek kartta çalıştırılabilir.

**Örnek:** Model boyutuna göre VRAM ihtiyacı tahmini:

```python
def vram_hesapla(parametre_sayisi: int, precision_bytes: int = 2):
    """Basit VRAM tahmin hesaplayıcı (sadece ağırlıklar)"""
    weight_size = parametre_sayisi * precision_bytes / (1024**3)
    aktivasyon_basina = weight_size * 0.5  # kaba tahmin
    
    return {
        "parametre_sayisi": f"{parametre_sayisi/1e9:.1f}B",
        "precision": f"{precision_bytes * 8}-bit",
        "weight_size_gb": round(weight_size, 1),
        "tahmini_egitim_vram_gb": round(weight_size * 3 + aktivasyon_basina, 1),
    }

modeller = [
    (7e9, 2),     # 7B FP16
    (13e9, 2),    # 13B FP16
    (70e9, 2),    # 70B FP16
    (7e9, 4),     # 7B FP32
    (70e9, 4),    # 70B FP32 (imkansıza yakın)
]

for params, prec in modeller:
    sonuc = vram_hesapla(params, prec)
    print(f"{sonuc['parametre_sayisi']} @ {sonuc['precision']}: "
          f"→ Ağırlık: {sonuc['weight_size_gb']} GB, "
          f"Tahmini Eğitim: {sonuc['tahmini_egitim_vram_gb']} GB")
"""
7.0B @ 16-bit → Ağırlık: 13.0 GB, Tahmini Eğitim: 45.6 GB
13.0B @ 16-bit → Ağırlık: 24.2 GB, Tahmini Eğitim: 84.7 GB
70.0B @ 16-bit → Ağırlık: 130.3 GB, Tahmini Eğitim: 456.0 GB
7.0B @ 32-bit → Ağırlık: 26.1 GB, Tahmini Eğitim: 91.2 GB
70.0B @ 32-bit → Ağırlık: 260.7 GB, Tahmini Eğitim: 911.9 GB
"""
```

**İlişkili:** GPU, Memory Bandwidth, Model Parallelism, Quantization, Batch Size

---

### GPU (CUDA Cores / Tensor Cores / RT Cores)
**Ne işe yarar:** AI iş yüklerini hızlandırmak için özel olarak tasarlanmış grafik işlemciler ve içlerindeki farklı çekirdek türleri.

**Detaylı:** NVIDIA GPU'lar üç tip çekirdekle gelir: CUDA Cores, Tensor Cores ve RT Cores. CUDA Cores temel paralel işlem birimleridir — matris çarpımından veri kopyalamaya kadar her şeyi yapabilirler. Ama AI için asıl oyuncu Tensor Cores. Tensor Cores, özellikle matris çarpımı (GEMM) işlemleri için optimize edilmiş özel çekirdeklerdir. 16-bit, 8-bit ve hatta 4-bit precision'da çalışarak CUDA Cores'a göre 4x-8x daha hızlı matris işlemi yapabilirler. RT Cores ise ray tracing için var — AI iş yüklerinde pek kullanılmaz. H100'de 18432 CUDA Core ve 576 Tensor Core varken, A100'de 6912 CUDA Core ve 432 Tensor Core bulunur. Eğitim sırasında Tensor Cores aktif değilse, H100 bile masaüstü GPU'sundan yavaş kalabilir.

**Örnek:** Tensor Cores kullanarak PyTorch'da matris çarpımı:

```python
import torch
import time

n = 4096
A = torch.randn(n, n, device="cuda")
B = torch.randn(n, n, device="cuda")

# Tensor Cores aktif değil (FP32)
start = time.time()
C_fp32 = A @ B
torch.cuda.synchronize()
print(f"FP32 (CUDA Cores): {time.time() - start:.3f}s")

# Tensor Cores aktif (TF32 — Ampere+ GPU'lar)
torch.backends.cuda.matmul.allow_tf32 = True
start = time.time()
C_tf32 = A @ B
torch.cuda.synchronize()
print(f"TF32 (Tensor Cores): {time.time() - start:.3f}s")

# BF16 ile Tensor Cores
A_bf16 = A.bfloat16()
B_bf16 = B.bfloat16()
start = time.time()
C_bf16 = A_bf16 @ B_bf16
torch.cuda.synchronize()
print(f"BF16 (Tensor Cores): {time.time() - start:.3f}s")

# Örnek çıktı (RTX 4090):
# FP32 (CUDA Cores): 0.042s
# TF32 (Tensor Cores): 0.018s
# BF16 (Tensor Cores): 0.012s
```

**İlişkili:** VRAM, CUDA, Tensor Cores, Flash Attention, Mixed Precision Training

---

### TPU (Tensor Processing Unit)
**Ne işe yarar:** Google'ın sadece tensor işlemleri için geliştirdiği, özellikle büyük dil modellerini eğitmek için optimize edilmiş özel donanım.

**Detaylı:** TPU, Google'ın 2016'da duyurduğu, tamamen matris işlemleri (matrix multiply, convolution) için sıfırdan tasarlanmış bir ASIC (Application-Specific Integrated Circuit). GPU'nun aksine, TPU'lar görüntü işleme veya grafik render için değil, sadece tensor matematik için üretilir. TPU v4 çipleri 275 TFLOPS BF16 işlem gücüne sahip ve 4096 TPU'luk bir pod'da 1.1 exaFLOPS'a ulaşabilir. TPU'ların en büyük avantajı, inter-chip bağlantılarının (ICI — Inter-Chip Interconnect) çok hızlı olması sayesinde binlerce TPU'yu neredeyse tek bir cihaz gibi kullanabilmenizdir. Dezavantajı ise esneklik: sadece TensorFlow/JAX ile çalışır, CUDA yoktur, PyTorch desteği Pallas kernel'leri ile sınırlıdır. Google, Gemini'yi kendi TPU v5p ve v5e pod'larında eğitmiştir.

**Örnek:** TPU'da JAX ile model eğitimi (genel yapı):

```python
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import optax

# TPU kullanılabilir mi kontrolü
print(f"TPU cihaz sayısı: {jax.device_count()}")
print(f"TPU türü: {jax.devices()[0].device_kind}")

# Basit bir lineer model (TPU'da çalışır)
def model(params, x):
    w, b = params
    return jnp.dot(x, w) + b

def loss_fn(params, x, y):
    pred = model(params, x)
    return jnp.mean((pred - y) ** 2)

@jit
def train_step(params, x, y, optimizer_state):
    grads = grad(loss_fn)(params, x, y)
    updates, optimizer_state = optimizer.update(grads, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state

# Parametreler
rng = jax.random.PRNGKey(0)
w = jax.random.normal(rng, (768, 10))
b = jnp.zeros(10)
params = (w, b)

optimizer = optax.adam(1e-3)
optimizer_state = optimizer.init(params)
```

**İlişkili:** GPU, ASIC, JAX, TensorFlow, Google Cloud TPU, MXU (Matrix Multiply Unit)

---

### HBM (HBM2e / HBM3 / HBM4) — High Bandwidth Memory
**Ne işe yarar:** GPU ve TPU gibi AI hızlandırıcılarında kullanılan, çok yüksek bant genişliğine sahip bir bellek teknolojisidir.

**Detaylı:** HBM, GDDR belleklerin aksine yığılmış (stacked) bir yapıda tasarlanır — bellek çipleri birbirinin üzerine dizilir ve aradaki bağlantılar silicon interposer üzerinden yapılır. Bu sayede GDDR'ye göre çok daha geniş bir veri yolu (1024-bit vs 384-bit) ve dolayısıyla çok daha yüksek bant genişliği elde edilir. A100'de HBM2e (2.0 TB/s), H100'de HBM3 (3.35 TB/s) kullanılır. HBM4 ise 2026'da piyasada olması beklenen ve ~6.4 TB/s bant genişliği vaat eden yeni nesil. Yüksek bant genişliği, büyük modellerde bellek erişimlerinin GPU hesaplamasını bekletmemesi (bottleneck olmaması) için kritiktir. Ama HBM'nin üretim maliyeti GDDR'ye göre çok daha yüksektir — bu yüzden sadece sunucu sınıfı GPU'larda (A100, H100, B200) kullanılır.

**Örnek:** Bellek bant genişliklerini karşılaştırma:

```python
# Farklı GPU'ların HBM bant genişlikleri
hbms = [
    {"ad": "A100 (HBM2e)",  "bant_genisligi_gbps": 2039, "kapasite_gb": 80, "uretim": 2020},
    {"ad": "H100 (HBM3)",   "bant_genisligi_gbps": 3350, "kapasite_gb": 80, "uretim": 2022},
    {"ad": "B200 (HBM3e)",  "bant_genisligi_gbps": 8000, "kapasite_gb": 192, "uretim": 2024},
    {"ad": "H200 (HBM3e)",  "bant_genisligi_gbps": 4800, "kapasite_gb": 141, "uretim": 2024},
]

# 70B modeli bir token üretmek için gereken bellek okuma süresi
# Model ağırlıkları: 70B * 2 bayt (BF16) = 140 GB
# Optimist: tüm ağırlık tek seferde okunur
model_size_gb = 140  # 70B BF16

print(f"{'GPU':20s} {'Bant Genişliği':15s} {'7B Inference':15s} {'70B Inference':15s}")
print("-" * 65)
for hbm in hbms:
    time_7b = (14 * 1000) / hbm["bant_genisligi_gbps"]  # ms
    time_70b = (model_size_gb * 1000) / hbm["bant_genisligi_gbps"]  # ms
    print(f"{hbm['ad']:20s} {hbm['bant_genisligi_gbps']:>5} GB/s    "
          f"{time_7b:>5.1f} ms        {time_70b:>5.1f} ms")
"""
GPU                  Bant Genişliği  7B Inference    70B Inference
-----------------------------------------------------------------
A100 (HBM2e)              2039 GB/s    6.9 ms         68.7 ms
H100 (HBM3)               3350 GB/s    4.2 ms         41.8 ms
B200 (HBM3e)              8000 GB/s    1.7 ms         17.5 ms
H200 (HBM3e)              4800 GB/s    2.9 ms         29.2 ms
"""
```

**İlişkili:** Memory Bandwidth, GDDR GPU Memory, VRAM, Interposer, Bandwidth Bottleneck

---

### GDDR (GDDR6 / GDDR6X / GDDR7) — Graphics Double Data Rate
**Ne işe yarar:** Tüketici sınıfı GPU'larda (RTX, Radeon) kullanılan, HBM'ye göre daha ucuz ama daha düşük bant genişliğine sahip bellek teknolojisi.

**Detaylı:** GDDR, adından da anlaşılacağı gibi grafik kartları için standart bellek türüdür. GDDR6 (RTX 3080, RTX 4070) ~16 Gbps hızında çalışırken, GDDR6X (RTX 3090, RTX 4090) PAM4 sinyalleme sayesinde ~21-23 Gbps'ye çıkar. GDDR7 ise 2024'te duyuruldu ve 32-37 Gbps hızlarına ulaşarak HBM3'ün tüketici versiyonu olmaya aday. GDDR'nin avantajı fiyat/performans oranı — HBM'ye göre çok daha ucuz ve seri üretime uygun. Dezavantajı ise daha yüksek güç tüketimi ve daha dar veri yolu (384-bit max). Eğer 24 GB VRAM'li bir RTX 4090 yeterli geliyorsa, bütçe dostu bir çözümdür. 70B+ gibi büyük modeller için ise HBM şart oluyor.

**Örnek:**
```python
# GDDR nesillerini karşılaştırma
gddr_nesiller = [
    {"ad": "GDDR6", "hiz_gbps": 16, "veri_yolu_bit": 384, "toplam_gbps": 16 * 384 / 8, "guc_w": 2.5},
    {"ad": "GDDR6X", "hiz_gbps": 21, "veri_yolu_bit": 384, "toplam_gbps": 21 * 384 / 8, "guc_w": 3.0},
    {"id": "GDDR7", "hiz_gbps": 32, "veri_yolu_bit": 384, "toplam_gbps": 32 * 384 / 8, "guc_w": 2.0},
]

print(f"{'Nesil':10s} {'Hız':8s} {'Veri Yolu':10s} {'Bant Genişliği':15s} {'Güç/Çip':10s}")
print("-" * 55)
for g in gddr_nesiller:
    print(f"{g['ad']:10s} {g['hiz_gbps']:>2} Gbps   {g['veri_yolu_bit']:>3}-bit  "
          f"{g['toplam_gbps']:>3.0f} GB/s      {g['guc_w']} W")
    
# GDDR6X (RTX 4090) ile HBM3 (H100) karşılaştırması
print()
print("Karşılaştırma:")
print(f"RTX 4090 (GDDR6X): 24 GB @ 1008 GB/s")
print(f"H100 (HBM3):      80 GB @ 3350 GB/s")
print(f"H100, RTX 4090'dan ~3.3x daha fazla bant genişliği sunar.")
```

**İlişkili:** HBM, VRAM, GPU Memory, Memory Bandwidth, PAM4 Signaling

---

### NPU (Neural Processing Unit)
**Ne işe yarar:** Akıllı telefonlardan laptoplara kadar her cihazda, AI iş yüklerini verimli bir şekilde çalıştırmak için özel olarak tasarlanmış işlemci birimi.

**Detaylı:** NPU, CPU ve GPU'dan bağımsız, sadece sinir ağı hesaplamaları (özellikle inference) için optimize edilmiş bir çiptir. Apple'ın Neural Engine (16-core, 35 TOPS), Qualcomm'un Hexagon NPU (45 TOPS), Intel'in NPU (11 TOPS) ve AMD'nin XDNA NPU'su (50 TOPS) en yaygın örnekler. NPU'ların en büyük avantajı çok düşük güç tüketimi — aynı işi, CPU'ya göre 10-20x daha az enerjiyle yapabilirler. Bu sayede fotoğraf düzenleme, ses tanıma, real-time filtreleme gibi işlemler bataryayı tüketmeden cihaz üzerinde yapılabilir. Windows Copilot+ PC'lerde NPU zorunlu hale geldi, çünkü Recall, Windows Studio Effects gibi özellikler NPU olmadan verimli çalışamaz. Apple Silicon'daki Neural Engine ise Face ID, Live Text, Siri ve fotoğraf işleme gibi her gün kullandığınız özellikleri sessizce çalıştırır.

**Örnek:** Apple Neural Engine kullanımı (CoreML):

```python
# Apple Neural Engine (ANE) — CoreML Inference
import coremltools as ct
import numpy as np

# ONNX modelini CoreML'e çevir (ANE uyumlu)
def modeli_ane_aktar():
    spec = ct.converters.onnx.convert(
        model="model.onnx",
        minimum_deployment_target=ct.target.iOS17,
        compute_units=ct.ComputeUnit.ALL  # CPU+GPU+ANE
    )
    spec.save("Model.mlpackage")

# Hangi birimin kullanıldığını kontrol et
# Eğer ANE kullanılıyorsa, güç tüketimi çok düşüktür
# Apple Silicon Mac'lerde: ANE ~15W, GPU ~40W, CPU ~30W
# Aynı işi ANE'de yapmak, GPU'ya göre 3x daha az enerji harcar

# Qualcomm Hexagon NPU kullanımı (Android)
# Qualcomm Snapdragon cihazlarda:
# Neural Networks API (NNAPI) ile NPU otomatik kullanılır
# Adreno GPU ile Hexagon NPU arasında otomatik geçiş yapılır
```

**İlişkili:** Edge AI, On-Device Inference, Apple Neural Engine, Qualcomm Hexagon, Intel NPU, Snapdragon

---

### FLOPS (TFLOPS / PFLOPS) — Floating Point Operations Per Second
**Ne işe yarar:** Bir işlemcinin saniyede kaç kayan nokta işlemi yapabildiğini ölçen performans birimidir.

**Detaylı:** FLOPS, AI donanımlarının ham hesaplama gücünü karşılaştırmak için kullanılan standart metriktir. 1 TFLOPS = 1 trilyon, 1 PFLOPS = 1 katrilyon işlem/saniye. Ama burada bir incelik var: precision'a göre FLOPS değişir. Bir GPU, FP32'de 20 TFLOPS yaparken, FP16'da 160 TFLOPS, INT8'de 320 TFLOPS yapabilir. NVIDIA genelde "sparse FP16 TFLOPS" değerini verir ki bu gerçekçi kullanımdan daha yüksektir. Training FLOPs ise full precision'da (genelde BF16) ölçülmelidir. GPT-3 (175B) ~3.14E23 FLOPs ile eğitilmişti — bu da 1000 A100 ile ~34 gün eder. TPU v4 pod'u ise ~1.1 exaFLOPS'a ulaşarak bu süreyi dramatik şekilde kısaltabilir.

**Örnek:**
```python
# GPU FLOPS karşılaştırması (BF16/TF32)
gpular = [
    {"ad": "RTX 4090",   "fp16_tflops": 82.6, "vram_gb": 24, "tdp_w": 450},
    {"ad": "A100 80GB",  "fp16_tflops": 312,  "vram_gb": 80, "tdp_w": 400},
    {"ad": "H100 SXM",   "fp16_tflops": 989,  "vram_gb": 80, "tdp_w": 700},
    {"ad": "B200",       "fp16_tflops": 2250, "vram_gb": 192, "tdp_w": 1000},
    {"ad": "TPU v5p",    "fp16_tflops": 459,  "vram_gb": 95,  "tdp_w": "~700"},
]

# GPT-3 (175B) eğitim maliyeti hesaplama
gpt3_flops = 3.14e23  # Total FLOPs for GPT-3 training

print(f"{'GPU':15s} {'TFLOPS':10s} {'Eğitim Süresi':18s}")
print("-" * 45)
for gpu in gpular:
    if isinstance(gpu["tdp_w"], int):
        # 1000 GPU ile eğitim süresi
        total_tflops = gpu["fp16_tflops"] * 1000
        sn = gpt3_flops / (total_tflops * 1e12)
        gun = sn / (3600 * 24)
        print(f"{gpu['ad']:15s} {gpu['fp16_tflops']:>5}     ~{gun:.0f} gün (1000 GPU)")
```

**İlişkili:** GPU Performance, Precision (FP16/FP32), Training Time, Throughput, MAC (Multiply-Accumulate)

---

### Memory Bandwidth (Bellek Bant Genişliği)
**Ne işe yarar:** GPU'nun VRAM'inden işlem birimlerine saniyede ne kadar veri aktarabildiğini ölçer — inference hızını doğrudan etkiler.

**Detaylı:** Memory bandwidth, AI performansında FLOPS'tan bile kritik olabilir. Çünkü modern GPU'lar o kadar hızlı hesaplama yapabiliyor ki, çoğu zaman asıl darboğaz veriye erişim hızı oluyor. Bir modelin inference'ı çoğunlukla memory-bound'dur, yani GPU sürekli ağırlıkları bellekten okuyup işlemek zorundadır. Bandwidth = (Veri Yolu Genişliği × Çalışma Frekansı × Transfer Sayısı) / 8 formülüyle hesaplanır. RTX 4090'da 1008 GB/s, H100'de 3350 GB/s, B200'de 8000 GB/s bant genişliği bulunur. Daha yüksek bant genişliği demek, model ağırlıklarının GPU'ya daha hızlı beslenmesi demektir — yani aynı model daha hızlı cevap verir.

**Örnek:** Bellek bant genişliğinin inference üzerindeki etkisi:

```python
import torch
import time

def bandwidth_test(device="cuda:0", size_mb=1024):
    """GPU'nun efektif bellek bant genişliğini ölç"""
    n = size_mb * 256 * 1024  # float32 sayısı
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    
    # Isınma
    z = x + y
    torch.cuda.synchronize()
    
    # Ölçüm (memory-bound operasyon)
    start = time.time()
    for _ in range(100):
        z = x + y
    torch.cuda.synchronize()
    
    sure = (time.time() - start) / 100
    # Her operasyonda okunan + yazılan byte: x(4*n) + y(4*n) + z(4*n) = 12*n
    bytes_transferred = 12 * n * 4  # float32
    bandwidth = (bytes_transferred / sure) / (1024**3)  # GB/s
    
    return {
        "operasyon_boyutu_mb": size_mb,
        "sure_ms": round(sure * 1000, 2),
        "efektif_bandwidth_gbps": round(bandwidth, 1),
    }

# Sonuç örneği (RTX 4090):
# {'operasyon_boyutu_mb': 1024, 'sure_ms': 1.02, 'efektif_bandwidth_gbps': 1008.0}
```

**İlişkili:** HBM, GDDR, VRAM, Throughput, Memory-Bound vs Compute-Bound, Roofline Model

---

### Interconnect (NVLink / PCIe / Infinity Fabric)
**Ne işe yarar:** Birden fazla GPU veya işlemcinin birbiriyle yüksek hızda veri paylaşmasını sağlayan bağlantı teknolojileridir.

**Detaylı:** Tek bir GPU'ya sığmayan bir model eğitiyorsanız, GPU'ların birbirleriyle konuşması gerekir. İşte bu noktada interconnect devreye girer. PCIe 4.0/5.0, GPU'ları anakarta bağlayan standart arayüzdür ama hızı sınırlıdır (PCIe 5.0 x16: ~63 GB/s). NVIDIA'nın NVLink'i ise GPU'lar arası özel bir bağlantıdır — H100'de 18 NVLink bağlantısıyla toplam 900 GB/s GPU-to-GPU bant genişliği sağlar. AMD'nin Infinity Fabric'i ise CPU-GPU ve CPU-CPU arasında benzer bir rol oynar. Büyük modellerde (70B+), gradient'lerin GPU'lar arasında paylaşılması için yüksek hızlı interconnect şarttır. NVLink olmadan PCIe üzerinden eğitim yapmak, iletişim yükü nedeniyle 2-3x yavaşlayabilir. Bu yüzden DGX gibi AI sunucuları tüm GPU'ları full NVLink mesh ile bağlar.

**Örnek:** NVLink vs PCIe performans testi (teorik):

```python
# İki GPU arasında veri transfer hızları
import torch
# Not: Bu kod 2 GPU'lu bir sistemde çalışır

def interconnect_test():
    n = 1024 * 1024 * 256  # 1 GB float32
    data = torch.randn(n, device="cuda:0")
    
    # PCIe üzerinden transfer
    start = time.time()
    data_pcie = data.to("cuda:1", non_blocking=True)
    torch.cuda.synchronize()
    pcie_time = time.time() - start
    
    # NVLink üzerinden transfer (eğer varsa)
    # Aynı işlem ama NVLink otomatik kullanılır
    print(f"PCIe transfer (1 GB): {pcie_time:.3f}s → {1/pcie_time:.0f} GB/s")
    
# NVLink vs PCIe bant genişlikleri
interconnectlar = [
    {"ad": "PCIe 4.0 x16",  "bant_genisligi_gbps": 31.5, "gecikme_us": 0.5},
    {"ad": "PCIe 5.0 x16",  "bant_genisligi_gbps": 63.0, "gecikme_us": 0.4},
    {"ad": "NVLink (A100)", "bant_genisligi_gbps": 600,  "gecikme_us": 0.1},
    {"ad": "NVLink (H100)", "bant_genisligi_gbps": 900,  "gecikme_us": 0.08},
    {"ad": "Infinity Fabric (AMD)", "bant_genisligi_gbps": 200, "gecikme_us": 0.2},
]

print(f"\n{'Bağlantı':25s} {'Bant Genişliği':18s} {'Gecikme':10s}")
print("-" * 55)
for ic in interconnectlar:
    print(f"{ic['ad']:25s} {ic['bant_genisligi_gbps']:>3} GB/s        {ic['gecikme_us']} μs")
```

**İlişkili:** Multi-GPU Training, Model Parallelism, Data Parallelism, NCCL, Distributed Training, DGX

---

### TDP / Power Consumption (Termal Tasarım Gücü / Güç Tüketimi)
**Ne işe yarar:** Bir GPU veya işlemcinin maksimum yük altında ne kadar ısı yaydığını (ve dolayısıyla ne kadar güç çektiğini) belirten değerdir.

**Detaylı:** TDP (Thermal Design Power), soğutma sisteminizin kaldırması gereken maksimum ısı yükünü watt cinsinden belirtir. AI iş yükleri GPU'yu tam yükte çalıştırdığı için TDP çok kritiktir. RTX 4090'ın TDP'si 450W, H100'ün 700W, B200'ün 1000W civarındadır. 8x H100'den oluşan bir DGX sisteminin güç tüketimi 700W × 8 = 5600W + CPU/RAM/soğutma ile ~7000W'ı bulur. Bu, ev tipi bir klima gibi düşünün — 7 kW sürekli çalışan bir cihaz. Bu yüzden büyük AI kümeleri soğutma altyapısına neredeyse GPU'lara harcadıkları kadar para harcarlar. Sıvı soğutma, H100 gibi yüksek TDP'li kartlar için standart hale gelmiştir. Ayrıca, güç kısıtlı ortamlarda (örneğin bir ofiste 3-4 GPU çalıştırmak) TDP'yi düşürmek için undervolt veya power limit ayarları yapılabilir — performansın %5-10'unu kaybedip güç tüketimini %30-40 azaltmak mümkündür.

**Örnek:** Bir AI sunucusunun güç tüketimi hesaplama:

```python
def sunucu_guc_hesapla(adet_gpu, gpu_tdp_w, diger_w=500, psu_verim=0.92):
    """Bir AI sunucusunun toplam güç tüketimini hesapla"""
    gpu_toplam = adet_gpu * gpu_tdp_w
    toplam_ac_dc = (gpu_toplam + diger_w) / psu_verim
    toplam_w = int(toplam_ac_dc)
    aylik_kwh = (toplam_w * 24 * 30) / 1000
    aylik_maliyet_tl = aylik_kwh * 2.5  # ~2.5 TL/kWh (Türkiye)
    
    return {
        "adet_gpu": adet_gpu,
        "gpu_basina_w": gpu_tdp_w,
        "diger_bilesenler_w": diger_w,
        "psu_kayip_yuzde": round((1 - psu_verim) * 100, 1),
        "toplam_duvar_tuketimi_w": toplam_w,
        "aylik_tahmini_kwh": round(aylik_kwh, 0),
        "aylik_tahmini_elektrik_tl": round(aylik_maliyet_tl, 0),
    }

# H100 kümesi (8 GPU)
h100_kume = sunucu_guc_hesapla(8, 700, diger_w=800, psu_verim=0.94)
print("H100 x8 DGX Sunucu Güç Hesabı:")
for k, v in h100_kume.items():
    print(f"  {k}: {v}")
"""
H100 x8 DGX Sunucu Güç Hesabı:
  adet_gpu: 8
  gpu_basina_w: 700
  diger_bilesenler_w: 800
  psu_kayip_yuzde: 6.4
  toplam_duvar_tuketimi_w: 6382
  aylik_tahmini_kwh: 4595.0
  aylik_tahmini_elektrik_tl: 11488.0
"""

# RTX 4090 ev sistemi (2 GPU)
rtx4090_ev = sunucu_guc_hesapla(2, 450, diger_w=400, psu_verim=0.90)
print(f"\nRTX 4090 x2 Ev Sistemi Güç Hesabı:")
for k, v in rtx4090_ev.items():
    print(f"  {k}: {v}")
"""
RTX 4090 x2 Ev Sistemi Güç Hesabı:
  adet_gpu: 2
  gpu_basina_w: 450
  diger_bilesenler_w: 400
  psu_kayip_yuzde: 11.1
  toplam_duvar_tuketimi_w: 1444
  aylik_tahmini_kwh: 1040.0
  aylik_tahmini_elektrik_tl: 2600.0
"""

# Performance-per-watt karşılaştırması
gpular = [
    {"ad": "RTX 4090", "tflops": 82.6, "tdp": 450},
    {"ad": "RTX 6000 Ada", "tflops": 91, "tdp": 300},
    {"ad": "A100 80GB", "tflops": 312, "tdp": 400},
    {"ad": "H100 SXM", "tflops": 989, "tdp": 700},
    {"ad": "B200", "tflops": 2250, "tdp": 1000},
]

print(f"\n{'GPU':20s} {'TFLOPS':10s} {'TDP(W)':10s} {'TFLOPS/W':10s}")
print("-" * 50)
for g in gpular:
    ppw = g["tflops"] / g["tdp"]
    print(f"{g['ad']:20s} {g['tflops']:>5}     {g['tdp']:>3}      {ppw:.2f}")
"""
GPU                   TFLOPS     TDP(W)     TFLOPS/W
--------------------------------------------------
RTX 4090               82.6     450       0.18
RTX 6000 Ada           91.0     300       0.30
A100 80GB             312.0     400       0.78
H100 SXM              989.0     700       1.41
B200                 2250.0    1000       2.25
"""
```

**İlişkili:** GPU Power Limit, Undervolt, Thermal Throttle, Data Center Cooling, PUE, Performance-per-Watt