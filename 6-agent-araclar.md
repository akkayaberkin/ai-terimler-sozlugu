# Gün 6: Agent & Araçlar

Bir modelin tek başına bazen yetmediği durumlar vardır — dış dünyayla etkileşime geçmesi, bir plan yapması, birden çok adım düşünmesi gerekir. İşte bu bölümde modellerin pasif bir bilgi kaynağı olmaktan çıkıp aktif birer "ajan"a dönüşmesini sağlayan kavramları anlatıyorum.

---

### AI Agent (Yapay Zeka Ajanı)
**Ne işe yarar:** Bir dil modelinin, çevresini algılayıp kararlar alarak birden çok adımda hedefe ulaşmasını sağlayan sistem.

**Detaylı:** Normalde bir dil modeline soru sorarsınız, cevap verir ve konu kapanır. Agent modeli ise bir hedef verdiğinizde o hedefe ulaşana kadar döngü halinde çalışır: düşünür, bir araç çağırır, sonucu değerlendirir, gerekiyorsa tekrar dener. Örneğin "İstanbul-Ankara arası en ucuz uçak biletini bul ve takvimime ekle" dediğinizde agent önce bir uçuş API'sine bakar, sonuçları inceler, en ucuzunu seçer ve takvim API'sine yazar — tüm bunları kendi kendine yapar. Agent'ları sıradan bir chatbot'tan ayıran şey: otonomi (kullanıcıya her adımda sormaz), araç kullanma yeteneği (API, dosya, veritabanı), ve bellek (önceki adımları unutmaz). Ne kadar karmaşık bir görevse, o kadar iyi tasarlanmış bir agent yapısına ihtiyacınız olur. Basit bir "hava durumunu sor-cevapla" işi için agente gerek yok, direkt API çağrısı yeter.

**Örnek:** LangChain ile basit bir agent:

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI

@tool
def hava_durumu(sehir: str) -> str:
    """Belirtilen şehir için hava durumunu döndürür."""
    # Gerçekte bir API çağrısı olurdu
    return f"{sehir}'de hava 22°C, parçalı bulutlu"

tools = [hava_durumu]
model = ChatOpenAI(model="gpt-4o")

agent = create_tool_calling_agent(model, tools)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

sonuc = executor.invoke({"input": "Ankara'da hava nasıl?"})
print(sonuc["output"])
```

**İlişkili:** Tool Calling, ReAct Loop, Multi-Agent System, Orchestrator Agent

---

### Tool Calling / Function Calling (Araç Çağırma / Fonksiyon Çağırma)
**Ne işe yarar:** Dil modelinin, konuşma akışı içinde harici bir API'yi veya fonksiyonu çağırmasına olanak tanıyan mekanizma.

**Detaylı:** Tool calling, modelin artık sadece metin üretmekle kalmayıp dış dünyayla etkileşime geçebilmesini sağlayan en temel yapı taşıdır. Şöyle işler: modele kullanabileceği araçların (fonksiyonların) şemalarını verirsiniz — "şu parametreleri alan şu fonksiyon var" diye. Model, kullanıcının isteğine göre bu fonksiyonlardan birini çağırmaya karar verir ve geriye sizin işleyeceğiniz bir JSON döndürür. Siz de o fonksiyonu çalıştırır, sonucu modele geri iletirsiniz. Model parametreleri de kendisi doldurur — mesela "Saat kaç?" dediğinizde { "zaman_dilimi": "Europe/Istanbul" } gibi bir JSON üretir. OpenAI'in function calling'i, Anthropic'in tool use'u, Google'ın function declaration'ı hep aynı mantık. Tool calling olmadan agent dediğimiz yapılar çalışamaz — bu mekanizma model ile dış dünya arasındaki köprüdür.

**Örnek:** Tools'u OpenAI API'sine tanımlamak:

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_hava_durumu",
            "description": "Belirtilen şehirdeki hava durumunu getirir",
            "parameters": {
                "type": "object",
                "properties": {
                    "sehir": {
                        "type": "string",
                        "description": "Şehir adı, örn: İstanbul"
                    }
                },
                "required": ["sehir"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "İstanbul'da hava nasıl?"}],
    tools=tools,
    tool_choice="auto"
)

# Model bize çağırmak istediği fonksiyonu ve parametrelerini söyler
tool_call = response.choices[0].message.tool_calls[0]
print(f"Çağrılacak fonksiyon: {tool_call.function.name}")
print(f"Parametreler: {tool_call.function.arguments}")
# {"sehir": "İstanbul"}
```

**İlişkili:** AI Agent, ReAct Loop, Tool Schema, JSON Mode

---

### ReAct Loop (Reasoning + Acting Döngüsü)
**Ne işe yarar:** Modelin her adımda önce düşünüp (Reasoning) sonra harekete geçtiği (Acting) bir döngüsel çalışma prensibi.

**Detaylı:** ReAct, "Reasoning" ve "Acting" kelimelerinin birleşimi. Bir agent çalışırken şöyle bir döngü izler: 1) Şu anki durumu değerlendir ve ne yapman gerektiğini düşün (Thought) 2) Bir araç çağır (Action) 3) Araçtan gelen sonucu gözlemle (Observation) 4) Hedefe ulaştıysan cevap ver, ulaşmadıysan 1'e dön. Bu döngü sayesinde model her adımda ne yaptığını ve neden yaptığını açıklar — bu da hem debug etmeyi kolaylaştırır hem de modelin kendi hatalarını fark edip düzeltmesini sağlar. Şöyle bir senaryo düşünün: "En yakın kahveciyi bul" dersiniz. Model önce konumunuzu almak için bir fonksiyon çağırır (Action), konumu alır (Observation), sonra o konuma en yakın kahvecileri bulmak için başka bir API çağırır (Action), sonuçları alır ve size cevap verir. ReAct, 2022'de Google ve Princeton araştırmacılarının yayınladığı bir makaleyle popüler oldu ve günümüzde neredeyse tüm agent framework'lerinin temelini oluşturuyor.

**Örnek:** ReAct döngüsünün mantığını gösteren basit bir örnek:

```python
import json

def react_loop(model, tools, kullanici_mesaji):
    mesajlar = [{"role": "user", "content": kullanici_mesaji}]
    max_adim = 5
    
    for adim in range(max_adim):
        # Düşün + Aksiyon al
        cevap = model.generate(mesajlar, tools=tools)
        mesajlar.append(cevap)
        
        if cevap.get("finish_reason") == "stop":
            return cevap["content"]  # Hedefe ulaştık
        
        # Araç çağrısı varsa çalıştır
        if cevap.get("tool_calls"):
            for tc in cevap["tool_calls"]:
                func = tools[tc.function.name]
                sonuc = func(**json.loads(tc.function.arguments))
                # Observation ekle
                mesajlar.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(sonuc)
                })
    
    return "Maksimum adım aşıldı."

# Thought → Action → Observation → Thought → ... → Final Answer
```

**İlişkili:** AI Agent, Tool Calling, Reflection, Planning

---

### Multi-Agent System (Çoklu Ajan Sistemi)
**Ne işe yarar:** Birden çok AI agent'ın birbirleriyle iletişim kurarak, iş birliği yaparak veya rekabet ederek karmaşık görevleri çözmesini sağlayan mimari.

**Detaylı:** Bazen tek bir agent her şeyi yapamaz. Bir projeyi düşünün: bir araştırmacı, bir yazar ve bir editör var. Her biri farklı uzmanlıkta. Multi-agent sistemler de tam olarak böyle çalışır. Her agent'ın farklı bir rolü, farklı araçları ve farklı bir hedefi vardır. Örneğin bir araştırma agent'ı web'de arama yapar, bir yazma agent'ı o araştırmayı makaleye döker, bir eleştiri agent'ı da yazıyı kontrol eder. Birbirleriyle mesajlaşarak, birbirlerinin çıktılarını kullanarak ilerlerler. Bu yaklaşım, tek bir monolitik agent'a göre birçok avantaj sağlar: her agent daha küçük ve odaklıdır, işlemler paralel çalışabilir, hatayı izole etmek kolaydır. Ama dezavantajları da var: iletişim maliyeti (token sayısı artar), tasarım karmaşıklığı, ve agent'ların birbirini yanlış anlaması veya sonsuz döngüye girmesi gibi riskler. AutoGen, CrewAI, LangGraph gibi framework'ler bu mimariyi kurmayı kolaylaştırıyor.

**Örnek:** CrewAI ile iki agent'lı bir ekip:

```python
from crewai import Agent, Task, Crew

arastirmaci = Agent(
    role="Araştırmacı",
    goal="Güncel teknoloji haberlerini topla",
    backstory="Teknoloji dünyasını takip eden meraklı bir araştırmacısın",
    tools=[web_arama_araci],
    verbose=True
)

yazar = Agent(
    role="Teknik Yazar",
    goal="Araştırmacının bulduklarını anlaşılır bir bültene dönüştür",
    backstory="Karmaşık konuları basitleştiren bir yazarsın",
    verbose=True
)

arastirma_gorevi = Task(
    description="Bugünün en önemli 3 yapay zeka haberini bul",
    agent=arastirmaci
)

yazma_gorevi = Task(
    description="Haftalık bülteni yaz",
    agent=yazar
)

ekip = Crew(agents=[arastirmaci, yazar], tasks=[arastirma_gorevi, yazma_gorevi])
sonuc = ekip.kickoff()
```

**İlişkili:** AI Agent, Orchestrator Agent, Tool Calling, Agent Communication

---

### Orchestrator Agent (Orkestratör Ajan)
**Ne işe yarar:** Bir multi-agent sistemde diğer agent'ları yöneten, görev dağıtan ve iş akışını koordine eden merkezi agent.

**Detaylı:** Orkestratör, bir şef gibidir — sahnede enstrüman çalmaz ama herkesin doğru anda, doğru şeyi çalmasını sağlar. Multi-agent sistemlerde orkestratör agent, gelen büyük görevi analiz eder, parçalara böler, uygun alt agent'lara dağıtır, onların çıktılarını toplar ve nihai sonucu birleştirir. Orkestratörün en önemli yeteneği: hangi agent'ın hangi işte uzman olduğunu bilmesi ve gerektiğinde yönlendirme yapmasıdır. Mesela bir müşteri destek sisteminde: önce mesajın türünü belirler (sipariş takibi, iade, teknik sorun), sonra doğru agent'a yönlendirir. Eğer iade agent'ı çözemediyse, insan operatöre aktarır. Orkestratör olmazsa her agent her şeye karışır, kaos çıkar. LangGraph'da StateGraph kullanarak orkestratör yapıları kurmak çok yaygın. Bazı sistemlerde orkestratör ayrı bir model (daha güçlü, daha pahalı) ile çalışırken, alt agent'lar daha hafif modeller kullanır.

**Örnek:** LangGraph ile orkestratör mantığı:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

class IsDurumu(TypedDict):
    mesaj: str
    kanal: str
    tarih: str

# Alt agent'lar
def siparis_agent(state: IsDurumu):
    return {"cevap": f"Siparişiniz {state['tarih']} itibarıyla kargoda"}

def iade_agent(state: IsDurumu):
    return {"cevap": "İade süreciniz başlatıldı, 3 iş günü içinde tamamlanır"}

# Orkestratör — yönlendirme kararını verir
def orkestrator(state: IsDurumu):
    if "sipariş" in state["mesaj"].lower():
        return "siparis"
    elif "iade" in state["mesaj"].lower():
        return "iade"
    else:
        return "insan_destek"

# Graf yapısı
graph = StateGraph(IsDurumu)
graph.add_node("orkestrator", orkestrator)
graph.add_node("siparis", siparis_agent)
graph.add_node("iade", iade_agent)

graph.set_conditional_edge_mapping("orkestrator", {
    "siparis": "siparis",
    "iade": "iade",
    "insan_destek": END
})
```

**İlişkili:** Multi-Agent System, AI Agent, State Machine, Supervisor Agent

---

### Agent Memory (Bellek — Buffer, Summary, Entity)
**Ne işe yarar:** Agent'ın önceki konuşmaları, etkileşimleri ve öğrendiği bilgileri saklayarak uzun süreli bağlam sağlaması.

**Detaylı:** Bir modelin context window'u sınırlıdır — her şeyi sonsuza kadar hatırlayamaz. Agent memory işte bu sınırı aşmak için kullanılan tekniklerin bütünüdür. Üç ana türü var. Buffer memory: en basiti, konuşma geçmişini olduğu gibi saklar ve token limitine takılınca en eski mesajları siler. Summary memory: eski konuşmaları özetleyip o özeti saklar, böylece detay kaybolmaz ama token tasarrufu sağlanır. Entity memory: konuşmada geçen kişi, yer, tarih gibi önemli varlıkları çıkarıp ayrı bir tabloda tutar — "Ahmet'in kedisinin adı Tekir" gibi bilgileri her konuşmada hatırlar. Bir de hybrid yaklaşım var: son N mesaj buffer'da, ondan öncekiler özetlenmiş şekilde tutulur. Bellek yönetimi bir agent sisteminde en kritik konulardan biridir — çünkü ya her şeyi hatırlayıp token'i şişirirsiniz ya da çok fazla unutup bağlamı kaybedersiniz. LangChain'in Memory modülleri, bu üç yaklaşımı da destekler.

**Örnek:** LangChain'de farklı bellek türleri:

```python
# Buffer Memory
from langchain.memory import ConversationBufferMemory
buffer_memory = ConversationBufferMemory(k=5)  # son 5 mesaj

# Summary Memory
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    max_token_limit=500
)

# Entity Memory
from langchain.memory import ConversationEntityMemory
entity_memory = ConversationEntityMemory(llm=ChatOpenAI(model="gpt-4o-mini"))

# Kullanım
entity_memory.save_context(
    {"input": "Köpeğimin adı Karabaş"},
    {"output": "Harika bir isim!"}
)
entity_memory.load_memory_variables({})
# {'entities': {'Karabaş': 'Bir köpeğin adı'}}
```

**İlişkili:** AI Agent, Context Window, RAG, Token Management

---

### Reflection (Yansıtma / Kendini Değerlendirme)
**Ne işe yarar:** Modelin kendi ürettiği çıktıyı eleştirerek hatalarını fark etmesi ve iyileştirmesi süreci.

**Detaylı:** Reflection, bir modelin kendi yazdığı kodu veya metni tekrar okuyup "Acaba burası yanlış mı?" diye sormasıdır. Basit bir teknik gibi görünse de çok etkilidir. Şöyle işler: model önce bir cevap üretir, sonra o cevabı bir "eleştirmen" modeline verir (aynı model olabilir, farklı bir prompt ile). Eleştirmen hataları, eksikleri, iyileştirme alanlarını listeler. Sonra orijinal model bu eleştirileri alır ve düzeltilmiş bir versiyon üretir. Bu döngü birkaç kez tekrarlanabilir. Özellikle kod yazmada çok işe yarar — model önce bir kod yazar, sonra kodu derlemeyi dener, hata alırsa düzeltir. Reflection loop'ları sayesinde GPT-4 gibi modeller tek seferde yazdıkları koda göre çok daha doğru çıktılar üretebiliyor. Self-Refine, Reflexion gibi popüler tekniklerin temelinde de bu mantık yatıyor. Tabii her adımda token harcadığınız için maliyet artıyor, bu yüzden sadece kritik görevlerde kullanmak daha mantıklı.

**Örnek:** Reflection döngüsü:

```python
from openai import OpenAI

client = OpenAI()

def reflect_ve_duzelt(kod):
    # Önce eleştiri al
    elestiri = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen bir kod eleştirmenisin. Verilen koddaki hataları ve iyileştirme alanlarını bul."},
            {"role": "user", "content": kod}
        ]
    ).choices[0].message.content
    
    # Sonra düzelt
    duzeltilmis = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sen bir yazılım geliştiricisin. Verilen eleştirilere göre kodu düzelt."},
            {"role": "user", "content": f"Kod:\n{kod}\n\nEleştiri:\n{elestiri}\n\nLütfen düzeltilmiş versiyonu yaz."}
        ]
    ).choices[0].message.content
    
    return duzeltilmis

# Birden çok tur refinement
kod = """def bol(a, b):
    return a / b"""

for tur in range(3):
    kod = reflect_ve_duzelt(kod)
    print(f"Tur {tur+1} tamamlandı")
```

**İlişkili:** AI Agent, ReAct Loop, Self-Refine, Self-Critique, Error Handling

---

### Planning (Planlama — Plan & Execute)
**Ne işe yarar:** Agent'ın karmaşık bir görevi başlamadan önce adımlara ayırarak bir plan oluşturması ve sonra o planı adım adım uygulaması.

**Detaylı:** ReAct döngüsü her seferinde bir sonraki adımı düşünür — kısa vadeli bir yaklaşımdır. Planning ise farklı: adım adım ilerlemek yerine önce tüm planı çıkarır, sonra uygulamaya geçer. "Bir araştırma makalesi yaz" dediğinizde, model önce şöyle bir plan yapar: 1) Konuyla ilgili 5 kaynak taraması yap 2) Kaynakları özetle 3) Taslak oluştur 4) Giriş bölümünü yaz 5) ... gibi. Bu yaklaşımın avantajı, modelin baştan sonra ne yapacağını bilmesi ve plana sadık kalmasıdır. Ama dezavantajı da var: plan çok katı olursa gerçek dünyada beklenmedik durumlarla baş edemez. Bu yüzden modern yaklaşımlar ikisini birleştirir — önce genel bir plan yapılır (Plan), sonra her adımda ReAct loop'u ile esnek kalınır (Execute). Plan-Execute pattern'i özellikle çok adımlı, bağımlılıkları olan görevlerde işe yarar. Mesela bir yemek tarifi uygulaması: önce hangi malzemeler lazım (Plan), sonra her adımda tarifi uygula (Execute), ama "yok bu malzeme bitmiş" gibi durumlarda alternatif üret (ReAct).

**Örnek:** Plan-Execute pattern'i:

```python
def plan_ve_calistir(gorev, model, araclar):
    # AŞAMA 1: Plan oluştur
    plan_prompt = f"""
    Görev: {gorev}
    
    Bu görevi gerçekleştirmek için adım adım bir plan yap.
    Her adımı numaralandır ve hangi aracı kullanacağını belirt.
    
    Plan:
    """
    
    plan = model.generate(plan_prompt)
    adimlar = plan.split("\n")
    
    # AŞAMA 2: Planı uygula
    for adim in adimlar:
        if not adim.strip():
            continue
        sonuc = react_adim(adim, model, araclar)
        print(f"Adım tamamlandı: {adim[:50]}...")
    
    # AŞAMA 3: Sonuçları birleştir
    final = model.generate(f"Tüm adımlar tamamlandı. Nihai sonucu hazırla. Plan: {plan}")
    return final

# Örnek plan çıktısı:
# Plan:
# 1. Kullanıcının email'ini oku (email_araci)
# 2. Önemli maddeleri not al (not_araci)
# 3. Takvimdeki boş zamanları kontrol et (takvim_araci)
# 4. Toplantı daveti gönder (davet_araci)
```

**İlişkili:** AI Agent, ReAct Loop, Task Decomposition, Chain-of-Thought, State Machine

---

### Sandbox Execution (Korumalı Alan Çalıştırma)
**Ne işe yarar:** Agent'ın ürettiği kodun, veritabanına veya ana sisteme zarar vermeden güvenli bir ortamda çalıştırılmasını sağlayan mekanizma.

**Detaylı:** Bir agent'ın kod yazıp çalıştırmasına izin veriyorsanız, bu kodun sisteminizi çökertmesini, dosyalarınızı silmesini veya veritabanınıza izinsiz erişmesini engellemeniz gerekir. İşte sandbox bu güvenlik katmanıdır. Agent'ın kodu, ana sunucunuzda değil, izole edilmiş bir ortamda çalışır — genelde bir Docker container, bir VM veya e2b.dev gibi bulut tabanlı bir çözüm. Bu ortamda ağ erişimi kısıtlanır, dosya sistemi sınırlanır, işlemci ve bellek kullanımı tavan yapılır. Agent yanlışlıkla "rm -rf /" yazsa bile sadece sandbox içindeki geçici dosyalar silinir, ana sisteminize bir şey olmaz. Code Interpreter (ChatGPT'nin kod çalıştırma özelliği) da aslında bir sandbox'dır — her çalıştırma için yeni bir container açılır, kullanıcı hiçbir sistem dosyasına erişemez. Sandbox'lar aynı zamanda çoklu kullanıcı ortamlarında güvenlik için de kritiktir: bir kullanıcının agent'ı diğer kullanıcının verilerine erişmesin diye.

**Örnek:** Docker sandbox'ında kod çalıştırma:

```python
import docker
import tempfile

client = docker.from_env()

def sandbox_calistir(kod: str) -> str:
    # Geçici bir Python dosyası oluştur
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w") as f:
        f.write(kod)
        f.flush()
        
        # İzole container'da çalıştır
        container = client.containers.run(
            image="python:3.11-slim",
            command=f"python /tmp/kod.py",
            volumes={f.name: {"bind": "/tmp/kod.py", "mode": "ro"}},
            mem_limit="256m",        # 256 MB bellek limiti
            network_disabled=True,   # internet yok
            read_only=True,          # dosya sistemi salt okunur
            remove=True,             # container bitince temizle
            timeout=30               # 30 saniye timeout
        )
        return container.decode("utf-8")

# Agent sandbox'ta kod çalıştırır
agent_kodu = "print('Merhaba dünya!')"
try:
    sonuc = sandbox_calistir(agent_kodu)
    print(f"Çıktı: {sonuc}")
except Exception as e:
    print(f"Sandbox hatası: {e}")
```

E2B (e2b.dev) gibi yönetilen sandbox'lar da var:
```python
from e2b_code_interpreter import CodeInterpreter

with CodeInterpreter() as sandbox:
    exec = sandbox.notebook.exec_cell(
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        x = np.linspace(0, 10, 100)
        plt.plot(x, np.sin(x))
        plt.savefig('/tmp/grafik.png')
        """
    )
    # Grafik dosyasını al
    dosya = sandbox.files.read("/tmp/grafik.png")
```

**İlişkili:** AI Agent, Tool Calling, Code Interpreter, Security, Container

---

### Human-in-the-Loop (Devrede İnsan — HITL)
**Ne işe yarar:** Agent'ın kritik kararlarında bir insanın onayını almasını veya yönlendirmesini gerektiren kontrol mekanizması.

**Detaylı:** Tam otonom bir agent her zaman iyi bir fikir değildir. Ya yanlış bir e-posta gönderirse? Ya müşteriye yanlış fatura keserse? Ya bir silme işlemi yaparsa? Human-in-the-loop, bu riskleri azaltmak için kritik adımlarda bir insanın devreye girmesini sağlar. Mesela bir agent "tüm eski e-postaları sil" dediğinde, bu işlem otomatik yapılmaz, önce size onay sorar. Siz "Evet sil" derseniz yapar, "Hayır" derseniz yapmaz. HITL üç seviyede olabilir: 1) Stratejik — sadece büyük kararlarda onay (örn. fatura kesmek) 2) Taktiksel — her adımda onay (örn. her API çağrısında) 3) Exception-based — sadece hata veya belirsizlik durumunda onay. En verimli yaklaşım genelde üçüncüsüdür: agent normal işleyişte otonomdur, emin olamadığı veya riskli gördüğü durumlarda insana sorar. Slack, e-posta veya web UI üzerinden onay akışları kurmak yaygındır.

**Örnek:** LangGraph'da human-in-the-loop:

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver

class Durum(TypedDict):
    mesaj: str
    onay: bool

def eposta_gonder(state: Durum):
    # Kritik adım - insan onayı bekleniyor
    return {"onay_bekleniyor": True, "mesaj": state["mesaj"]}

def onay_kontrol(state: Durum):
    if state.get("onay"):
        return "gonder"
    else:
        return END

# Checkpoint ile state'i kaydet — insan onayı için bekle
checkpointer = MemorySaver()

graph = StateGraph(Durum)
graph.add_node("onay_bekle", eposta_gonder)
graph.add_node("gonder", lambda s: print(f"E-posta gönderildi: {s['mesaj']}"))

graph.add_conditional_edges("onay_bekle", onay_kontrol, {
    "gonder": "gonder",
    END: END
})

# Çalıştır ve bekle
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "123"}}

# Agent çalışır, onay bekler
app.invoke({"mesaj": "Tüm eski kayıtlar silinecek"}, config)

# ... burada insan onay verir veya vermez ...
app.update_state(config, {"onay": True})  # İnsan onayladı
app.invoke(None, config)  # Kaldığı yerden devam
```

Slack üzerinden onay akışı:
```python
def onay_istemi_slack(gorev: str, agent_id: str):
    slack_client.chat_postMessage(
        channel="@berkin",
        text=f"🤖 Agent '{agent_id}' şu işlemi yapmak istiyor:\n\n{gorev}\n\nOnaylıyor musun?",
        blocks=[
            {"type": "section", "text": {"type": "mrkdwn", "text": f"*Onay Gerekiyor* 🛑\n{gorev}"}},
            {"type": "actions", "elements": [
                {"type": "button", "text": "✅ Onayla", "value": "onay", "action_id": f"approve_{agent_id}"},
                {"type": "button", "text": "❌ Reddet", "value": "red", "action_id": f"reject_{agent_id}"}
            ]}
        ]
    )
```

**İlişkili:** AI Agent, Safety, Guardrails, Approval Flow, Review Mechanisms
