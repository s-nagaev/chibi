<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Chibi Logosu"></h1>

<p align="center">
  <strong>Dijital yol arkadaşınız. Bir araç değil. Bir ortak.</strong><br/>
  <span>Gerçek işleri tamamlamak için birden fazla yapay zekâ sağlayıcısını, aracı ve alt ajanı orkestre eden, self-hosted ve asenkron Telegram botu.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="PyPI İndirmeleri"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="Mimariler"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="Lisans"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Dokümantasyon"></a>
</p>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="../README.md">English</a> •
  <a href="README.es.md">Español</a> •
  <a href="README.pt-BR.md">Português (Brasil)</a> •
  <a href="README.uk.md">Українська</a> •
  <a href="README.id.md">Bahasa Indonesia</a> •
  <strong>Türkçe</strong> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

Chibi, “bir AI aracı”ndan fazlasına ihtiyaç duyduğunuzu fark ettiğiniz an için tasarlandı. Modelleri koordine edebilen, işleri arka planda çalıştırabilen ve sistemlerinize entegre olabilen—siz prompt’ları sürekli takip etmeden—bir **ortak**.

**Chibi**, birden fazla AI sağlayıcısını ve aracı orkestre ederek sonuç üreten; asenkron, self-hosted **Telegram tabanlı dijital yol arkadaşıdır**: kod değişiklikleri, araştırma sentezleri, medya üretimi ve operasyonel görevler.

---

## Neden Chibi

- **Tek arayüz (Telegram).** Mobil/masaüstü/web, her zaman yanınızda.
- **Sağlayıcıdan bağımsız.** Her görev için en iyi modeli kullanın—vendor lock-in olmadan.
- **Otonom yürütme.** Alt ajanlar paralel çalışır; uzun işler asenkron yürür.
- **Araçlarla bağlantılı.** Dosya sistemi + terminal + MCP entegrasyonları (GitHub, tarayıcı, veritabanları vb.).
- **Self-hosted.** Veriniz, anahtarlarınız, kurallarınız.

---

## Desteklenen AI sağlayıcıları (ve endpoint’ler)

Chibi, tek bir sohbetin arkasında birden fazla sağlayıcıyı destekler. Bir anahtar ekleyin ya da birden fazlasını—Chibi göreve göre yönlendirebilir.

### LLM sağlayıcıları

- **OpenAI** (GPT modelleri)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **DeepSeek**
- **Alibaba Cloud** (Qwen)
- **xAI** (Grok)
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **ZhipuAI** (GLM modelleri)
- **Novita AI**
- **Xiaomi** (MiMo modelleri)
- **Melious**
- **Cheaper Inference**
- **DeepInfra**
- **Together AI**
- **Fireworks AI**
- **Nebius** (Token Factory)
- **Baseten**
- **SambaNova**
- **SiliconFlow**
- **Parasail**
- **Featherless**
- **OpenRouter** (birçok modele birleşik erişim)
- **Cloudflare Workers AI** (birçok açık kaynak model)

### OpenAI uyumlu endpoint’ler (self-host / local)

- **Ollama**
- **vLLM**
- **LM Studio**
- **Herhangi bir** OpenAI uyumlu API

### Multimodal sağlayıcılar (opsiyonel)

- **Görseller:** Google (Imagen, Nano Banana), OpenAI (GPT Image), Alibaba (Qwen Image, Wan), xAI (Grok Imagine), ZhipuAI (GLM Image), MiniMax, Cheaper Inference (Nano Banana Pro)
- **Müzik:** Suno
- **Ses:** ElevenLabs, OpenAI (GPT-4o Transcribe / TTS), MiniMax (TTS)

> Model erişilebilirliği, yapılandırdığınız sağlayıcı anahtarlarına ve etkinleştirdiğiniz özelliklere bağlıdır.

---

## 🚀 Hızlı başlangıç (pip)

Chibi'yi pip ile kurun ve bir komut satırı uygulaması olarak çalıştırın:

```bash
# Paketi kur
pip install chibi-bot

# Ajanı ayarla (API anahtarları ekle, ayarları güncelle vb.)
chibi config

# Botu başlat
chibi start
```

Bot arka plan hizmeti olarak çalışır. Yönetmek için CLI komutlarını kullanın.

### CLI Komutları

| Komut           | Açıklama                                |
|-----------------|-----------------------------------------|
| `chibi start`   | Botu arka plan hizmeti olarak başlatır  |
| `chibi stop`    | Çalışan botu durdurur                   |
| `chibi restart` | Botu yeniden başlatır                   |
| `chibi config`  | Yapılandırma oluşturur veya düzenler    |
| `chibi logs`    | Bot günlüklerini görüntüler             |

---

## Yerel istemci arayüzleri

Chibi, yerel istemcilerine sürümlü yerel JSONL protokolü üzerinden hizmet verebilir:

```bash
chibi stdio --tui
chibi stdio --vscode
chibi stdio --pycharm
chibi stdio --neovim
```

Komut etkileşimli kullanım için değil, istemci arayüzü tarafından başlatılmak üzere tasarlanmıştır. Chibi `v1` protokolünün sahibidir ve aynı sürümü görüşen uyumlu istemcileri destekler. stdio işlemi stdout'a yalnızca protokol çerçevelerini, stderr'e ise tanıları yazar. Chibi'nin mevcut komut, araç, izin ve moderasyon davranışı yetkili olmaya devam eder; istemci herhangi bir araç politikası katmanı eklemez.

---
## 🚀 Hızlı başlangıç (Docker)

`docker-compose.yml` oluşturun:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Zorunlu
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # Ya da başka bir sağlayıcı
      # Gerektikçe daha fazla API anahtarı ekleyin
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) [@BotFather](https://t.me/BotFather) üzerinden bir bot token alın

2) Gizli bilgileri `.env` içine koyun

3) Çalıştırın:

```bash
docker-compose up -d
```

Sonraki adımlar:
- **Kurulum rehberi:** https://chibi.bot/installation
- **Yapılandırma referansı:** https://chibi.bot/configuration

---

## 🔑 API Anahtarlarını Alma

Her sağlayıcı kendi API anahtarını gerektirir. İşte doğrudan bağlantılar:

**Ana Sağlayıcılar:**
- **OpenAI** (GPT, DALL·E): [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **Anthropic** (Claude): [console.anthropic.com](https://console.anthropic.com/)
- **Google** (Gemini, Nano Banana, Imagen): [aistudio.google.com/apikey](https://aistudio.google.com/app/apikey)
- **DeepSeek**: [platform.deepseek.com](https://platform.deepseek.com/)
- **xAI** (Grok): [console.x.ai](https://console.x.ai/)
- **Alibaba** (Qwen, Wan): [modelstudio.console.alibabacloud.com](https://modelstudio.console.alibabacloud.com?tab=playground#/api-key)
- **Mistral AI**: [console.mistral.ai](https://console.mistral.ai/)
- **Moonshot** (Kimi): [platform.moonshot.cn](https://platform.moonshot.cn/)
- **MiniMax** (Voice, MiniMax-M2.x): [minimax.io](https://www.minimax.io)
- **Cloudflare Workers AI**: [dash.cloudflare.com/profile/api-tokens](https://dash.cloudflare.com/profile/api-tokens)
- **OpenRouter** (birçok modele birleşik erişim): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **ZhipuAI** (GLM, CogView): [z.ai/manage-apikey/apikey-list](https://z.ai/manage-apikey/apikey-list)

**Yaratıcı Araçlar:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Kurulum talimatlarıyla birlikte tam kılavuz:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## İlk 5 dakikada şunları deneyin

Deploy ettikten sonra bunları Telegram’a yapıştırın.

1) **Planlama + yürütme**
> Hedefimi netleştirmek için bana 3 soru sor, sonra bir plan öner ve 1. adımı uygula.

2) **Paralel çalışma (alt ajanlar)**
> 3 alt ajan oluştur: biri seçenekleri araştırsın, biri öneri taslağı hazırlasın, biri riskleri listelesin. Tek bir karar döndür.

3) **Ajan modu (araçlar)**
> Proje dosyalarını incele ve bu repo’nun ne yaptığını özetle. Sonra 5 iyileştirme öner ve bir kontrol listesi aç.

4) **Arka plan görevi**
> Bir arka plan görevi başlat: X hakkında kaynakları topla ve 30 dakika içinde bir sentez teslim et. Beni güncel tut.

---

## Chibi’yi farklı kılan

### 🎭 Çoklu sağlayıcı orkestrasyonu
Chibi, aynı konuşma içinde sağlayıcı değiştirirken bağlamı koruyabilir veya her adım için en iyi modeli seçebilir—**maliyet**, **yetenek** ve **hız** dengesini gözeterek.

### 🤖 Otonom ajan yetenekleri
- **Özyinelemeli delegasyon:** alt ajanlar kendi alt ajanlarını oluşturabilir
- **Arka plan işleme:** uzun süren işler asenkron yürür
- **Dosya sistemi erişimi:** dosyaları oku/yaz/ara/düzenle
- **Terminal çalıştırma:** komutları LLM tarafından denetlenen güvenlikle çalıştır
- **Kalıcı bellek:** konuşma geçmişi, bağlam yönetimi/özetleme ile yeniden başlatmalarda korunur
- **Skills:** ajanın gerektiğinde sistem promptuna yükleyebileceği yeniden kullanılabilir yönerge modülleri (`load_builtin_skill`)

### 🔌 MCP (Model Context Protocol) ile genişletilebilir
Chibi’yi harici araçlara ve servislere bağlayın (ya da kendinizinkini yazın):

- GitHub (PR’lar, issue’lar, code review)
- Tarayıcı otomasyonu
- Docker / bulut servisleri
- Veritabanları
- Yaratıcı araçlar (Blender, Figma)

Bir araç MCP üzerinden sunulabiliyorsa, Chibi onu kullanmayı öğrenebilir.

### 🎨 Zengin içerik üretimi
- **Görseller:** Nano Banana, Imagen, Qwen, Wan, GPT Image, Grok Imagine, GLM Image
- **Müzik:** Suno (custom mode dahil: stil/şarkı sözü/vokal)
- **Ses:** transkripsiyon + metinden sese (ElevenLabs, OpenAI, MiniMax)

---

## Kullanım senaryoları

**Geliştiriciler**
```
Siz: “Testleri çalıştır ve bozulanı düzelt. Ben frontend’e bakacağım.”
Chibi: *alt ajan oluşturur, testleri çalıştırır, hataları analiz eder, düzeltme önerir*
```

**Araştırmacılar**
```
Siz: “Kuantum bilişimdeki son gelişmeleri araştır. Yarın için bir sentez lazım.”
Chibi: *birden fazla araştırma ajanı oluşturur, kaynakları birleştirir, rapor teslim eder*
```

**Üreticiler**
```
Siz: “Cyberpunk bir şehir manzarası üret ve buna uygun bir synthwave parçası bestele.”
Chibi: *görsel üretir, müzik oluşturur, ikisini de teslim eder*
```

**Ekipler**
```
Siz: “Bu PR’ı incele ve dokümantasyonu buna göre güncelle.”
Chibi: *değişiklikleri analiz eder, iyileştirme önerir, MCP ile dokümanları günceller*
```

---

## Gizlilik, kontrol ve güvenlik

- **Self-hosted:** veriniz kendi altyapınızda kalır
- **Public Mode:** kullanıcılar kendi API anahtarlarını getirebilir (paylaşılan master key gerekmez)
- **Erişim kontrolü:** kullanıcı/grup/model whitelist
- **Depolama seçenekleri:** yerel volume’lar, Redis veya DynamoDB
- **Araç güvenliği:** ajan araçları yapılandırılabilir; terminal çalıştırma denetlenir ve kısıtlanabilir

---

### `MAX_HISTORY_TOKENS` — bağlam özetleme eşiği (varsayılanda geriye dönük uyumsuz değişiklik)

`MAX_HISTORY_TOKENS`, bağlamı yönetilebilir tutmak için Chibi'nin bir konuşmayı otomatik olarak özetlediği eşiktir. **Anlamı değişti**: artık eski yalnızca konuşma `content` + `role` sezgiseli yerine, **sağlayıcının bildirdiği gerçek prompt token sayısıyla** (sistem promptu + etkin Skills + araç şemaları + araç çağrısı bağımsız değişkenleri + ileti başına yapısal ek yük + konuşma içeriği dahil tüm giden istek) karşılaştırılır. Gerçek sayı, aynı konuşma için eski tahminden yaklaşık **4,8 kat daha büyüktür** (ölçülen döküm için `fix_context_size/context_size_accounting_analysis.md` dosyasına bakın).

- **Varsayılan yeniden tabanlandı:** `64000` → `100000`. Yeni değer, yaygın olarak desteklenen en küçük bağlam penceresini (128k token) korur: `100000`, 128k pencerenin ~%78'idir (özetleme 128k model taşmadan *önce* çalışır) ve 200k pencerenin ~%50'sidir (rahat pay bırakır). Eski `64000`, gerçek taşmadan önce asla ulaşılmayan yalnızca geçmiş tahminiydi; çünkü Kiril alfabesini ~2 kat eksik sayar ve tur başına sabit ek yükü hariç tutar (sistem promptu ~3,7k, araç şemaları ~6,8k, etkin Skills ~5,9k, `user_info` ~0,75–3k).
- **Geçiş:** `.env` dosyanızda `MAX_HISTORY_TOKENS` değerini açıkça ayarladıysanız eski değeriniz yalnızca geçmişe dayalı tahmine göre ayarlanmıştı ve artık aynı konuşma için ~4,8 kat büyük gerçek bir sayıyla karşılaştırılır. Özetlemenin en küçük modelinizin bağlam penceresi taştıktan sonra değil önce tetiklenmesi için değeri **~100k ölçeğine** yeniden ayarlayın (ör. `64000` → `100000`). Hiç ayarlamadıysanız yeni varsayılan otomatik olarak uygulanır.
- Soğuk başlangıç yedeği (işlem yeniden başladıktan sonraki ilk turda, henüz önbelleğe alınmış sağlayıcı verisi yokken) özetlemenin çalışmaya devam etmesi için eski sezgiseli kullanır.

### `REACTIVE_CONTEXT_RECOVERY` — bağlam taşmasından sonra tek seferlik yeniden deneme

`REACTIVE_CONTEXT_RECOVERY` (varsayılan: `true`), proaktif `MAX_HISTORY_TOKENS` eşiğini tamamlayan reaktif güvenlik ağıdır. Sağlayıcı, istekleri `context_length_exceeded` türünde bir hatayla reddettiğinde Chibi geçmişi otomatik özetler ve turu **tam olarak bir kez** yeniden dener. Başarılı olursa normal bir yanıt alırsınız (bağlamın sıkıştırıldığına dair kısa notla). Yeniden taşarsa veya kurtarma herhangi bir nedenle başarısız olursa tur mevcut özür yoluna döner — özetle+yeniden dene döngüsü özgün tur başına asla birden fazla çalışmaz. Devre dışı bırakmak ve önceki davranışı korumak için `false` yapın (günlük + özür, yeniden deneme yok).

### Çalışma dizini thread kapsamındadır (`WORKING_DIR`)

Ajanın terminal komutlarında kullanılan ve modele mevcut CWD olarak bildirilen çalışma dizini, seçili LLM modelinin de her thread'e bağlı olması gibi **thread kapsamındadır**.

- **Varsayılan:** eski kullanıcı düzeyi değer aracılığıyla `WORKING_DIR` ayarından gelir (varsayılan `~/chibi`); yeni kurulumlar ayarı doğrudan devralır.
- **Thread başına geçersiz kılma:** herhangi bir thread/konuşma, `set_working_dir` ajan aracıyla çalışma dizinini **izole şekilde** geçersiz kılabilir (yalnızca LLM yönlendirmeli; slash komutu yoktur, `FILESYSTEM_ACCESS` etkin olduğunda kullanılabilir). Böylece farklı thread'lerde iki ajan birbirini etkilemeden farklı projelerde eşzamanlı çalışabilir.
- **Çözümleme sırası:** thread geçersiz kılması → eski kullanıcı düzeyi dizin → `WORKING_DIR` ayarı.
- **Yol normalleştirme:** ayarladığınız değerler kaydedilirken mutlak yollara genişletilir (`~/x`, `/abs/x` olur); dokunulmayan varsayılanlar ham biçimlerini korur.
- Bir thread içinde oluşturulan **alt ajanlar** o thread'in çalışma dizinini paylaşır; aynı etkin yol, sistem promptlarına ve araç çağrılarına eklenir.
- Geçersiz kılmalar **thread klonlamasında korunur**: `/new_thread_with_current_context`, çalışma dizinini iletiler ve model tercihleriyle birlikte taşır.

---

## Dokümantasyon

- **Buradan başlayın:** https://chibi.bot
- Giriş ve felsefe: https://chibi.bot/introduction
- Kurulum: https://chibi.bot/installation
- Yapılandırma: https://chibi.bot/configuration
- Ajan modu: https://chibi.bot/agent-mode
- MCP rehberi: https://chibi.bot/guides/mcp
- Destek / troubleshooting: https://chibi.bot/support

---

## Sistem gereksinimleri

- **Minimum:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Mimariler:** `linux/amd64`, `linux/arm64`
- **Bağımlılıklar:** Docker (ve opsiyonel Docker Compose)

---

## Katkıda bulunma

- Issues: https://github.com/s-nagaev/chibi/issues
- PR’lar: https://github.com/s-nagaev/chibi/pulls
- Tartışmalar: https://github.com/s-nagaev/chibi/discussions

Göndermeden önce lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını okuyun.

---

## Lisans

MIT — [LICENSE](LICENSE) dosyasına bakın.

---

<p align="center">
  <strong>Dijital yol arkadaşınızla tanışmaya hazır mısınız?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Başlayın →</strong></a>
</p>
