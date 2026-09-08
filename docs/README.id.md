<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Logo Chibi"></h1>

<p align="center">
  <strong>Teman digital Anda. Bukan alat. Mitra.</strong><br/>
  <span>Bot Telegram self-hosted dan asinkron yang mengorkestrasi banyak penyedia AI, tools, dan sub-agent untuk menyelesaikan pekerjaan nyata.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://pypi.org/project/chibi-bot/"><img src="https://static.pepy.tech/personalized-badge/chibi-bot?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=pip+installs" alt="Unduhan PyPI"></a>  
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/arch-arm64%20%7C%20amd64-informational" alt="Arsitektur"></a>
  <a href="https://github.com/s-nagaev/chibi/blob/main/LICENSE"><img src="https://img.shields.io/github/license/s-nagaev/chibi" alt="Lisensi"></a>
  <a href="https://chibi.bot"><img src="https://img.shields.io/badge/docs-chibi.bot-blue" alt="Dokumentasi"></a>
</p>

<p align="center">
  <strong>🌍 Read this in other languages:</strong><br/>
  <a href="../README.md">English</a> •
  <a href="README.es.md">Español</a> •
  <a href="README.pt-BR.md">Português (Brasil)</a> •
  <a href="README.uk.md">Українська</a> •
  <strong>Bahasa Indonesia</strong> •
  <a href="README.tr.md">Türkçe</a> •
  <a href="README.ru.md">Русский</a> •
  <a href="README.ja.md">日本語</a> •
  <a href="README.zh-TW.md">繁體中文</a> •
  <a href="README.zh-CN.md">简体中文</a>
</p>

---

Chibi dibuat untuk momen ketika Anda sadar bahwa Anda butuh lebih dari sekadar “alat AI”. Anda butuh **mitra** yang bisa mengoordinasikan model, menjalankan pekerjaan di background, dan terintegrasi dengan sistem Anda—tanpa Anda harus terus mengawasi prompt.

**Chibi** adalah **teman digital berbasis Telegram** yang asinkron dan self-hosted, yang mengorkestrasi banyak penyedia AI dan tools untuk menghasilkan outcome: perubahan kode, sintesis riset, pembuatan media, dan tugas operasional.

---

## Mengapa Chibi

- **Satu antarmuka (Telegram).** Mobile/desktop/web, selalu bersama Anda.
- **Agnostik penyedia.** Gunakan model terbaik untuk tiap tugas—tanpa vendor lock-in.
- **Eksekusi otonom.** Sub-agent bekerja paralel; tugas panjang berjalan asinkron.
- **Terhubung ke tools.** Filesystem + terminal + integrasi MCP (GitHub, browser, DB, dll.).
- **Self-hosted.** Data Anda, kunci Anda, aturan Anda.

---

## Penyedia AI yang didukung (dan endpoint)

Chibi mendukung banyak penyedia dalam satu percakapan. Tambahkan satu API key atau banyak—Chibi bisa melakukan routing per tugas.

### Penyedia LLM

- **OpenAI** (model GPT)
- **Anthropic** (Claude)
- **Google** (Gemini)
- **DeepSeek**
- **Alibaba Cloud** (Qwen)
- **xAI** (Grok)
- **Mistral AI**
- **Moonshot AI**
- **MiniMax**
- **ZhipuAI** (model GLM)
- **Novita AI**
- **Xiaomi** (model MiMo)
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
- **OpenRouter** (akses terpadu ke banyak model)
- **Cloudflare Workers AI** (banyak model open-source)

### Endpoint kompatibel OpenAI (self-host / lokal)

- **Ollama**
- **vLLM**
- **LM Studio**
- **API kompatibel OpenAI apa pun**

### Penyedia multimodal (opsional)

- **Gambar:** Google (Imagen, Nano Banana), OpenAI (GPT Image), Alibaba (Qwen Image, Wan), xAI (Grok Imagine), ZhipuAI (GLM Image), MiniMax, Cheaper Inference (Nano Banana Pro)
- **Musik:** Suno
- **Suara:** ElevenLabs, OpenAI (GPT-4o Transcribe / TTS), MiniMax (TTS)

> Ketersediaan model yang tepat bergantung pada API key yang Anda konfigurasi dan fitur yang diaktifkan.

---

## 🚀 Quick Start (pip)

Instal Chibi melalui pip dan jalankan sebagai aplikasi baris perintah:

```bash
# Menginstal paket
pip install chibi-bot

# Mengatur agen (tambahkan kunci API, perbarui pengaturan, dll.)
chibi config

# Memulai bot
chibi start
```

Bot akan berjalan sebagai layanan background. Gunakan perintah CLI untuk mengelolanya.

### Perintah CLI

| Perintah        | Deskripsi                                |
|-----------------|------------------------------------------|
| `chibi start`   | Memulai bot sebagai layanan background   |
| `chibi stop`    | Menghentikan bot yang berjalan           |
| `chibi restart` | Memulai ulang bot                        |
| `chibi config`  | Membuat atau menyunting konfigurasi      |
| `chibi logs`    | Menampilkan log bot                      |

---

## Klien lokal

Chibi dapat melayani klien lokal melalui protokol JSONL lokal berversi:

```bash
chibi stdio --tui
chibi stdio --vscode
chibi stdio --pycharm
chibi stdio --neovim
```

Perintah ini ditujukan untuk dijalankan oleh frontend klien, bukan digunakan secara interaktif. Chibi memiliki protokol `v1` dan mendukung klien kompatibel yang menegosiasikan versi sama. Proses stdio hanya menulis frame protokol ke stdout dan diagnostik ke stderr. Perilaku Chibi yang ada untuk perintah, tool, izin, dan moderasi tetap berwenang; klien tidak menambahkan lapisan kebijakan tool.

---
## 🚀 Quick start (Docker)

Buat `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chibi:
    image: pysergio/chibi:latest
    restart: unless-stopped
    environment:
      TELEGRAM_BOT_TOKEN: ${TELEGRAM_BOT_TOKEN}  # Wajib
      OPENAI_API_KEY: ${OPENAI_API_KEY}          # Atau penyedia lain
      # Tambahkan API key lain sesuai kebutuhan
    volumes:
      - chibi_data:/app/data

volumes:
  chibi_data: {}
```

1) Dapatkan token bot dari [@BotFather](https://t.me/BotFather)

2) Simpan secret ke `.env`

3) Jalankan:

```bash
docker-compose up -d
```

Selanjutnya:
- **Panduan instalasi:** https://chibi.bot/installation
- **Referensi konfigurasi:** https://chibi.bot/configuration

---

## 🔑 Mendapatkan Kunci API

Setiap penyedia memerlukan kunci API sendiri. Berikut adalah tautan langsung:

**Penyedia Utama:**
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
- **OpenRouter** (akses terpadu ke banyak model): [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys)
- **ZhipuAI** (GLM, CogView): [z.ai/manage-apikey/apikey-list](https://z.ai/manage-apikey/apikey-list)

**Alat Kreatif:**
- **ElevenLabs** (Voice): [elevenlabs.io](https://elevenlabs.io/)
- **Suno** (Music, unofficial): [sunoapi.org](https://sunoapi.org/)

> 📖 **Panduan lengkap dengan instruksi pengaturan:** [chibi.bot/guides/get-api-keys](https://chibi.bot/guides/get-api-keys)

---

## Coba ini dalam 5 menit pertama

Tempelkan ini di Telegram setelah deploy.

1) **Perencanaan + eksekusi**
> Ajukan 3 pertanyaan untuk memperjelas tujuan saya, lalu usulkan rencana dan jalankan langkah 1.

2) **Kerja paralel (sub-agent)**
> Buat 3 sub-agent: satu untuk riset opsi, satu untuk menyusun rekomendasi, satu untuk mencantumkan risiko. Kembalikan satu keputusan.

3) **Mode agen (tools)**
> Periksa file proyek dan ringkas apa yang dilakukan repo ini. Lalu usulkan 5 perbaikan dan buat checklist.

4) **Tugas background**
> Mulai tugas background: kumpulkan sumber tentang X dan berikan sintesis dalam 30 menit. Beri saya update.

---

## Apa yang membuat Chibi berbeda

### 🎭 Orkestrasi multi-penyedia
Chibi dapat menjaga konteks sambil berpindah penyedia di tengah thread, atau memilih model terbaik per langkah—menyeimbangkan **biaya**, **kapabilitas**, dan **kecepatan**.

### 🤖 Kemampuan agen otonom
- **Delegasi rekursif:** sub-agent dapat membuat sub-agent mereka sendiri
- **Pemrosesan background:** tugas jangka panjang berjalan asinkron
- **Akses filesystem:** baca/tulis/cari/rapikan file
- **Eksekusi terminal:** menjalankan perintah dengan keamanan yang dimoderasi LLM
- **Memori persisten:** riwayat percakapan tetap ada setelah restart dengan manajemen konteks/ringkasan
- **Skills:** modul instruksi yang dapat digunakan kembali dan dimuat agen ke system prompt-nya sesuai kebutuhan (`load_builtin_skill`)

### 🔌 Dapat diperluas via MCP (Model Context Protocol)
Hubungkan Chibi ke tools dan layanan eksternal (atau buat sendiri):

- GitHub (PR, issue, code review)
- Otomasi browser
- Docker / layanan cloud
- Database
- Tools kreatif (Blender, Figma)

Jika sebuah tool bisa diekspos via MCP, Chibi bisa belajar menggunakannya.

### 🎨 Generasi konten kaya
- **Gambar:** Nano Banana, Imagen, Qwen, Wan, GPT Image, Grok Imagine, GLM Image
- **Musik:** Suno (termasuk custom mode: style/lyrics/vocal)
- **Suara:** transkripsi + text-to-speech (ElevenLabs, OpenAI, MiniMax)

---

## Use cases

**Developer**
```
Anda: “Jalankan test dan perbaiki yang rusak. Saya akan mengerjakan frontend.”
Chibi: *membuat sub-agent, menjalankan test, menganalisis kegagalan, mengusulkan perbaikan*
```

**Peneliti**
```
Anda: “Riset perkembangan terbaru di komputasi kuantum. Saya butuh sintesis besok.”
Chibi: *membuat beberapa agen riset, menggabungkan sumber, mengirim laporan*
```

**Kreator**
```
Anda: “Buat cityscape cyberpunk dan komposisikan track synthwave yang cocok.”
Chibi: *menghasilkan gambar, membuat musik, mengirim keduanya*
```

**Tim**
```
Anda: “Review PR ini dan perbarui dokumentasi sesuai perubahan.”
Chibi: *menganalisis perubahan, menyarankan perbaikan, memperbarui docs via MCP*
```

---

## Privasi, kontrol, dan keamanan

- **Self-hosted:** data Anda tetap di infrastruktur Anda
- **Mode publik:** pengguna bisa memakai API key mereka sendiri (tanpa master key bersama)
- **Kontrol akses:** whitelist user/grup/model
- **Opsi penyimpanan:** volume lokal, Redis, atau DynamoDB
- **Keamanan tools:** tools agen dapat dikonfigurasi; eksekusi terminal dimoderasi dan bisa dibatasi

---

### `MAX_HISTORY_TOKENS` — ambang peringkasan konteks (perubahan default yang bersifat breaking)

`MAX_HISTORY_TOKENS` adalah ambang saat Chibi otomatis meringkas percakapan agar konteks tetap terkendali. **Semantiknya berubah**: kini nilainya dibandingkan dengan **jumlah token prompt nyata yang dilaporkan provider** (seluruh permintaan keluar: system prompt + Skills aktif + skema tool + argumen pemanggilan tool + overhead struktural per pesan + isi percakapan), bukan heuristik lama yang hanya mengukur `content` + `role` percakapan. Angka nyata sekitar **4,8x lebih besar** dari estimasi lama untuk percakapan yang sama (lihat `fix_context_size/context_size_accounting_analysis.md` untuk rincian pengukuran).

- **Default dikalibrasi ulang:** `64000` → `100000`. Nilai baru melindungi jendela konteks terkecil yang umum didukung (128k token): `100000` adalah ~78% dari jendela 128k (peringkasan berjalan *sebelum* model 128k overflow) dan ~50% dari jendela 200k (menyisakan ruang lega). Nilai lama `64000` adalah estimasi riwayat saja yang tak pernah tercapai sebelum overflow nyata, karena menghitung karakter Kiril ~2x lebih rendah dan mengecualikan overhead tetap per giliran (system prompt ~3,7k, skema tool ~6,8k, Skills aktif ~5,9k, `user_info` ~0,75–3k).
- **Migrasi:** bila Anda menetapkan `MAX_HISTORY_TOKENS` secara eksplisit di `.env`, nilai lama disetel terhadap estimasi lama berbasis riwayat saja dan kini dibandingkan dengan angka yang benar, ~4,8x lebih besar untuk percakapan sama. Setel ulang ke skala **~100k** (misalnya `64000` → `100000`) agar peringkasan terjadi sebelum jendela konteks model terkecil overflow. Bila belum pernah diatur, default baru berlaku otomatis.
- Fallback cold-start (giliran pertama setelah proses dimulai ulang, ketika belum ada data provider dalam cache) tetap menggunakan heuristik lama agar peringkasan tetap berfungsi.

### `REACTIVE_CONTEXT_RECOVERY` — satu kali coba ulang setelah konteks overflow

`REACTIVE_CONTEXT_RECOVERY` (default: `true`) adalah jaring pengaman reaktif yang melengkapi ambang proaktif `MAX_HISTORY_TOKENS`. Saat provider menolak permintaan dengan error bertipe `context_length_exceeded`, Chibi otomatis meringkas riwayat dan mencoba ulang giliran tersebut **tepat satu kali**. Jika berhasil, Anda menerima jawaban normal (dengan catatan singkat bahwa konteks dikompresi). Jika kembali overflow atau pemulihan gagal karena alasan apa pun, giliran tersebut mengikuti jalur permintaan maaf yang sudah ada — siklus ringkas+coba ulang tidak akan berjalan lebih dari sekali untuk tiap giliran asli. Setel ke `false` untuk menonaktifkannya dan mempertahankan perilaku lama (log + permintaan maaf, tanpa coba ulang).

### Direktori kerja memiliki cakupan per thread (`WORKING_DIR`)

Direktori kerja agen — digunakan untuk perintah terminal dan dilaporkan ke model sebagai CWD saat ini — memiliki cakupan **per thread**, seperti model LLM yang dipilih juga terikat per thread.

- **Default:** berasal dari pengaturan `WORKING_DIR` (default `~/chibi`) melalui nilai lama tingkat pengguna; deployment baru mewarisi pengaturan ini langsung.
- **Override per thread:** thread/percakapan apa pun dapat mengubah direktori kerjanya **secara terisolasi** dengan tool agen `set_working_dir` (hanya digerakkan LLM; tidak ada slash command, tersedia saat `FILESYSTEM_ACCESS` aktif). Ini memungkinkan dua agen di thread berbeda bekerja di proyek berbeda bersamaan tanpa saling mengganggu.
- **Urutan resolusi:** override thread → direktori lama tingkat pengguna → pengaturan `WORKING_DIR`.
- **Normalisasi path:** nilai yang Anda set diperluas menjadi path absolut saat disimpan (`~/x` menjadi `/abs/x`); default yang belum diubah mempertahankan bentuk mentahnya.
- **Sub-agent** yang dibuat dalam thread berbagi direktori kerja thread tersebut — path efektif yang sama dimasukkan ke system prompt dan tool call mereka.
- Override **tetap ada saat cloning thread**: `/new_thread_with_current_context` membawa direktori kerja bersama pesan dan preferensi model.

---

## Dokumentasi

- **Mulai di sini:** https://chibi.bot
- Pengantar & filosofi: https://chibi.bot/introduction
- Instalasi: https://chibi.bot/installation
- Konfigurasi: https://chibi.bot/configuration
- Mode agen: https://chibi.bot/agent-mode
- Panduan MCP: https://chibi.bot/guides/mcp
- Dukungan / troubleshooting: https://chibi.bot/support

---

## Kebutuhan sistem

- **Minimum:** Raspberry Pi 4 / AWS EC2 t4g.nano (2 vCPU, 512MB RAM)
- **Arsitektur:** `linux/amd64`, `linux/arm64`
- **Dependensi:** Docker (dan opsional Docker Compose)

---

## Berkontribusi

- Issues: https://github.com/s-nagaev/chibi/issues
- PR: https://github.com/s-nagaev/chibi/pulls
- Diskusi: https://github.com/s-nagaev/chibi/discussions

Silakan baca [CONTRIBUTING.md](CONTRIBUTING.md) sebelum mengirim.

---

## Lisensi

MIT — lihat [LICENSE](LICENSE).

---

<p align="center">
  <strong>Siap bertemu teman digital Anda?</strong><br/>
  <a href="https://chibi.bot/start"><strong>Mulai →</strong></a>
</p>
