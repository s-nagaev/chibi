<h1 align="center"><img width=150 src="https://github.com/s-nagaev/chibi/raw/main/docs/logo.png" alt="Logo Chibi"></h1>

<p align="center">
  <strong>Teman digital Anda. Bukan alat. Mitra.</strong><br/>
  <span>Bot Telegram self-hosted dan asinkron yang mengorkestrasi banyak penyedia AI, tools, dan sub-agent untuk menyelesaikan pekerjaan nyata.</span>
</p>

<p align="center">
  <a href="https://github.com/s-nagaev/chibi/actions/workflows/build.yml"><img src="https://github.com/s-nagaev/chibi/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://www.codefactor.io/repository/github/s-nagaev/chibi"><img src="https://www.codefactor.io/repository/github/s-nagaev/chibi/badge" alt="CodeFactor"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi"><img src="https://img.shields.io/docker/pulls/pysergio/chibi" alt="Docker Pulls"></a>
  <a href="https://hub.docker.com/r/pysergio/chibi/tags"><img src="https://img.shields.io/badge/docker%20image%20arch-arm64%20%7C%20amd64-informational" alt="Arsitektur"></a>
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
- **Cloudflare Workers AI** (banyak model open-source)

### Endpoint kompatibel OpenAI (self-host / lokal)

- **Ollama**
- **vLLM**
- **LM Studio**
- **API kompatibel OpenAI apa pun**

### Penyedia multimodal (opsional)

- **Gambar:** Google (Imagen, Nano Banana), OpenAI (DALL·E), Alibaba (Qwen Image), xAI (Grok Image), Wan
- **Musik:** Suno
- **Suara:** ElevenLabs, MiniMax, OpenAI (Whisper)

> Ketersediaan model yang tepat bergantung pada API key yang Anda konfigurasi dan fitur yang diaktifkan.

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

### 🔌 Dapat diperluas via MCP (Model Context Protocol)
Hubungkan Chibi ke tools dan layanan eksternal (atau buat sendiri):

- GitHub (PR, issue, code review)
- Otomasi browser
- Docker / layanan cloud
- Database
- Tools kreatif (Blender, Figma)

Jika sebuah tool bisa diekspos via MCP, Chibi bisa belajar menggunakannya.

### 🎨 Generasi konten kaya
- **Gambar:** Nano Banana, Imagen, Qwen, Wan, DALL·E, Grok
- **Musik:** Suno (termasuk custom mode: style/lyrics/vocal)
- **Suara:** transkripsi + text-to-speech (ElevenLabs, MiniMax, OpenAI)

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
