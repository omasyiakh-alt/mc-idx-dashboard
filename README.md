# Dashboard Simulasi Monte Carlo Saham IDX
Foreign Flow + Bandarmology → TP/SL berbasis probabilitas

## Cara pakai tanpa menjalankan file (Streamlit Community Cloud, gratis)
1. Upload seluruh isi folder ini ke repository GitHub (boleh private). File `app.py` harus ada di root repo.
2. Buka https://share.streamlit.io, login dengan GitHub, klik **Create app** lalu pilih repo, branch `main`, file `app.py`.
3. Di **Advanced settings > Secrets**, tempel (lihat `.streamlit/secrets.toml.example`):
   ```toml
   INDEXALPHA_API_KEY = "api_key_anda"
   APP_PASSWORD = "kata_sandi_pilihan_anda"   # opsional
   ```
4. Klik **Deploy**. Anda akan mendapat link `https://<nama>.streamlit.app` yang bisa dibuka dari browser atau HP.
   Biarkan app tetap private (hanya Anda) karena syarat penyedia data melarang membagikan datanya ke orang lain.

## Menjalankan di komputer sendiri (opsional)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data otomatis dari Index Alpha
Dengan `INDEXALPHA_API_KEY` terisi, foreign flow dan broker summary terambil otomatis begitu kode emiten diketik.
- **Mode Lengkap**: 1 request per hari bursa untuk foreign flow dan broker summary (≈ 2 × jendela sinyal, default 40 request per emiten). Memberi grafik net asing harian dan grafik akumulasi/distribusi harian. Cocok untuk paket berbayar.
- **Mode Hemat**: 2 request per emiten (agregat jendela sinyal). Cocok untuk paket gratis (5 request/hari).
- Hasil disimpan sementara 3 jam, jadi membuka emiten yang sama lagi tidak memotong kuota.
- Data end-of-day, diperbarui 19:00 WIB. Riwayat mulai 1 Januari 2025.
- Harga diambil dari Yahoo Finance; bila gagal, otomatis memakai OHLCV Index Alpha.

## Apa yang ditampilkan
- **Tab 1D, 1W, 1M, 3M, 6M** (1, 5, 21, 63, 126 hari bursa): fan chart jalur harga (pita 5–95% dan 25–75%, median, contoh jalur), distribusi harga akhir, VaR/CVaR 95% & 99%, peluang naik, dan saran **TP/SL** yang sudah dibulatkan ke fraksi harga IDX.
- **P(TP lebih dulu) / P(SL lebih dulu)** dihitung per jalur (mana yang tersentuh duluan), bukan hanya dari harga akhir.
- **Peta optimasi TP/SL**: expectancy tiap kombinasi TP% × SL% (horizon ≥ 1W).
- **Ringkasan**: tabel semua horizon + grafik VaR/CVaR, bisa diunduh CSV.
- **Sinyal**: grafik net foreign, top broker net buy/sell, atau indikator proxy.

## Sumber data: mana yang riil, mana yang proxy
| Data | Sumber | Status |
|---|---|---|
| Harga OHLCV | Yahoo Finance (`KODE.JK`) via yfinance, atau unggah CSV | Riil |
| Foreign Flow | **Otomatis via Index Alpha API**, atau unggah CSV dari aplikasi sekuritas/Stockbit/RTI, atau isi skor manual | Riil (API berbayar, ada paket gratis 5 request/hari). Tidak ada proxy |
| Bandarmology | **Otomatis via Index Alpha API** (broker summary), atau unggah CSV | Riil |
| Bandarmology (default) | Proxy dari OHLCV: CMF(20), A/D line, OBV + divergensi | **Proxy**, bukan data broker |

Dashboard selalu menampilkan label sumber di setiap skor dan memberi peringatan bila memakai proxy.

### Format CSV (contoh di folder `contoh_data/`, isinya data karangan untuk uji format saja)
**Foreign flow**: `date, net_foreign` (Rupiah, + = net buy asing) atau `date, foreign_buy, foreign_sell`.

**Broker summary** (satu baris per broker per hari): `date, broker, buy_lot, sell_lot, buy_value, sell_value`
(`buy_value/sell_value` opsional, dipakai untuk menghitung harga rata-rata akumulator).

**Harga** (opsional, bila Yahoo gagal): `Date, Open, High, Low, Close, Volume`.

## Model
Log-return harian `r = μ_t + σ·z`:
- `σ` = campuran volatilitas historis dan EWMA (λ = 0,94).
- `z` = bootstrap return historis (default, menangkap fat-tail riil), Student-t, atau Normal.
- Skor FF dan Bandar (−1 … +1) digabung berbobot dan menggeser drift sebesar `skor × kekuatan × σ`, memudar dengan waktu paruh (default 21 hari). Skor negatif (distribusi) juga menaikkan volatilitas.
- Tiap langkah dipotong batas ARA (35% / 25% / 20% sesuai fraksi harga) dan ARB (default 15%, bisa diubah).
- **TP/SL (default: Target R:R per timeframe)**. Rentang R:R mengikuti kaidah umum dan bisa diubah di sidebar:

  | Timeframe | Gaya | R:R |
  |---|---|---|
  | 1D | scalping | 1:1 – 1:1,5 |
  | 1W | transisi | 1:1,5 – 1:2 |
  | 1M, 3M | swing | 1:2 – 1:3 |
  | 6M | posisi | 1:2,5 – 1:3,5 |

  Kandidat SL diambil dari harga terendah intra-periode yang disentuh 5%–50% jalur. Untuk tiap SL, TP = harga masuk + R × risiko,
  dengan R di dalam rentang. Setiap pasangan diuji pada semua jalur (mana yang tersentuh lebih dulu). Lalu dipilih yang
  **edge**-nya terbesar: win rate simulasi (TP lebih dulu ÷ TP atau SL tersentuh) dikurangi win rate impas 1/(1+R).
  Bila tidak ada pasangan dengan edge positif, dashboard memilih peluang TP tertinggi dan memberi peringatan.
  Pilihan lain: expectancy tertinggi atau peluang TP tertinggi. Tabel alternatif menampilkan 10 pasangan teratas.
- Metode lama (**Peluang tersentuh**) tetap tersedia: TP = harga yang disentuh X% jalur, SL = harga yang disentuh Y% jalur.

### Skor Foreign Flow
50% intensitas (net asing ÷ nilai transaksi), 25% konsistensi (hari net buy vs net sell), 25% tren kumulatif.

### Skor Bandarmology (broker summary)
Metode Top-N: (Σ net lot top-5 pembeli − Σ net lot top-5 penjual) ÷ total lot, ditambah Top-1, rasio jumlah broker pembeli vs penjual, dan posisi harga terhadap rata-rata harga akumulator.

## Catatan penting
- Minimal 10.000 simulasi (input dikunci di ≥ 10.000; default 20.000, maks. 200.000).
- Parameter pengaruh sinyal (kekuatan, bobot, waktu paruh) adalah asumsi, bukan hasil kalibrasi. Sebaiknya diuji mundur (backtest) sebelum dipakai serius.
- Alat bantu manajemen risiko, bukan rekomendasi beli/jual.
