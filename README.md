# Resim Düzenleyici

Python ile yazılmış, masaüstünde çalışan bir resim düzenleme ve görüntü işleme uygulaması.
Arayüz **Tkinter**, görüntü işleme **Pillow**, **OpenCV** ve **NumPy** ile yapılmıştır.

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python resim_editor.py            # boş açılır
python resim_editor.py foto.jpg   # resimle açılır
```

## Özellikler

| Kategori | İşlemler |
|---|---|
| **Dönüşüm** | Sola/sağa 90°, 180°, serbest açıyla döndürme, yatay (ayna) ve dikey ters çevirme, fareyle kırpma, yeniden boyutlandırma, çerçeve ekleme |
| **Filtreler** | Ortalama, Gauss, medyan, bilateral ve hareket bulanıklığı, keskinleştirme, unsharp mask, kabartma, kenar vurgulama, gürültü giderme |
| **Renk** | Gri tonlama, negatif, sepya, otomatik kontrast, histogram eşitleme, CLAHE, eşikleme (normal, Otsu, uyarlamalı), posterize, solarize, R/G/B kanal ayırma |
| **Kenar & Morfoloji** | Canny, Sobel, Laplace, kontur, aşındırma, genişletme, açma, kapama, morfolojik gradyan, top-hat, black-hat |
| **Efektler** | Karikatür, karakalem (gri ve renkli), suluboya, detay artırma, vinyet, pikselleştirme, Gauss ve tuz-biber gürültüsü, yüz tespiti, yüz bulanıklaştırma, yazı ekleme |
| **Analiz** | RGB ve parlaklık histogramı (istatistiklerle), resim bilgisi |

### Canlı ayarlar
Sağ paneldeki kaydırıcılarla parlaklık, kontrast, doygunluk, keskinlik, renk tonu, gama ve
sıcaklık ayarlanır. Değişiklikler anında önizlenir, **Uygula** ile kalıcı hale gelir.

### Kısayollar ve kullanım

| Tuş / Fare | İşlev |
|---|---|
| Ctrl+O / Ctrl+S / Ctrl+Shift+S | Aç / Kaydet / Farklı kaydet |
| Ctrl+Z / Ctrl+Y | Geri al / İleri al (30 adım) |
| Ctrl + / Ctrl - / Ctrl+0 | Yakınlaştır / Uzaklaştır / Pencereye sığdır |
| Fare tekerleği | Yakınlaştırma |
| Sağ tık + sürükle | Resmi kaydırma |
| Esc | Kırpma / yazı modundan çıkma |
| "Orijinali Göster" (basılı tut) | Orijinal resimle karşılaştırma |

Desteklenen kayıt formatları: PNG, JPG, BMP, WEBP, TIFF.
