# -*- coding: utf-8 -*-
"""
Resim Düzenleyici - Python ile Görüntü İşleme Uygulaması
=========================================================
Kullanılan kütüphaneler:
    - Tkinter : Grafik arayüz
    - Pillow  : Resim açma/kaydetme, temel işlemler
    - OpenCV  : Görüntü işleme algoritmaları
    - NumPy   : Piksel düzeyinde hesaplamalar

Çalıştırma:
    python resim_editor.py            (boş açılır)
    python resim_editor.py foto.jpg   (resimle açılır)
"""

import os
import sys
from functools import partial
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageFont


# =============================================================================
# Yardımcı dönüşümler
# =============================================================================
def pil_to_cv(img):
    """PIL (RGB) -> OpenCV (BGR) numpy dizisi."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cv_to_pil(arr):
    """OpenCV (BGR veya gri) numpy dizisi -> PIL (RGB)."""
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def to_gray(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)


def odd(n):
    """Çekirdek boyutları tek sayı olmalı."""
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


# =============================================================================
# 1) Geometrik dönüşümler
# =============================================================================
def rotate_left(img):
    return img.transpose(Image.Transpose.ROTATE_90)


def rotate_right(img):
    return img.transpose(Image.Transpose.ROTATE_270)


def rotate_180(img):
    return img.transpose(Image.Transpose.ROTATE_180)


def rotate_angle(img, angle):
    """Pozitif açı: saat yönünün tersine döndürür. Taşan köşeler beyazla doldurulur."""
    return img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True,
                      fillcolor=(255, 255, 255))


def flip_horizontal(img):
    return ImageOps.mirror(img)


def flip_vertical(img):
    return ImageOps.flip(img)


def resize_image(img, w, h):
    return img.resize((int(w), int(h)), Image.Resampling.LANCZOS)


def add_border(img, size, color):
    return ImageOps.expand(img, border=int(size), fill=color)


# =============================================================================
# 2) Filtreler (bulanıklaştırma, keskinleştirme, gürültü giderme)
# =============================================================================
def box_blur(img, k):
    k = max(1, int(k))
    return cv_to_pil(cv2.blur(pil_to_cv(img), (k, k)))


def gaussian_blur(img, k):
    k = odd(k)
    return cv_to_pil(cv2.GaussianBlur(pil_to_cv(img), (k, k), 0))


def median_blur(img, k):
    return cv_to_pil(cv2.medianBlur(pil_to_cv(img), odd(k)))


def bilateral_filter(img, d, sigma):
    return cv_to_pil(cv2.bilateralFilter(pil_to_cv(img), int(d), sigma, sigma))


def motion_blur(img, k):
    k = max(1, int(k))
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv_to_pil(cv2.filter2D(pil_to_cv(img), -1, kernel))


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)
    return cv_to_pil(cv2.filter2D(pil_to_cv(img), -1, kernel))


def unsharp_mask(img, radius, percent):
    return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(percent), threshold=3))


def emboss(img):
    return img.filter(ImageFilter.EMBOSS)


def edge_enhance(img):
    return img.filter(ImageFilter.EDGE_ENHANCE_MORE)


def denoise(img, h):
    return cv_to_pil(cv2.fastNlMeansDenoisingColored(pil_to_cv(img), None, h, h, 7, 21))


# =============================================================================
# 3) Renk işlemleri
# =============================================================================
def grayscale(img):
    return ImageOps.grayscale(img).convert("RGB")


def negative(img):
    return ImageOps.invert(img)


def sepia(img):
    arr = np.asarray(img, dtype=np.float32)
    m = np.array([[0.393, 0.769, 0.189],
                  [0.349, 0.686, 0.168],
                  [0.272, 0.534, 0.131]], np.float32)
    out = np.clip(arr @ m.T, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def threshold(img, t):
    _, out = cv2.threshold(to_gray(img), int(t), 255, cv2.THRESH_BINARY)
    return cv_to_pil(out)


def otsu_threshold(img):
    _, out = cv2.threshold(to_gray(img), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv_to_pil(out)


def adaptive_threshold(img, block, c):
    block = max(3, odd(block))
    out = cv2.adaptiveThreshold(to_gray(img), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, block, c)
    return cv_to_pil(out)


def equalize_histogram(img):
    """Parlaklık (Y) kanalında histogram eşitleme - renkler korunur."""
    ycrcb = cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv_to_pil(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR))


def clahe(img, clip):
    """Kontrast sınırlı uyarlamalı histogram eşitleme (LAB uzayında L kanalı)."""
    lab = cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv_to_pil(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))


def auto_contrast(img):
    return ImageOps.autocontrast(img, cutoff=1)


def posterize(img, bits):
    return ImageOps.posterize(img, int(bits))


def solarize(img, t):
    return ImageOps.solarize(img, int(t))


def isolate_channel(img, idx):
    """Sadece tek bir renk kanalını bırakır (0=R, 1=G, 2=B)."""
    arr = np.array(img)
    for i in range(3):
        if i != idx:
            arr[..., i] = 0
    return Image.fromarray(arr)


def swap_rb(img):
    return Image.fromarray(np.array(img)[..., ::-1].copy())


# =============================================================================
# 4) Kenar bulma ve morfolojik işlemler
# =============================================================================
def canny(img, low, high):
    gray = cv2.GaussianBlur(to_gray(img), (3, 3), 0)
    return cv_to_pil(cv2.Canny(gray, int(low), int(high)))


def sobel(img):
    gray = to_gray(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv_to_pil(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))


def laplacian(img):
    gray = cv2.GaussianBlur(to_gray(img), (3, 3), 0)
    return cv_to_pil(cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_64F, ksize=3)))


def contour(img):
    return img.filter(ImageFilter.CONTOUR)


MORPH_OPS = {
    "erode": None,
    "dilate": None,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
    "tophat": cv2.MORPH_TOPHAT,
    "blackhat": cv2.MORPH_BLACKHAT,
}


def morphology(img, k, iterations, op):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(k), int(k)))
    bgr = pil_to_cv(img)
    it = int(iterations)
    if op == "erode":
        out = cv2.erode(bgr, kernel, iterations=it)
    elif op == "dilate":
        out = cv2.dilate(bgr, kernel, iterations=it)
    else:
        out = cv2.morphologyEx(bgr, MORPH_OPS[op], kernel, iterations=it)
    return cv_to_pil(out)


# =============================================================================
# 5) Efektler
# =============================================================================
def cartoon(img):
    bgr = pil_to_cv(img)
    gray = cv2.medianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(bgr, 9, 250, 250)
    return cv_to_pil(cv2.bitwise_and(color, color, mask=edges))


def pencil_sketch(img, colored=False):
    gray, color = cv2.pencilSketch(pil_to_cv(img), sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return cv_to_pil(color if colored else gray)


def stylization(img):
    return cv_to_pil(cv2.stylization(pil_to_cv(img), sigma_s=60, sigma_r=0.45))


def detail_enhance(img):
    return cv_to_pil(cv2.detailEnhance(pil_to_cv(img), sigma_s=10, sigma_r=0.15))


def vignette(img, strength):
    """Kenarları karartan Gauss maskesi. strength: 1-100."""
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape[:2]
    factor = 1.0 - 0.0075 * float(strength)          # 0.9925 .. 0.25
    kx = cv2.getGaussianKernel(w, w * factor)
    ky = cv2.getGaussianKernel(h, h * factor)
    mask = ky @ kx.T
    mask = mask / mask.max()
    return Image.fromarray(np.clip(arr * mask[..., None], 0, 255).astype(np.uint8))


def pixelate(img, block):
    block = max(1, int(block))
    w, h = img.size
    small = img.resize((max(1, w // block), max(1, h // block)), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.NEAREST)


def gaussian_noise(img, sigma):
    arr = np.asarray(img, dtype=np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def salt_pepper(img, percent):
    arr = np.array(img)
    h, w = arr.shape[:2]
    n = int(h * w * percent / 100.0 / 2)
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    arr[ys, xs] = 255
    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    arr[ys, xs] = 0
    return Image.fromarray(arr)


_face_cascade = None


def detect_faces(img, blur=False):
    """Haar Cascade ile yüz bulma. (yeni_resim, yüz_sayısı) döndürür."""
    global _face_cascade
    if _face_cascade is None:
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _face_cascade = cv2.CascadeClassifier(path)
        if _face_cascade.empty():
            raise RuntimeError("Yüz tanıma modeli yüklenemedi:\n" + path)
    bgr = pil_to_cv(img)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    thickness = max(2, min(img.size) // 200)
    for (x, y, w, h) in faces:
        if blur:
            roi = bgr[y:y + h, x:x + w]
            k = odd(max(w, h) // 3)
            bgr[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (k, k), 0)
        else:
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), thickness)
    return cv_to_pil(bgr), len(faces)


def load_font(size):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def add_text(img, pos, text, size, color):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = load_font(int(size))
    outline = max(1, int(size) // 20)
    draw.text(pos, text, font=font, fill=color,
              stroke_width=outline, stroke_fill=(0, 0, 0) if sum(color) > 380 else (255, 255, 255))
    return out


# =============================================================================
# 6) Canlı ayarlar (sağ panel kaydırıcıları)
# =============================================================================
ADJ_DEFAULTS = {
    "brightness": 100, "contrast": 100, "saturation": 100, "sharpness": 100,
    "hue": 0, "gamma": 1.0, "temperature": 0,
}


def adjust_image(img, brightness=100, contrast=100, saturation=100, sharpness=100,
                 hue=0, gamma=1.0, temperature=0):
    if brightness != 100:
        img = ImageEnhance.Brightness(img).enhance(brightness / 100.0)
    if contrast != 100:
        img = ImageEnhance.Contrast(img).enhance(contrast / 100.0)
    if saturation != 100:
        img = ImageEnhance.Color(img).enhance(saturation / 100.0)
    if sharpness != 100:
        img = ImageEnhance.Sharpness(img).enhance(sharpness / 100.0)
    if hue:
        hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
        hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(hue) // 2) % 180).astype(np.uint8)
        img = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    if abs(gamma - 1.0) > 1e-6:
        lut = [min(255, int(255 * (i / 255.0) ** (1.0 / gamma) + 0.5)) for i in range(256)]
        img = img.point(lut * 3)
    if temperature:
        t = int(temperature)
        r_lut = [max(0, min(255, i + t)) for i in range(256)]
        b_lut = [max(0, min(255, i - t)) for i in range(256)]
        img = img.point(r_lut + list(range(256)) + b_lut)
    return img


# =============================================================================
# Diyalog pencereleri
# =============================================================================
class ParamDialog(simpledialog.Dialog):
    """Sayısal parametre isteyen genel diyalog.
    fields: [(etiket, varsayılan, min, max, adım), ...]  - varsayılan int ise sonuç int olur."""

    def __init__(self, parent, title, fields):
        self.fields = fields
        self.result = None
        super().__init__(parent, title)

    def body(self, master):
        self.vars = []
        first = None
        for i, (label, default, lo, hi, step) in enumerate(self.fields):
            ttk.Label(master, text=label).grid(row=i, column=0, sticky="w", padx=6, pady=4)
            var = tk.StringVar(value=str(default))
            sb = ttk.Spinbox(master, from_=lo, to=hi, increment=step, textvariable=var, width=10)
            sb.grid(row=i, column=1, padx=6, pady=4)
            ttk.Label(master, text=f"({lo} – {hi})", foreground="gray").grid(row=i, column=2, sticky="w")
            self.vars.append((var, isinstance(default, int), lo, hi, label))
            first = first or sb
        return first

    def validate(self):
        values = []
        for var, is_int, lo, hi, label in self.vars:
            try:
                v = float(var.get().replace(",", "."))
            except ValueError:
                messagebox.showwarning("Hatalı değer", f"'{label}' bir sayı olmalı.", parent=self)
                return False
            if not lo <= v <= hi:
                messagebox.showwarning("Hatalı değer", f"'{label}' {lo} ile {hi} arasında olmalı.", parent=self)
                return False
            values.append(int(round(v)) if is_int else v)
        self.result = values
        return True


class ResizeDialog(simpledialog.Dialog):
    def __init__(self, parent, size):
        self.orig_w, self.orig_h = size
        self.result = None
        self._lock = False
        super().__init__(parent, "Yeniden Boyutlandır")

    def body(self, master):
        self.w_var = tk.StringVar(value=str(self.orig_w))
        self.h_var = tk.StringVar(value=str(self.orig_h))
        self.p_var = tk.StringVar(value="100")
        self.keep = tk.BooleanVar(value=True)

        ttk.Label(master, text=f"Mevcut boyut: {self.orig_w} × {self.orig_h} px").grid(
            row=0, column=0, columnspan=2, pady=(0, 8))
        ttk.Label(master, text="Genişlik (px):").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        e_w = ttk.Entry(master, textvariable=self.w_var, width=10)
        e_w.grid(row=1, column=1, pady=3)
        ttk.Label(master, text="Yükseklik (px):").grid(row=2, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(master, textvariable=self.h_var, width=10).grid(row=2, column=1, pady=3)
        ttk.Label(master, text="Yüzde (%):").grid(row=3, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(master, textvariable=self.p_var, width=10).grid(row=3, column=1, pady=3)
        ttk.Checkbutton(master, text="En-boy oranını koru", variable=self.keep).grid(
            row=4, column=0, columnspan=2, pady=(6, 0))

        self.w_var.trace_add("write", lambda *a: self._sync("w"))
        self.h_var.trace_add("write", lambda *a: self._sync("h"))
        self.p_var.trace_add("write", lambda *a: self._sync("p"))
        return e_w

    def _sync(self, src):
        if self._lock:
            return
        self._lock = True
        try:
            if src == "p":
                p = float(self.p_var.get()) / 100.0
                self.w_var.set(str(max(1, round(self.orig_w * p))))
                self.h_var.set(str(max(1, round(self.orig_h * p))))
            elif self.keep.get():
                if src == "w":
                    w = int(self.w_var.get())
                    self.h_var.set(str(max(1, round(w * self.orig_h / self.orig_w))))
                else:
                    h = int(self.h_var.get())
                    self.w_var.set(str(max(1, round(h * self.orig_w / self.orig_h))))
        except (ValueError, ZeroDivisionError):
            pass
        finally:
            self._lock = False

    def validate(self):
        try:
            w, h = int(self.w_var.get()), int(self.h_var.get())
        except ValueError:
            messagebox.showwarning("Hatalı değer", "Genişlik ve yükseklik tam sayı olmalı.", parent=self)
            return False
        if not (1 <= w <= 20000 and 1 <= h <= 20000):
            messagebox.showwarning("Hatalı değer", "Boyutlar 1 ile 20000 arasında olmalı.", parent=self)
            return False
        self.result = (w, h)
        return True


# =============================================================================
# Ana uygulama
# =============================================================================
class ImageEditorApp(tk.Tk):
    HISTORY_LIMIT = 30
    FILETYPES_OPEN = [("Resim dosyaları", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"),
                      ("Tüm dosyalar", "*.*")]
    FILETYPES_SAVE = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"),
                      ("WEBP", "*.webp"), ("TIFF", "*.tif")]

    def __init__(self, path=None):
        super().__init__()
        self.title("Resim Düzenleyici")
        self.geometry("1350x820")
        self.minsize(1000, 600)

        # Durum
        self.original = None        # İlk açılan resim
        self.image = None           # Üzerinde çalışılan resim
        self.path = None
        self.history, self.redo_stack = [], []
        self.dirty = False
        self.zoom, self.fit_mode = 1.0, True
        self.offset = (0, 0)
        self.showing_original = False
        self.mode = None            # None | "crop" | "text"
        self.crop_start, self.crop_rect = None, None
        self._cache = None
        self._render_job = None

        self._build_style()
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._build_menu()
        self._bind_keys()

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self.render)
        if path:
            self.after(100, lambda: self.load_image(path))

    # ------------------------------------------------------------------ arayüz
    def _build_style(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Tool.TButton", anchor="w", padding=(8, 4))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _sections(self):
        """Sol paneldeki ve menüdeki tüm araçlar: (bölüm, [(etiket, komut), ...])"""
        op = self.op
        return [
            ("Dönüşüm", [
                ("⟲  Sola Döndür (90°)", op(rotate_left, "Sola döndürme")),
                ("⟳  Sağa Döndür (90°)", op(rotate_right, "Sağa döndürme")),
                ("↻  180° Döndür", op(rotate_180, "180° döndürme")),
                ("∠  Serbest Açıyla Döndür…", op(rotate_angle, "Serbest döndürme",
                                                  [("Açı (° , + saat yönü tersi)", 15.0, -360, 360, 1)])),
                ("⇆  Yatay Ters Çevir (Ayna)", op(flip_horizontal, "Yatay ters çevirme")),
                ("⇅  Dikey Ters Çevir", op(flip_vertical, "Dikey ters çevirme")),
                ("✂  Kırp (fare ile seç)", self.toggle_crop),
                ("⤢  Yeniden Boyutlandır…", self.do_resize),
                ("▭  Çerçeve Ekle…", self.do_border),
            ]),
            ("Filtreler", [
                ("Ortalama Bulanıklık…", op(box_blur, "Ortalama bulanıklık",
                                            [("Çekirdek boyutu", 5, 1, 99, 1)])),
                ("Gauss Bulanıklığı…", op(gaussian_blur, "Gauss bulanıklığı",
                                          [("Çekirdek boyutu (tek)", 7, 1, 99, 2)])),
                ("Medyan Filtre…", op(median_blur, "Medyan filtre",
                                      [("Çekirdek boyutu (tek)", 5, 1, 99, 2)])),
                ("Bilateral Filtre…", op(bilateral_filter, "Bilateral filtre",
                                         [("Komşuluk çapı", 9, 1, 30, 1), ("Sigma", 75.0, 1, 250, 5)])),
                ("Hareket Bulanıklığı…", op(motion_blur, "Hareket bulanıklığı",
                                            [("Uzunluk (px)", 15, 1, 99, 1)])),
                ("Keskinleştir", op(sharpen, "Keskinleştirme")),
                ("Keskinlik Maskesi (Unsharp)…", op(unsharp_mask, "Unsharp mask",
                                                    [("Yarıçap", 2.0, 0.1, 20, 0.5),
                                                     ("Güç (%)", 150, 1, 500, 10)])),
                ("Kabartma (Emboss)", op(emboss, "Kabartma")),
                ("Kenar Vurgula", op(edge_enhance, "Kenar vurgulama")),
                ("Gürültü Gider…", op(denoise, "Gürültü giderme",
                                      [("Filtre gücü", 10.0, 1, 50, 1)])),
            ]),
            ("Renk", [
                ("Gri Tonlama", op(grayscale, "Gri tonlama")),
                ("Negatif", op(negative, "Negatif")),
                ("Sepya", op(sepia, "Sepya")),
                ("Otomatik Kontrast", op(auto_contrast, "Otomatik kontrast")),
                ("Histogram Eşitleme", op(equalize_histogram, "Histogram eşitleme")),
                ("CLAHE…", op(clahe, "CLAHE", [("Kırpma limiti", 2.0, 0.5, 20, 0.5)])),
                ("Eşikleme (Siyah-Beyaz)…", op(threshold, "Eşikleme",
                                               [("Eşik değeri", 127, 0, 255, 1)])),
                ("Otsu Eşikleme", op(otsu_threshold, "Otsu eşikleme")),
                ("Uyarlamalı Eşikleme…", op(adaptive_threshold, "Uyarlamalı eşikleme",
                                            [("Blok boyutu (tek)", 11, 3, 99, 2),
                                             ("Sabit C", 2, -50, 50, 1)])),
                ("Posterize…", op(posterize, "Posterize", [("Bit sayısı", 3, 1, 8, 1)])),
                ("Solarize…", op(solarize, "Solarize", [("Eşik", 128, 0, 255, 1)])),
                ("Sadece Kırmızı Kanal", op(partial(isolate_channel, idx=0), "Kırmızı kanal")),
                ("Sadece Yeşil Kanal", op(partial(isolate_channel, idx=1), "Yeşil kanal")),
                ("Sadece Mavi Kanal", op(partial(isolate_channel, idx=2), "Mavi kanal")),
                ("Kırmızı ↔ Mavi Değiştir", op(swap_rb, "Kanal değiştirme")),
            ]),
            ("Kenar & Morfoloji", [
                ("Canny Kenar Bulma…", op(canny, "Canny", [("Alt eşik", 50, 0, 500, 5),
                                                           ("Üst eşik", 150, 0, 500, 5)])),
                ("Sobel Kenar Bulma", op(sobel, "Sobel")),
                ("Laplace Kenar Bulma", op(laplacian, "Laplace")),
                ("Kontur Çizimi", op(contour, "Kontur")),
            ] + [
                (label, op(partial(morphology, op=key), name,
                           [("Çekirdek boyutu", 3, 1, 51, 1), ("Tekrar sayısı", 1, 1, 20, 1)]))
                for label, name, key in [
                    ("Aşındırma (Erode)…", "Aşındırma", "erode"),
                    ("Genişletme (Dilate)…", "Genişletme", "dilate"),
                    ("Açma (Opening)…", "Açma", "open"),
                    ("Kapama (Closing)…", "Kapama", "close"),
                    ("Morfolojik Gradyan…", "Morfolojik gradyan", "gradient"),
                    ("Top-Hat…", "Top-hat", "tophat"),
                    ("Black-Hat…", "Black-hat", "blackhat"),
                ]
            ]),
            ("Efektler", [
                ("Karikatür", op(cartoon, "Karikatür")),
                ("Karakalem (Gri)", op(pencil_sketch, "Karakalem")),
                ("Karakalem (Renkli)", op(partial(pencil_sketch, colored=True), "Renkli karakalem")),
                ("Suluboya / Stilize", op(stylization, "Stilize")),
                ("Detay Artırma", op(detail_enhance, "Detay artırma")),
                ("Vinyet…", op(vignette, "Vinyet", [("Yoğunluk", 50, 1, 100, 5)])),
                ("Pikselleştir (Mozaik)…", op(pixelate, "Pikselleştirme",
                                              [("Blok boyutu (px)", 12, 2, 200, 1)])),
                ("Gauss Gürültüsü Ekle…", op(gaussian_noise, "Gauss gürültüsü",
                                             [("Sigma", 25.0, 1, 100, 1)])),
                ("Tuz-Biber Gürültüsü…", op(salt_pepper, "Tuz-biber gürültüsü",
                                            [("Oran (%)", 5.0, 0.1, 50, 0.5)])),
                ("🙂  Yüz Tespiti", lambda: self.do_faces(False)),
                ("🙂  Yüzleri Bulanıklaştır", lambda: self.do_faces(True)),
                ("T  Yazı Ekle (tıkla)", self.toggle_text),
            ]),
            ("Analiz", [
                ("📊  Histogram", self.show_histogram),
                ("ℹ  Resim Bilgisi", self.show_info),
            ]),
        ]

    def _build_toolbar(self):
        bar = ttk.Frame(self, padding=(6, 4))
        bar.pack(side="top", fill="x")

        def btn(text, cmd, tip_side="left"):
            b = ttk.Button(bar, text=text, command=cmd)
            b.pack(side=tip_side, padx=2)
            return b

        btn("📂 Aç", self.open_image)
        btn("💾 Kaydet", self.save_image)
        btn("💾 Farklı Kaydet", self.save_image_as)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        btn("↶ Geri Al", self.undo)
        btn("↷ İleri Al", self.redo)
        btn("⟲ Orijinale Dön", self.reset_to_original)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        btn("⟲ 90°", self.op(rotate_left, "Sola döndürme"))
        btn("⟳ 90°", self.op(rotate_right, "Sağa döndürme"))
        btn("⇆ Ayna", self.op(flip_horizontal, "Yatay ters çevirme"))
        btn("⇅ Ters", self.op(flip_vertical, "Dikey ters çevirme"))
        btn("✂ Kırp", self.toggle_crop)
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=6)
        btn("−", lambda: self.zoom_by(1 / 1.25))
        btn("+", lambda: self.zoom_by(1.25))
        btn("Sığdır", self.zoom_fit)
        btn("%100", self.zoom_actual)

        cmp_btn = ttk.Button(bar, text="👁 Orijinali Göster (basılı tut)")
        cmp_btn.pack(side="right", padx=2)
        cmp_btn.bind("<ButtonPress-1>", lambda e: self.set_show_original(True))
        cmp_btn.bind("<ButtonRelease-1>", lambda e: self.set_show_original(False))

    def _build_body(self):
        body = ttk.Frame(self)
        body.pack(side="top", fill="both", expand=True)

        # --- Sol panel: kaydırılabilir araç listesi
        left = ttk.Frame(body, width=250)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        self.tool_canvas = tk.Canvas(left, highlightthickness=0, width=230)
        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tool_canvas.yview)
        self.tool_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.tool_canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(self.tool_canvas, padding=6)
        win_id = self.tool_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: self.tool_canvas.configure(
            scrollregion=self.tool_canvas.bbox("all")))
        self.tool_canvas.bind("<Configure>", lambda e: self.tool_canvas.itemconfigure(win_id, width=e.width))
        left.bind("<Enter>", lambda e: self.bind_all("<MouseWheel>", self._scroll_tools))
        left.bind("<Leave>", lambda e: self.unbind_all("<MouseWheel>"))

        for title, items in self._sections():
            lf = ttk.LabelFrame(inner, text=title, padding=4, style="Section.TLabelframe")
            lf.pack(fill="x", pady=(0, 8))
            for label, cmd in items:
                ttk.Button(lf, text=label, command=cmd, style="Tool.TButton").pack(fill="x", pady=1)

        # --- Sağ panel: canlı ayarlar
        right = ttk.Frame(body, padding=8, width=270)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        ttk.Label(right, text="Ayarlar (canlı önizleme)", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(right, text="Kaydırıcılar önizlemeyi değiştirir.\n'Uygula' ile kalıcı hale gelir.",
                  foreground="gray").pack(anchor="w", pady=(0, 8))

        self.adj_vars = {}
        sliders = [
            ("brightness", "Parlaklık (%)", 0, 200, 1),
            ("contrast", "Kontrast (%)", 0, 200, 1),
            ("saturation", "Doygunluk (%)", 0, 200, 1),
            ("sharpness", "Keskinlik (%)", 0, 300, 1),
            ("hue", "Renk Tonu (°)", -180, 180, 1),
            ("gamma", "Gama", 0.1, 3.0, 0.05),
            ("temperature", "Sıcaklık (soğuk ↔ sıcak)", -100, 100, 1),
        ]
        for key, label, lo, hi, res in sliders:
            var = tk.DoubleVar(value=ADJ_DEFAULTS[key])
            self.adj_vars[key] = var
            ttk.Label(right, text=label).pack(anchor="w")
            tk.Scale(right, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var,
                     length=240, command=lambda v: self.schedule_render(40)).pack(anchor="w", pady=(0, 6))

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=8)
        ttk.Button(btns, text="✔ Uygula", command=self.apply_adjustments).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="✖ Sıfırla", command=self.reset_sliders).pack(side="left", expand=True, fill="x", padx=2)

        ttk.Separator(right).pack(fill="x", pady=10)
        ttk.Label(right, text="İpuçları", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(right, foreground="gray", justify="left", wraplength=250, text=(
            "• Fare tekerleği: yakınlaştır / uzaklaştır\n"
            "• Sağ tık + sürükle: resmi kaydır\n"
            "• Kırp: araç seçip fareyle alan çizin\n"
            "• Esc: kırp / yazı modundan çık\n"
            "• Ctrl+O aç, Ctrl+S kaydet\n"
            "• Ctrl+Z geri al, Ctrl+Y ileri al")).pack(anchor="w")

        # --- Orta: resim alanı
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(center, bg="#2b2b2b", highlightthickness=0, cursor="arrow")
        hs = ttk.Scrollbar(center, orient="horizontal", command=self.canvas.xview)
        vs = ttk.Scrollbar(center, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=hs.set, yscrollcommand=vs.set)
        vs.pack(side="right", fill="y")
        hs.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Configure>", lambda e: self.schedule_render(30))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

    def _build_statusbar(self):
        bar = ttk.Frame(self, relief="sunken", padding=(6, 2))
        bar.pack(side="bottom", fill="x")
        self.status_msg = ttk.Label(bar, text="Hoş geldiniz! Bir resim açmak için Ctrl+O'ya basın.")
        self.status_msg.pack(side="left")
        self.status_info = ttk.Label(bar, text="")
        self.status_info.pack(side="right")
        self.status_pixel = ttk.Label(bar, text="", width=34)
        self.status_pixel.pack(side="right", padx=12)

    def _build_menu(self):
        menubar = tk.Menu(self)
        m_file = tk.Menu(menubar, tearoff=False)
        m_file.add_command(label="Aç…", accelerator="Ctrl+O", command=self.open_image)
        m_file.add_command(label="Kaydet", accelerator="Ctrl+S", command=self.save_image)
        m_file.add_command(label="Farklı Kaydet…", accelerator="Ctrl+Shift+S", command=self.save_image_as)
        m_file.add_separator()
        m_file.add_command(label="Çıkış", command=self.on_close)
        menubar.add_cascade(label="Dosya", menu=m_file)

        m_edit = tk.Menu(menubar, tearoff=False)
        m_edit.add_command(label="Geri Al", accelerator="Ctrl+Z", command=self.undo)
        m_edit.add_command(label="İleri Al", accelerator="Ctrl+Y", command=self.redo)
        m_edit.add_command(label="Orijinale Dön", command=self.reset_to_original)
        menubar.add_cascade(label="Düzen", menu=m_edit)

        m_view = tk.Menu(menubar, tearoff=False)
        m_view.add_command(label="Yakınlaştır", accelerator="Ctrl++", command=lambda: self.zoom_by(1.25))
        m_view.add_command(label="Uzaklaştır", accelerator="Ctrl+-", command=lambda: self.zoom_by(1 / 1.25))
        m_view.add_command(label="Pencereye Sığdır", accelerator="Ctrl+0", command=self.zoom_fit)
        m_view.add_command(label="Gerçek Boyut (%100)", command=self.zoom_actual)
        menubar.add_cascade(label="Görünüm", menu=m_view)

        for title, items in self._sections():
            m = tk.Menu(menubar, tearoff=False)
            for label, cmd in items:
                m.add_command(label=label.strip(), command=cmd)
            menubar.add_cascade(label=title, menu=m)
        self.config(menu=menubar)

    def _bind_keys(self):
        self.bind("<Control-o>", lambda e: self.open_image())
        self.bind("<Control-s>", lambda e: self.save_image())
        self.bind("<Control-S>", lambda e: self.save_image_as())
        self.bind("<Control-z>", lambda e: self.undo())
        self.bind("<Control-y>", lambda e: self.redo())
        self.bind("<Control-plus>", lambda e: self.zoom_by(1.25))
        self.bind("<Control-equal>", lambda e: self.zoom_by(1.25))
        self.bind("<Control-minus>", lambda e: self.zoom_by(1 / 1.25))
        self.bind("<Control-0>", lambda e: self.zoom_fit())
        self.bind("<Escape>", lambda e: self.set_mode(None))

    def _scroll_tools(self, event):
        self.tool_canvas.yview_scroll(int(-event.delta / 120), "units")

    # --------------------------------------------------------------- yardımcı
    def set_status(self, text):
        self.status_msg.config(text=text)

    def require_image(self):
        if self.image is None:
            messagebox.showinfo("Resim yok", "Önce bir resim açın (Dosya → Aç).", parent=self)
            return False
        return True

    def ask(self, title, fields):
        return ParamDialog(self, title, fields).result

    def op(self, func, name, fields=None):
        """Bir görüntü işleme fonksiyonunu (gerekirse parametre sorarak) çalıştıran komut üretir."""
        def run():
            if not self.require_image():
                return
            args = []
            if fields:
                args = self.ask(name, fields)
                if args is None:
                    return
            self.apply(lambda im: func(im, *args), name)
        return run

    def apply(self, func, name):
        """func(resim) -> yeni resim (veya None = değişiklik yok). Geri alma geçmişine ekler."""
        if not self.require_image():
            return False
        self.commit_adjustments()
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            new = func(self.image)
        except Exception as e:  # noqa: BLE001 - kullanıcıya hata göster
            messagebox.showerror("Hata", f"'{name}' uygulanamadı:\n{e}", parent=self)
            return False
        finally:
            self.config(cursor="")
        if new is None:
            return False
        self.push_history()
        self.image = new if new.mode == "RGB" else new.convert("RGB")
        self.dirty = True
        self.set_status(f"✔ {name} uygulandı")
        self.render()
        return True

    # ---------------------------------------------------------- geçmiş (undo)
    def push_history(self):
        self.history.append(self.image)
        if len(self.history) > self.HISTORY_LIMIT:
            self.history.pop(0)
        self.redo_stack.clear()

    def undo(self):
        if self.adjust_pending():
            self.reset_sliders()
            return
        if not self.history:
            self.set_status("Geri alınacak işlem yok")
            return
        self.redo_stack.append(self.image)
        self.image = self.history.pop()
        self.dirty = True
        self.set_status("↶ Geri alındı")
        self.render()

    def redo(self):
        if not self.redo_stack:
            self.set_status("İleri alınacak işlem yok")
            return
        self.history.append(self.image)
        self.image = self.redo_stack.pop()
        self.dirty = True
        self.set_status("↷ İleri alındı")
        self.render()

    def reset_to_original(self):
        if not self.require_image():
            return
        self.reset_sliders(render=False)
        self.push_history()
        self.image = self.original.copy()
        self.dirty = True
        self.set_status("Orijinal resme dönüldü (Geri Al ile geri getirebilirsiniz)")
        self.render()

    # ------------------------------------------------------------ canlı ayar
    def adjust_params(self):
        return {k: v.get() for k, v in self.adj_vars.items()}

    def adjust_pending(self):
        return any(abs(v.get() - ADJ_DEFAULTS[k]) > 1e-9 for k, v in self.adj_vars.items())

    def reset_sliders(self, render=True):
        for k, v in self.adj_vars.items():
            v.set(ADJ_DEFAULTS[k])
        if render:
            self.render()

    def commit_adjustments(self):
        """Bekleyen kaydırıcı ayarlarını resme kalıcı olarak uygular."""
        if self.image is None or not self.adjust_pending():
            return False
        self.push_history()
        self.image = adjust_image(self.image, **self.adjust_params())
        self.dirty = True
        self.reset_sliders(render=False)
        return True

    def apply_adjustments(self):
        if not self.require_image():
            return
        if self.commit_adjustments():
            self.set_status("✔ Ayarlar uygulandı")
            self.render()
        else:
            self.set_status("Uygulanacak ayar yok (kaydırıcılar varsayılanda)")

    # ------------------------------------------------------------ dosya
    def confirm_discard(self):
        if not self.dirty:
            return True
        ans = messagebox.askyesnocancel("Kaydedilmemiş değişiklikler",
                                        "Değişiklikler kaydedilsin mi?", parent=self)
        if ans is None:
            return False
        if ans:
            return self.save_image()
        return True

    def open_image(self):
        if not self.confirm_discard():
            return
        path = filedialog.askopenfilename(parent=self, title="Resim Aç", filetypes=self.FILETYPES_OPEN)
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                img = img.convert("RGBA")
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1])
                img = bg
            else:
                img = img.convert("RGB")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Açılamadı", f"Resim açılamadı:\n{e}", parent=self)
            return
        self.original, self.image, self.path = img, img.copy(), path
        self.history.clear()
        self.redo_stack.clear()
        self.dirty = False
        self.fit_mode = True
        self.set_mode(None)
        self.reset_sliders(render=False)
        self.title(f"Resim Düzenleyici — {os.path.basename(path)}")
        self.set_status(f"Açıldı: {path}")
        self.render()

    def save_image(self):
        if not self.require_image():
            return False
        if not self.path:
            return self.save_image_as()
        return self._save_to(self.path)

    def save_image_as(self):
        if not self.require_image():
            return False
        initial = os.path.splitext(os.path.basename(self.path))[0] + "_duzenlendi.png" if self.path else "resim.png"
        path = filedialog.asksaveasfilename(parent=self, title="Farklı Kaydet", initialfile=initial,
                                            defaultextension=".png", filetypes=self.FILETYPES_SAVE)
        if not path:
            return False
        return self._save_to(path)

    def _save_to(self, path):
        self.commit_adjustments()
        ext = os.path.splitext(path)[1].lower()
        kwargs = {"quality": 95} if ext in (".jpg", ".jpeg", ".webp") else {}
        try:
            self.image.save(path, **kwargs)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Kaydedilemedi", f"Dosya kaydedilemedi:\n{e}", parent=self)
            return False
        self.path = path
        self.dirty = False
        self.title(f"Resim Düzenleyici — {os.path.basename(path)}")
        self.set_status(f"💾 Kaydedildi: {path}")
        self.render()
        return True

    def on_close(self):
        if self.confirm_discard():
            self.destroy()

    # ------------------------------------------------------------ görüntüleme
    def schedule_render(self, delay=30):
        if self._render_job:
            self.after_cancel(self._render_job)
        self._render_job = self.after(delay, self.render)

    def render(self):
        self._render_job = None
        c = self.canvas
        c.delete("all")
        cw, ch = max(c.winfo_width(), 50), max(c.winfo_height(), 50)

        if self.image is None:
            c.create_text(cw // 2, ch // 2, fill="#bbbbbb", font=("Segoe UI", 16),
                          text="Resim açmak için buraya tıklayın\nveya Ctrl+O tuşlarına basın",
                          justify="center")
            c.configure(scrollregion=(0, 0, cw, ch))
            self.status_info.config(text="")
            return

        src = self.original if self.showing_original else self.image
        iw, ih = src.size
        if self.fit_mode:
            self.zoom = min(cw / iw, ch / ih)
        dw, dh = max(1, round(iw * self.zoom)), max(1, round(ih * self.zoom))

        if self._cache and self._cache[0] is src and self._cache[1] == (dw, dh):
            disp = self._cache[2]
        else:
            resample = Image.Resampling.BILINEAR if self.zoom < 1 else Image.Resampling.NEAREST
            disp = src.resize((dw, dh), resample, reducing_gap=2.0 if self.zoom < 0.5 else None)
            self._cache = (src, (dw, dh), disp)

        if not self.showing_original and self.adjust_pending():
            disp = adjust_image(disp, **self.adjust_params())

        self.tk_img = ImageTk.PhotoImage(disp)
        ox, oy = max(0, (cw - dw) // 2), max(0, (ch - dh) // 2)
        self.offset = (ox, oy)
        c.create_image(ox, oy, anchor="nw", image=self.tk_img)
        c.configure(scrollregion=(0, 0, max(cw, dw), max(ch, dh)))
        if self.showing_original:
            c.create_text(ox + 10, oy + 10, anchor="nw", text="ORİJİNAL", fill="yellow",
                          font=("Segoe UI", 14, "bold"))

        w, h = self.image.size
        info = f"Boyut: {w} × {h} px   |   Yakınlaştırma: %{self.zoom * 100:.0f}"
        if self.adjust_pending():
            info += "   |   ⚠ Uygulanmamış ayar var"
        self.status_info.config(text=info)

    def set_show_original(self, value):
        if self.original is None:
            return
        self.showing_original = value
        self.render()

    def zoom_by(self, factor):
        if self.image is None:
            return
        iw, ih = self.image.size
        max_zoom = min(16.0, 12000 / max(iw, ih))
        self.fit_mode = False
        self.zoom = max(0.02, min(max_zoom, self.zoom * factor))
        self.render()

    def zoom_fit(self):
        self.fit_mode = True
        self.render()

    def zoom_actual(self):
        self.fit_mode = False
        self.zoom = 1.0
        self.render()

    def _on_wheel(self, event):
        self.zoom_by(1.15 if event.delta > 0 else 1 / 1.15)

    def canvas_to_image(self, ex, ey):
        cx, cy = self.canvas.canvasx(ex), self.canvas.canvasy(ey)
        return (cx - self.offset[0]) / self.zoom, (cy - self.offset[1]) / self.zoom

    def _on_motion(self, event):
        if self.image is None:
            return
        x, y = self.canvas_to_image(event.x, event.y)
        w, h = self.image.size
        if 0 <= x < w and 0 <= y < h:
            r, g, b = self.image.getpixel((int(x), int(y)))
            self.status_pixel.config(text=f"x={int(x)}, y={int(y)}   RGB({r}, {g}, {b})")
        else:
            self.status_pixel.config(text="")

    # ------------------------------------------------------------ fare modları
    def set_mode(self, mode):
        self.mode = mode
        if self.crop_rect:
            self.canvas.delete(self.crop_rect)
            self.crop_rect = None
        self.crop_start = None
        cursors = {"crop": "crosshair", "text": "xterm"}
        self.canvas.config(cursor=cursors.get(mode, "arrow"))
        if mode == "crop":
            self.set_status("✂ Kırpma modu: fareyle bir alan seçin (Esc: iptal)")
        elif mode == "text":
            self.set_status("T Yazı modu: yazının konulacağı yere tıklayın (Esc: iptal)")
        else:
            self.set_status("Hazır")

    def toggle_crop(self):
        if self.require_image():
            self.set_mode(None if self.mode == "crop" else "crop")

    def toggle_text(self):
        if self.require_image():
            self.set_mode(None if self.mode == "text" else "text")

    def _on_press(self, event):
        if self.image is None:
            self.open_image()
            return
        if self.mode == "crop":
            self.crop_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            if self.crop_rect:
                self.canvas.delete(self.crop_rect)
            self.crop_rect = self.canvas.create_rectangle(*self.crop_start, *self.crop_start,
                                                          outline="#00e5ff", width=2, dash=(6, 3))
        elif self.mode == "text":
            x, y = self.canvas_to_image(event.x, event.y)
            self.set_mode(None)
            self.do_text(int(x), int(y))

    def _on_drag(self, event):
        if self.mode == "crop" and self.crop_start and self.crop_rect:
            self.canvas.coords(self.crop_rect, *self.crop_start,
                               self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _on_release(self, event):
        if self.mode != "crop" or not self.crop_start:
            return
        x0, y0 = self.canvas_to_image(*self._canvas_to_event(self.crop_start))
        x1, y1 = self.canvas_to_image(event.x, event.y)
        w, h = self.image.size
        left, right = sorted((max(0, min(w, round(x0))), max(0, min(w, round(x1)))))
        top, bottom = sorted((max(0, min(h, round(y0))), max(0, min(h, round(y1)))))
        self.set_mode(None)
        if right - left < 2 or bottom - top < 2:
            self.set_status("Kırpma alanı çok küçük, iptal edildi")
            return
        box = (left, top, right, bottom)
        self.apply(lambda im: im.crop(box), f"Kırpma ({right - left}×{bottom - top})")

    def _canvas_to_event(self, pt):
        """Canvas koordinatını tekrar pencere (event) koordinatına çevirir."""
        return pt[0] - self.canvas.canvasx(0), pt[1] - self.canvas.canvasy(0)

    # ------------------------------------------------------------ özel araçlar
    def do_resize(self):
        if not self.require_image():
            return
        res = ResizeDialog(self, self.image.size).result
        if res:
            self.apply(lambda im: resize_image(im, *res), f"Yeniden boyutlandırma ({res[0]}×{res[1]})")

    def do_border(self):
        if not self.require_image():
            return
        vals = self.ask("Çerçeve Ekle", [("Kalınlık (px)", 20, 1, 1000, 1)])
        if not vals:
            return
        color = colorchooser.askcolor(color="#ffffff", title="Çerçeve rengi", parent=self)[0]
        if color:
            color = tuple(int(c) for c in color)
            self.apply(lambda im: add_border(im, vals[0], color), "Çerçeve ekleme")

    def do_text(self, x, y):
        text = simpledialog.askstring("Yazı Ekle", "Eklenecek yazı:", parent=self)
        if not text:
            return
        default_size = max(12, min(self.image.size) // 12)
        vals = self.ask("Yazı Boyutu", [("Yazı boyutu (px)", default_size, 6, 1000, 2)])
        if not vals:
            return
        color = colorchooser.askcolor(color="#ffffff", title="Yazı rengi", parent=self)[0]
        if color:
            color = tuple(int(c) for c in color)
            self.apply(lambda im: add_text(im, (x, y), text, vals[0], color), "Yazı ekleme")

    def do_faces(self, blur):
        if not self.require_image():
            return
        found = {}

        def run(im):
            out, n = detect_faces(im, blur)
            found["n"] = n
            return out if n else None

        name = "Yüz bulanıklaştırma" if blur else "Yüz tespiti"
        self.apply(run, name)
        n = found.get("n")
        if n == 0:
            messagebox.showinfo(name, "Resimde yüz bulunamadı.", parent=self)
        elif n:
            self.set_status(f"✔ {n} yüz bulundu" + (" ve bulanıklaştırıldı" if blur else ""))

    # ------------------------------------------------------------ analiz
    def current_view_image(self):
        if self.adjust_pending():
            return adjust_image(self.image, **self.adjust_params())
        return self.image

    def show_histogram(self):
        if not self.require_image():
            return
        img = self.current_view_image()
        win = tk.Toplevel(self)
        win.title("Histogram")
        win.resizable(False, False)
        W, H, P = 600, 320, 30
        c = tk.Canvas(win, width=W, height=H, bg="white")
        c.pack(padx=10, pady=10)

        hist = img.histogram()
        channels = [("Kırmızı", "#e53935", hist[0:256]),
                    ("Yeşil", "#43a047", hist[256:512]),
                    ("Mavi", "#1e88e5", hist[512:768]),
                    ("Parlaklık", "#424242", img.convert("L").histogram())]
        mx = max(max(h) for _, _, h in channels) or 1

        # Eksenler
        c.create_line(P, H - P, W - P, H - P)
        c.create_line(P, P, P, H - P)
        for v in (0, 64, 128, 192, 255):
            x = P + v * (W - 2 * P) / 255
            c.create_line(x, H - P, x, H - P + 4)
            c.create_text(x, H - P + 14, text=str(v), font=("Segoe UI", 8))

        for _, color, h in channels:
            pts = []
            for i, v in enumerate(h):
                pts += [P + i * (W - 2 * P) / 255, H - P - v / mx * (H - 2 * P)]
            c.create_line(*pts, fill=color, width=1.5)

        arr = np.asarray(img, dtype=np.float32)
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        stats = ttk.Frame(win)
        stats.pack(padx=10, pady=(0, 10), fill="x")
        rows = [("Kanal", "Ortalama", "Std. sapma", "Min", "Max")]
        for i, (name, _, _) in enumerate(channels[:3]):
            ch_ = arr[..., i]
            rows.append((name, f"{ch_.mean():.1f}", f"{ch_.std():.1f}", f"{ch_.min():.0f}", f"{ch_.max():.0f}"))
        rows.append(("Parlaklık", f"{gray.mean():.1f}", f"{gray.std():.1f}", f"{gray.min():.0f}", f"{gray.max():.0f}"))
        for r, row in enumerate(rows):
            color = channels[r - 1][1] if r else "black"
            for col, val in enumerate(row):
                ttk.Label(stats, text=val, foreground=color if col == 0 else "black",
                          font=("Segoe UI", 9, "bold" if r == 0 else "normal"), width=12).grid(row=r, column=col)

    def show_info(self):
        if not self.require_image():
            return
        w, h = self.image.size
        ow, oh = self.original.size
        lines = [
            f"Dosya: {self.path or '-'}",
            f"Dosya boyutu: {os.path.getsize(self.path) / 1024:.1f} KB" if self.path and os.path.exists(self.path) else "",
            f"Orijinal boyut: {ow} × {oh} px",
            f"Mevcut boyut: {w} × {h} px  ({w * h / 1e6:.2f} MP)",
            f"En-boy oranı: {w / h:.3f}",
            f"Renk modu: {self.image.mode}",
            f"Geri alınabilir işlem sayısı: {len(self.history)}",
            f"Kaydedilmemiş değişiklik: {'Evet' if self.dirty else 'Hayır'}",
        ]
        messagebox.showinfo("Resim Bilgisi", "\n".join(l for l in lines if l), parent=self)


# =============================================================================
if __name__ == "__main__":
    try:  # Windows'ta yüksek DPI ekranlarda net görüntü
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:  # noqa: BLE001
        pass
    app = ImageEditorApp(sys.argv[1] if len(sys.argv) > 1 else None)
    app.mainloop()
