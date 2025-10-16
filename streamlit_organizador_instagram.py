# -*- coding: utf-8 -*-
"""
Organizador Fine Art — Versión Final Unificada
Autor: Naim Heredia + GPT-5
Fecha: 2025-10
Descripción:
App de Streamlit para organizar fotos artísticamente por color y luminosidad,
generar vistas tipo feed de Instagram y exportar PDF. Incluye refinamiento
de gradientes cromáticas y manejo de errores/logs.
"""

# ==========================================================
# 1️⃣ IMPORTS Y CONFIGURACIÓN BASE
# ==========================================================
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io, os, tempfile, logging
import colorgram
from sklearn.cluster import KMeans
from fpdf import FPDF

# Intentar importar OpenCV (usando versión headless para Streamlit Cloud)
try:
    import cv2
    _have_cv2 = True
except Exception:
    cv2 = None
    _have_cv2 = False

# Configuración inicial de la página
st.set_page_config(page_title="Organizador Fine Art — Color + Luminosidad", layout="wide")
st.title("🎨 Organizador Fine Art — Color + Luminosidad")
st.write("Ajustá los sliders antes de subir las imágenes y generá tu composición estética.")

# ==========================================================
# 2️⃣ SISTEMA DE LOGGING
# ==========================================================
log_path = os.path.join(tempfile.gettempdir(), "organizador_log.txt")
logging.basicConfig(
    filename=log_path,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("=== Nueva sesión iniciada ===")

# ==========================================================
# 3️⃣ PANEL DE SLIDERS Y PARÁMETROS
# ==========================================================
st.sidebar.header("⚙️ Parámetros de análisis y refinamiento")

num_clusters = st.sidebar.slider("Número de clústeres (paleta por imagen)", 2, 10, 3)
reduccion = st.sidebar.slider("Reducción para análisis (px)", 100, 800, 300, step=50)
peso_color = st.sidebar.slider("Peso del color", 0.0, 1.0, 0.6, step=0.05)
peso_lum = st.sidebar.slider("Peso de la luminosidad", 0.0, 1.0, 0.4, step=0.05)
sensibilidad_cromatica = st.sidebar.slider("Sensibilidad cromática", 0.0, 1.0, 0.6, step=0.05)
peso_lum_pipeline = st.sidebar.slider("Peso luminosidad pipeline", 0.0, 1.0, 0.6, step=0.05)
suavizado_local = st.sidebar.slider("Nivel de suavizado local", 0.0, 1.0, 0.4, step=0.05)
modo_bn = st.sidebar.checkbox("🔲 Usar B/N para análisis de luminosidad", value=False)
invertir = st.sidebar.checkbox("↔️ Invertir orden final", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Ajustá estos valores antes de subir tus imágenes para mejores resultados.")

# ==========================================================
# 4️⃣ FUNCIONES AUXILIARES
# ==========================================================

def pil_to_bytes(pil_img, fmt="JPEG", quality=85):
    """Convierte una imagen PIL en bytes comprimidos."""
    b = io.BytesIO()
    pil_img.save(b, format=fmt, quality=quality)
    b.seek(0)
    return b.getvalue()

@st.cache_data
def compute_luminosity_from_bytes(img_bytes, use_bn=False):
    """Calcula luminosidad promedio (escala de grises)."""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if use_bn:
            img = img.convert("L")
        arr = np.array(img.convert("L").resize((64, 64)))
        return float(np.mean(arr))
    except Exception as e:
        logging.exception(f"Luminosity computation failed: {e}")
        return 0.0

@st.cache_data
def compute_colorgram_palette_from_bytes(img_bytes, n_colors=3):
    """Extrae paleta dominante usando colorgram."""
    try:
        bio = io.BytesIO(img_bytes)
        colors = colorgram.extract(bio, n_colors)
        return [(int(c.rgb.r), int(c.rgb.g), int(c.rgb.b)) for c in colors]
    except Exception as e:
        logging.exception(f"Colorgram palette failed: {e}")
        return []

def reduce_image_safe(img, max_size):
    """Reduce tamaño de imagen para ahorrar memoria."""
    try:
        img_copy = img.copy()
        try: img_copy.draft("RGB", max_size)
        except Exception: pass
        img_copy = img_copy.convert("RGB")
        img_copy.thumbnail(max_size)
        return img_copy
    except Exception as e:
        logging.exception(f"reduce_image_safe failed: {e}")
        return img

def rgb_to_lab_numpy(rgb):
    """Convierte color RGB a espacio LAB (usa OpenCV si disponible)."""
    try:
        if _have_cv2:
            arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            return lab[0, 0].astype(float)
        else:
            r, g, b = rgb
            L = 0.2126*r + 0.7152*g + 0.0722*b
            a = r - g
            bval = (r + g)/2 - b
            return np.array([L, a, bval], dtype=float)
    except Exception as e:
        logging.exception(f"rgb_to_lab_numpy failed: {e}")
        return np.array([0,0,0], dtype=float)

def lab_distance(c1, c2):
    """Distancia euclidiana entre dos colores en LAB."""
    return float(np.linalg.norm(np.array(c1) - np.array(c2)))

def create_color_bar(colores, ancho=300, alto=30):
    """Crea barra horizontal representando una paleta."""
    if not colores: return None
    barra = Image.new("RGB", (ancho, alto), (255,255,255))
    d = ImageDraw.Draw(barra)
    n = len(colores)
    w = max(1, ancho // n)
    for i, c in enumerate(colores):
        d.rectangle([i*w, 0, (i+1)*w, alto], fill=c)
    return barra

def create_gradient_overview(colores_seq, ancho_por_color=30, alto=50):
    """Genera imagen horizontal mostrando transición de color."""
    if not colores_seq: return None
    n = len(colores_seq)
    img = np.zeros((alto, n*ancho_por_color, 3), dtype=np.uint8)
    for i, c in enumerate(colores_seq):
        img[:, i*ancho_por_color:(i+1)*ancho_por_color, :] = c
    return img

# ==========================================================
# 5️⃣ CARGA DE IMÁGENES
# ==========================================================
if "images_reduced" not in st.session_state:
    st.session_state["images_reduced"] = []
if "ordered" not in st.session_state:
    st.session_state["ordered"] = []
if "last_method" not in st.session_state:
    st.session_state["last_method"] = None

uploaded_files = st.file_uploader(
    "📁 Subí tus fotos (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True
)

if uploaded_files:
    st.session_state["images_reduced"] = []
    for f in uploaded_files:
        try:
            raw = f.read()
            pil = Image.open(io.BytesIO(raw))
            pil = reduce_image_safe(pil, (reduccion, reduccion))
            b = pil_to_bytes(pil)
            st.session_state["images_reduced"].append({
                "name": f.name, "bytes": b, "pil": pil
            })
        except Exception as e:
            logging.exception(f"Error al cargar {f.name}: {e}")
            st.error(f"Error cargando {f.name}. Ver log.")
    st.success(f"{len(st.session_state['images_reduced'])} imágenes preparadas para análisis.")

# ==========================================================
# 6️⃣ BOTONES DE ORGANIZACIÓN
# ==========================================================
col1, col2, col3 = st.columns([1,1,1])
with col1: btn_classic = st.button("🎨 Organizar clásico")
with col2: btn_palette = st.button("🌈 Pipeline paleta + luminosidad")
with col3: btn_reorder = st.button("🔁 Reorganizar con ajustes actuales")

# ==========================================================
# 7️⃣ FUNCIONES DE ORGANIZACIÓN
# ==========================================================
def compute_classic_order(images):
    """Ordena usando color promedio + luminosidad."""
    res = []
    for it in images:
        lum = compute_luminosity_from_bytes(it["bytes"], use_bn=modo_bn)
        arr = np.array(it["pil"]).reshape(-1, 3).astype(float)
        color_mean = np.mean(arr, axis=0)
        score = peso_color*np.linalg.norm(color_mean) + peso_lum*lum
        res.append({
            "name": it["name"], "pil": it["pil"], "bytes": it["bytes"],
            "color_mean": tuple(map(int,color_mean)), "lum": lum, "score": score
        })
    return sorted(res, key=lambda x: x["score"])

def compute_palette_pipeline_order(images):
    """Pipeline avanzado: paleta dominante + ΔE + luminosidad."""
    res = []
    for it in images:
        lum = compute_luminosity_from_bytes(it["bytes"], use_bn=modo_bn)
        pal = compute_colorgram_palette_from_bytes(it["bytes"], num_clusters)
        dominant = tuple(map(int, np.mean(np.array(pal), axis=0))) if pal else (128,128,128)
        lab = rgb_to_lab_numpy(dominant)
        res.append({
            "name": it["name"], "pil": it["pil"], "bytes": it["bytes"],
            "lum": lum, "palette": pal, "dominant_rgb": dominant, "dominant_lab": lab
        })
    res.sort(key=lambda x: x["lum"]*peso_lum_pipeline)
    ordered = [res.pop(0)]
    while res:
        last_lab = ordered[-1]["dominant_lab"]
        idx = min(range(len(res)), key=lambda i: lab_distance(last_lab, res[i]["dominant_lab"]))
        ordered.append(res.pop(idx))
    return ordered

# ==========================================================
# 8️⃣ BOTONES DE EJECUCIÓN
# ==========================================================
if btn_classic and st.session_state["images_reduced"]:
    with st.spinner("Organizando (clásico)..."):
        st.session_state["ordered"] = compute_classic_order(st.session_state["images_reduced"])
        if invertir: st.session_state["ordered"].reverse()
        st.session_state["last_method"] = "classic"
    st.success("Organización clásica completada.")

if btn_palette and st.session_state["images_reduced"]:
    with st.spinner("Organizando (pipeline)..."):
        st.session_state["ordered"] = compute_palette_pipeline_order(st.session_state["images_reduced"])
        if invertir: st.session_state["ordered"].reverse()
        st.session_state["last_method"] = "palette"
    st.success("Organización por paletas completada.")

if btn_reorder and st.session_state["ordered"]:
    with st.spinner("Reaplicando organización..."):
        method = st.session_state["last_method"]
        if method == "classic":
            st.session_state["ordered"] = compute_classic_order(st.session_state["images_reduced"])
        else:
            st.session_state["ordered"] = compute_palette_pipeline_order(st.session_state["images_reduced"])
        if invertir: st.session_state["ordered"].reverse()
    st.success("Reorganización completada.")

# ==========================================================
# 9️⃣ SUAVIZADO Y REFINAMIENTO DE GRADIENTE
# ==========================================================
def ordenar_por_gradiente_lab(imagenes):
    if not imagenes: return []
    labs = [rgb_to_lab_numpy(img.get("dominant_rgb") or img.get("color_mean", (0,0,0))) for img in imagenes]
    start = int(np.argmin([l[0] for l in labs]))
    ordered = [imagenes[start]]
    used = {start}
    curr = labs[start]
    for _ in range(len(imagenes)-1):
        idx = min((i for i in range(len(labs)) if i not in used),
                  key=lambda i: lab_distance(curr, labs[i]))
        ordered.append(imagenes[idx])
        used.add(idx)
        curr = labs[idx]
    return ordered

def aplicar_suavizado_local(imagenes, nivel):
    """Interpola colores entre vecinos para transición más fluida."""
    if not imagenes or len(imagenes)<3 or nivel<=0: return imagenes
    out = []
    for i in range(len(imagenes)):
        if i==0 or i==len(imagenes)-1:
            out.append(imagenes[i])
            continue
        def get_rgb(it): return np.array(it.get("dominant_rgb") or it.get("color_mean") or (128,128,128))
        rgb_prev, rgb_curr, rgb_next = get_rgb(imagenes[i-1]), get_rgb(imagenes[i]), get_rgb(imagenes[i+1])
        mix = rgb_prev*(nivel/2) + rgb_curr*(1-nivel) + rgb_next*(nivel/2)
        it_new = imagenes[i].copy()
        it_new["dominant_rgb"] = tuple(map(int, mix))
        out.append(it_new)
    return out

if st.session_state["ordered"]:
    if st.button("🌈 Suavizar gradiente cromática (ΔE + local)"):
        try:
            imgs = st.session_state["ordered"]
            with st.spinner("Suavizando gradiente..."):
                imgs = ordenar_por_gradiente_lab(imgs)
                imgs = aplicar_suavizado_local(imgs, suavizado_local)
                st.session_state["ordered"] = imgs
            st.success("✅ Gradiente suavizada.")
        except Exception as e:
            logging.exception(e)
            st.error("Error en suavizado.")

# ==========================================================
# 🔟 VISUALIZACIÓN Y PDF
# ==========================================================
if st.session_state["ordered"]:
    ordered = st.session_state["ordered"]
    colores_seq = [(it.get("dominant_rgb") or it.get("color_mean") or (200,200,200)) for it in ordered]
    grad = create_gradient_overview(colores_seq)
    st.subheader("🎨 Gradiente cromática general")
    st.image(grad, use_container_width=True)

    st.subheader("🖼️ Feed 3x3 (previsualización)")
    n = len(ordered)
    for i in range(0, n, 3):
        cols = st.columns(3)
        for j, c in enumerate(cols):
            idx = i + j
            if idx < n:
                it = ordered[idx]
                c.image(it["pil"], caption=it["name"], use_container_width=True)
                bar = create_color_bar(it.get("palette") or [it.get("color_mean")])
                if bar:
                    c.image(bar, use_container_width=True)

    if st.button("📄 Generar PDF (feed 3x3)"):
        try:
            pdf = FPDF()
            pdf.set_auto_page_break(False)
            pdf.add_page()
            w, h = 210, 297
            m = 10
            cell_w = (w - 2*m) / 3
            cell_h = cell_w
            for i, it in enumerate(ordered):
                if i % 9 == 0 and i > 0:
                    pdf.add_page()
                r, c_idx = divmod(i, 3)
                x = m + c_idx * cell_w
                y = m + (r % 3) * (cell_h + 15)
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                im = it["pil"].copy()
                im.thumbnail((800,800))
                im.save(tmp.name, "JPEG", quality=60)
                pdf.image(tmp.name, x=x, y=y, w=cell_w, h=cell_h)
                os.remove(tmp.name)
            out = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(out.name)
            with open(out.name, "rb") as f:
                st.download_button("⬇️ Descargar PDF", f, file_name="organizacion_fineart.pdf")
            os.remove(out.name)
        except Exception as e:
            logging.exception(e)
            st.error("Error generando PDF. Revisá log.")

# ==========================================================
# 🧾 PANEL DE LOGS
# ==========================================================
st.markdown("---")
st.subheader("🧾 Registro de errores (debug)")
if os.path.exists(log_path):
    log = open(log_path, "r", encoding="utf-8").read()
    with st.expander("📜 Ver últimos errores"):
        st.text(log[-1500:] if log else "Sin errores recientes.")
    st.download_button("⬇️ Descargar log completo", log, file_name="organizador_log.txt")
else:
    st.info("✅ No hay errores registrados.")
