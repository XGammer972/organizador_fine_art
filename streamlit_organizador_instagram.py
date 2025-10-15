# -*- coding: utf-8 -*-
"""
streamlit_organizador_instagram.py
Versión final unificada y optimizada (incluye refinamiento cromático)
- Todos los sliders anteriores: num_clusters, reduccion, peso_color, peso_luminosidad, umbral_similitud
- Nuevos sliders: sensibilidad_cromatica, peso_lum_pipeline
- Botones: clásico, paleta (pipeline), reorganizar, generar PDF, limpiar sesión
- Previsualización feed 3x3 y barras de color
- Barra horizontal de gradiente
- Caching para cálculos pesados
- Reducción temprana de imágenes (draft + thumbnail)
- Logging a archivo + panel descargable
- Optimizado para Streamlit Cloud (opencv-python-headless)
"""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io, os, tempfile, logging, traceback
import colorgram
from sklearn.cluster import KMeans
from fpdf import FPDF

# Intentar importar cv2 (headless recomendado en requirements)
try:
    import cv2
    _have_cv2 = True
except Exception:
    cv2 = None
    _have_cv2 = False

# ------------------------
# Logging (archivo temporal)
# ------------------------
log_path = os.path.join(tempfile.gettempdir(), "organizador_log.txt")
logging.basicConfig(
    filename=log_path,
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("=== Nueva sesión iniciada ===")

# ------------------------
# Streamlit page config
# ------------------------
st.set_page_config(page_title="Organizador Fine Art — Color + Lum", layout="wide")
st.title("🎨 Organizador Fine Art — Color + Luminosidad")
st.write("Ajustá los sliders en la barra lateral antes de subir imágenes. Usa los botones para organizar y exportar.")

# ------------------------
# Sidebar sliders (todos los pedidos)
# ------------------------
st.sidebar.header("⚙️ Ajustes de análisis y refinamiento")

# Original sliders
num_clusters = st.sidebar.slider("Número de clusters (paleta por imagen)", 2, 10, 3)
reduccion = st.sidebar.slider("Reducción para análisis (px)", 100, 800, 300, step=50)
peso_color = st.sidebar.slider("Peso del color (0.0–1.0)", 0.0, 1.0, 0.6, step=0.05)
peso_lum = st.sidebar.slider("Peso de la luminosidad (0.0–1.0)", 0.0, 1.0, 0.4, step=0.05)
umbral_similitud = st.sidebar.slider("Umbral similitud color (menor = más estricto)", 10, 400, 120)

# New refinement sliders (added here)
sensibilidad_cromatica = st.sidebar.slider(
    "Sensibilidad cromática (0=suave,1=fuerte)", 0.0, 1.0, 0.6, step=0.05
)
peso_lum_pipeline = st.sidebar.slider(
    "Peso del orden por luminosidad en pipeline (0=ignorar,1=priorizar)", 0.0, 1.0, 0.6, step=0.05
)

# Misc options
modo_bn = st.sidebar.checkbox("🔲 Usar blanco y negro para cálculo de luminosidad (suaviza gradiente)", value=False)
invertir = st.sidebar.checkbox("↔️ Invertir orden final", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Nota: ajustá sliders antes de presionar los botones de organización.")

# ------------------------
# Helper utilities
# ------------------------

def pil_to_bytes(pil_img, fmt="JPEG", quality=85):
    """Convierte imagen PIL a bytes (útil para caching y colorgram)."""
    b = io.BytesIO()
    pil_img.save(b, format=fmt, quality=quality)
    b.seek(0)
    return b.getvalue()

@st.cache_data
def compute_luminosity_from_bytes(img_bytes, use_bn=False):
    """
    Cached: calcula luminosidad desde bytes.
    - use_bn: si True se analiza en L (blanco y negro)
    """
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if use_bn:
            img = img.convert("L")
        arr = np.array(img.convert("L").resize((64, 64)))
        return float(np.mean(arr))
    except Exception as e:
        logging.exception(f"compute_luminosity_from_bytes failed: {e}")
        return 0.0

@st.cache_data
def compute_colorgram_palette_from_bytes(img_bytes, n_colors=3):
    """
    Cached: obtiene paleta dominante con colorgram desde bytes.
    Devuelve lista de RGB tuples.
    """
    try:
        bio = io.BytesIO(img_bytes)
        colors = colorgram.extract(bio, n_colors)
        pal = [(int(c.rgb.r), int(c.rgb.g), int(c.rgb.b)) for c in colors]
        return pal
    except Exception as e:
        logging.exception(f"compute_colorgram_palette_from_bytes failed: {e}")
        return []

def reduce_image_safe(img, max_size):
    """
    Reducción temprana y segura:
    - intentamos 'draft' cuando esté disponible (puede no soportarlo)
    - convertimos a RGB y hacemos thumbnail
    """
    try:
        img_copy = img.copy()
        try:
            img_copy.draft("RGB", max_size)
        except Exception:
            pass
        img_copy = img_copy.convert("RGB")
        img_copy.thumbnail(max_size)
        return img_copy
    except Exception as e:
        logging.exception(f"reduce_image_safe failed: {e}")
        return img

def rgb_to_lab_numpy(rgb):
    """Convierte (r,g,b) a Lab usando OpenCV si está disponible, fallback por aproximación."""
    try:
        if _have_cv2:
            arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            return lab[0,0].astype(float)
        else:
            # Fallback: usar rgb scaled -> pseudo-Lab (L ~ luminance, a,b ~ chroma approxim)
            r,g,b = rgb
            L = 0.2126*r + 0.7152*g + 0.0722*b
            a = r - g
            bval = (r + g)/2 - b
            return np.array([L, a, bval], dtype=float)
    except Exception as e:
        logging.exception(f"rgb_to_lab_numpy failed: {e}")
        return np.array([0.0,0.0,0.0], dtype=float)

def lab_distance(c1_lab, c2_lab):
    """Distancia euclidiana en Lab (proxy ΔE)."""
    return float(np.linalg.norm(np.array(c1_lab) - np.array(c2_lab)))

def create_color_bar(colores, ancho=300, alto=30):
    """Crea barra PIL con la paleta dada (lista de tuples RGB)."""
    if not colores:
        return None
    barra = Image.new("RGB", (ancho, alto), (255,255,255))
    draw = ImageDraw.Draw(barra)
    n = len(colores)
    w = max(1, ancho // n)
    for i, c in enumerate(colores):
        draw.rectangle([i*w, 0, (i+1)*w, alto], fill=c)
    return barra

def create_gradient_overview(colores_seq, ancho_por_color=30, alto=50):
    """Crea un numpy image con tiras de colores en secuencia."""
    if not colores_seq:
        return None
    n = len(colores_seq)
    img = np.zeros((alto, n*ancho_por_color, 3), dtype=np.uint8)
    for i, c in enumerate(colores_seq):
        img[:, i*ancho_por_color:(i+1)*ancho_por_color, :] = c
    return img

# ------------------------
# Session state init
# ------------------------
if "images_reduced" not in st.session_state:
    st.session_state["images_reduced"] = []  # list of dicts: {name, bytes, pil}
if "ordered" not in st.session_state:
    st.session_state["ordered"] = []  # list of dicts of ordered items
if "last_method" not in st.session_state:
    st.session_state["last_method"] = None  # 'classic' or 'palette'

# ------------------------
# File uploader (we DO NOT organize automatically)
# - We read all files into reduced bytes+PIL to save memory
# ------------------------
uploaded_files = st.file_uploader(
    "📁 Subí tus fotos (JPG/PNG). Ajustá sliders antes de organizar.",
    type=["jpg","jpeg","png"], accept_multiple_files=True
)

if uploaded_files:
    # Reset images_reduced on new upload
    st.session_state["images_reduced"] = []
    count_ok = 0
    for f in uploaded_files:
        try:
            raw = f.read()
            pil = Image.open(io.BytesIO(raw))
            pil = reduce_image_safe(pil, (reduccion, reduccion))  # reduce asap
            b = pil_to_bytes(pil, fmt="JPEG", quality=85)
            st.session_state["images_reduced"].append({
                "name": f.name,
                "bytes": b,
                "pil": pil
            })
            count_ok += 1
        except Exception as e:
            logging.exception(f"Error loading file {f.name}: {e}")
            st.error(f"Error cargando {f.name} (ver log).")
    st.success(f"{count_ok} imágenes preparadas para análisis (reducidas).")

# ------------------------
# Buttons: Classic / Palette pipeline / Reorder
# ------------------------
col1, col2, col3 = st.columns([1,1,1])
with col1:
    btn_classic = st.button("🎨 Organizar: Color promedio + Lum (clásico)")
with col2:
    btn_palette = st.button("🌈 Organizar: Paletas (colorgram) + Lum (pipeline)")
with col3:
    btn_reorder = st.button("🔁 Reorganizar (reaplica parámetros actuales)")

# ------------------------
# Core algorithms
# ------------------------

def compute_classic_order(images_info):
    """
    Classic ordering:
    - color_mean (RGB) from reduced PIL
    - lum from cached bytes
    - score = peso_color * ||color|| + peso_lum * lum
    """
    res = []
    for it in images_info:
        try:
            b = it["bytes"]
            lum = compute_luminosity_from_bytes(b, use_bn=modo_bn)
            arr = np.array(it["pil"]).reshape(-1,3).astype(float)
            color_mean = np.mean(arr, axis=0)
            color_norm = np.linalg.norm(color_mean)
            score = peso_color * color_norm + peso_lum * lum
            res.append({
                "name": it["name"],
                "pil": it["pil"],
                "bytes": b,
                "lum": lum,
                "color_mean": tuple(map(int, color_mean)),
                "score": float(score)
            })
        except Exception as e:
            logging.exception(f"compute_classic_order item failed: {e}")
    # sort by score ascending
    res_sorted = sorted(res, key=lambda x: x["score"])
    return res_sorted

def compute_palette_pipeline_order(images_info):
    """
    Pipeline (refinada):
    1) compute lum (cached) and paleta (colorgram cached)
    2) initial sort by weighted luminosity (peso_lum_pipeline)
    3) greedy smoothing by Lab distance with sensibilidad_cromatica weight
    """
    res = []
    for it in images_info:
        try:
            b = it["bytes"]
            lum = compute_luminosity_from_bytes(b, use_bn=modo_bn)
            pal = compute_colorgram_palette_from_bytes(b, n_colors=num_clusters)
            if pal:
                dominant = tuple(map(int, np.mean(np.array(pal), axis=0)))
            else:
                arr = np.array(it["pil"]).reshape(-1,3).astype(float)
                dominant = tuple(map(int, np.mean(arr, axis=0)))
            # compute lab
            dominant_lab = rgb_to_lab_numpy(dominant)
            res.append({
                "name": it["name"],
                "pil": it["pil"],
                "bytes": b,
                "lum": float(lum),
                "palette": pal,
                "dominant_rgb": dominant,
                "dominant_lab": dominant_lab
            })
        except Exception as e:
            logging.exception(f"compute_palette_pipeline_order item failed: {e}")

    # Step 2: initial ordering by luminosity weighted by peso_lum_pipeline
    try:
        res.sort(key=lambda x: x["lum"] * peso_lum_pipeline)
    except Exception:
        res.sort(key=lambda x: x["lum"])

    # Step 3: greedy nearest neighbor smoothing in Lab
    ordered = []
    remaining = res.copy()
    if not remaining:
        return []
    ordered.append(remaining.pop(0))
    while remaining:
        last_lab = ordered[-1]["dominant_lab"]
        # compute distances with sensitivity: multiply by (1 - sensibilidad_cromatica) to reduce effect if low
        dists = []
        for x in remaining:
            d = lab_distance(last_lab, x["dominant_lab"])
            # adjust by sensibilidad_cromatica: if high, d keeps strong; if low, dampen differences
            adjusted = d * (1.0 + (1.0 - sensibilidad_cromatica))
            dists.append(adjusted)
        idx = int(np.argmin(dists))
        ordered.append(remaining.pop(idx))
    # Optionally, we could apply grouping by umbral_similitud here — left as future tweak
    return ordered

# ------------------------
# Execute on button presses with try/except + logging
# ------------------------
try:
    if btn_classic and st.session_state.get("images_reduced"):
        with st.spinner("Organizando (método clásico)..."):
            st.session_state["ordered"] = compute_classic_order(st.session_state["images_reduced"])
            st.session_state["last_method"] = "classic"
            if invertir:
                st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
        st.success("Organización clásica completada.")

    if btn_palette and st.session_state.get("images_reduced"):
        with st.spinner("Organizando (pipeline paletas + luminosidad)..."):
            st.session_state["ordered"] = compute_palette_pipeline_order(st.session_state["images_reduced"])
            st.session_state["last_method"] = "palette"
            if invertir:
                st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
        st.success("Organización por paletas completada.")

    if btn_reorder and st.session_state.get("images_reduced"):
        # reaplicar según last_method
        method = st.session_state.get("last_method", "classic")
        with st.spinner("Reaplicando organización con parámetros actuales..."):
            if method == "classic":
                st.session_state["ordered"] = compute_classic_order(st.session_state["images_reduced"])
            else:
                st.session_state["ordered"] = compute_palette_pipeline_order(st.session_state["images_reduced"])
            if invertir:
                st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
        st.success("Reorganización completada.")
except MemoryError as e:
    logging.exception(f"MemoryError during organization: {e}\n{traceback.format_exc()}")
    st.error("MemoryError: reducí 'reducción' o subí menos imágenes.")
except Exception as e:
    logging.exception(f"Unexpected error in organization: {e}\n{traceback.format_exc()}")
    st.error("Error inesperado al organizar. Revisá el log.")

# ------------------------
# If ordered result exists: show gradient, preview 3x3, bars, PDF export, cleanup
# ------------------------
if st.session_state.get("ordered"):
    ordered = st.session_state["ordered"]

    # Build sequence of representative colors for gradient overview
    colores_seq = []
    for item in ordered:
        if item.get("palette"):
            colores_seq.append(tuple(map(int, item["palette"][0])))  # first color of palette
        elif item.get("color_mean"):
            colores_seq.append(tuple(map(int, item["color_mean"])))
        elif item.get("dominant_rgb"):
            colores_seq.append(tuple(map(int, item["dominant_rgb"])))
        else:
            colores_seq.append((200,200,200))

    # Gradient overview (horizontal)
    st.subheader("🎨 Gradiente cromático")
    grad = create_gradient_overview(colores_seq, ancho_por_color=30, alto=50)
    if grad is not None:
        st.image(grad, use_container_width=True)

    # Previsualización feed 3x3
    st.subheader("🖼️ Previsualización (3x3)")
    n = len(ordered)
    for i in range(0, n, 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < n:
                item = ordered[idx]
                pil_img = item.get("pil")
                caption = item.get("name", f"{idx+1}")
                with col:
                    # use_container_width to avoid deprecation warning (we fixed it)
                    st.image(pil_img, caption=caption, use_container_width=True)
                    # show color bar (palette preferred)
                    if item.get("palette"):
                        bar = create_color_bar(item["palette"], ancho=300, alto=30)
                        if bar: st.image(bar, use_container_width=True)
                    elif item.get("color_mean"):
                        bar = create_color_bar([tuple(map(int, item["color_mean"]))], ancho=300, alto=30)
                        if bar: st.image(bar, use_container_width=True)
                    else:
                        bar = create_color_bar([(200,200,200)], ancho=300, alto=30)
                        if bar: st.image(bar, use_container_width=True)

    # Export PDF and cleanup
    st.subheader("📄 Exportar y limpiar")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("🧾 Generar PDF (feed 3x3)"):
            try:
                with st.spinner("Generando PDF (optimizado)..."):
                    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    pdf = FPDF(unit="mm", format="A4")
                    pdf.set_auto_page_break(False)
                    page_w, page_h = 210, 297
                    margin = 10
                    cell_w = (page_w - 2*margin) / 3.0
                    cell_h = cell_w
                    # iterate in blocks of 9
                    for i in range(0, len(ordered), 9):
                        pdf.add_page()
                        block = ordered[i:i+9]
                        for idx, item in enumerate(block):
                            r = idx // 3
                            ccol = idx % 3
                            x_mm = margin + ccol * cell_w
                            # pdf coordinates: y from top = margin + r*(cell_h + some gap)
                            y_mm = margin + r * (cell_h + 15)
                            # prepare image reduced for PDF (limit resolution)
                            pil_img = item.get("pil")
                            if pil_img is None and item.get("bytes"):
                                pil_img = Image.open(io.BytesIO(item["bytes"]))
                            if pil_img is None:
                                pil_img = Image.new("RGB", (600,600), (240,240,240))
                            im_copy = pil_img.copy()
                            im_copy.thumbnail((800, 800))  # <-- PDF optimization (change previously causing crash)
                            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            im_copy.save(tmp_img.name, "JPEG", quality=60, optimize=True, progressive=True)
                            # draw image (FPDF expects path)
                            # Place image at (x_mm, page_h - y_mm - cell_h) because FPDF places origin at top-left differently
                            pdf.image(tmp_img.name, x=x_mm, y=y_mm, w=cell_w, h=cell_h)
                            # draw color bar under image if exists
                            bar_img = None
                            if item.get("palette"):
                                bar_img = create_color_bar(item["palette"], ancho=300, alto=30)
                            elif item.get("color_mean"):
                                bar_img = create_color_bar([tuple(map(int, item["color_mean"]))], ancho=300, alto=30)
                            if bar_img:
                                tmp_bar = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                                bar_img.save(tmp_bar.name, "JPEG", quality=70)
                                # bar position: just below image
                                pdf.image(tmp_bar.name, x=x_mm, y=y_mm + cell_h - 8, w=cell_w, h=8)
                                try: os.remove(tmp_bar.name)
                                except: pass
                            try:
                                tmp_img.close()
                                os.remove(tmp_img.name)
                            except:
                                pass
                    pdf.output(tmp_pdf.name)
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("⬇️ Descargar PDF final", f, file_name="feed_instagram.pdf")
                    st.success("PDF generado correctamente.")
                    try:
                        tmp_pdf.close(); os.remove(tmp_pdf.name)
                    except: pass
            except MemoryError as me:
                logging.exception(f"MemoryError in PDF generation: {me}")
                st.error("MemoryError: reducí 'reducción' o subí menos imágenes antes de generar PDF.")
            except Exception as e:
                logging.exception(f"Exception in PDF generation: {e}\n{traceback.format_exc()}")
                st.error("Error al generar PDF. Revisá el log.")

    with c2:
        if st.button("🔄 Limpiar sesión (borrar imágenes y orden)"):
            st.session_state["images_reduced"] = []
            st.session_state["ordered"] = []
            st.session_state["last_method"] = None
            st.success("Sesión limpiada.")

    with c3:
        # Button to refine gradient further with local smoothing (optional)
        if st.button("✨ Refinar gradiente (ajuste fino local)"):
            try:
                # Local smoothing: greedy local swaps to reduce Lab jumps
                seq = st.session_state["ordered"]
                if not seq or len(seq) < 3:
                    st.info("No hay suficientes imágenes para refinar.")
                else:
                    # Convert representative lab list
                    labs = []
                    for it in seq:
                        if it.get("dominant_lab") is not None:
                            labs.append(it["dominant_lab"])
                        elif it.get("color_mean") is not None:
                            labs.append(rgb_to_lab_numpy(it["color_mean"]))
                        elif it.get("dominant_rgb") is not None:
                            labs.append(rgb_to_lab_numpy(it["dominant_rgb"]))
                        else:
                            labs.append(np.array([0.0,0.0,0.0]))
                    # Attempt a few local swaps that reduce adjacent distance sum
                    improved = True
                    iterations = 0
                    max_iter = 200
                    while improved and iterations < max_iter:
                        improved = False
                        iterations += 1
                        for i in range(len(seq)-2):
                            # check swap i+1 and i+2
                            a = labs[i]; b = labs[i+1]; c = labs[i+2]
                            current = lab_distance(a,b)+lab_distance(b,c)
                            swapped = lab_distance(a,c)+lab_distance(c,b)
                            if swapped < current:
                                # perform swap in both seq and labs
                                seq[i+1], seq[i+2] = seq[i+2], seq[i+1]
                                labs[i+1], labs[i+2] = labs[i+2], labs[i+1]
                                improved = True
                    st.session_state["ordered"] = seq
                    st.success("Refinamiento aplicado.")
            except Exception as e:
                logging.exception(f"Error during gradient refinement: {e}")
                st.error("Error al refinar gradiente. Revisá el log.")

# ------------------------
# Logging panel and download
# ------------------------
st.markdown("---")
st.subheader("🧾 Registro de errores (debug)")

if os.path.exists(log_path):
    try:
        with open(log_path, "r", encoding="utf-8") as f:
            log_content = f.read()
    except Exception as e:
        log_content = f"Error leyendo log: {e}"
else:
    log_content = ""

with st.expander("📜 Ver últimos errores / advertencias"):
    st.code(log_content[-2000:] if log_content else "No hay entradas en el log.")

if log_content:
    st.download_button("⬇️ Descargar log completo", data=log_content, file_name="organizador_log.txt", mime="text/plain")

# ------------------------
# End of script
# ------------------------
