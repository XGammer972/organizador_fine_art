# -*- coding: utf-8 -*-
"""
streamlit_organizador_instagram.py
Versión final unificada y optimizada:
- sliders (clusters, reducción, peso color, peso luminosidad, umbral)
- botones separados: clásico vs colorgram (pipeline por pasos)
- reordenar sin recargar, invertir orden
- vista previa 3x3 + barras de color bajo cada imagen
- barra horizontal de gradiente cromática
- exportar PDF (feed 3x3) con barras de color
- optimizaciones para evitar MemoryError
- comentarios detallados para aprender/editar
"""

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import io
import os
import tempfile
import colorgram
import cv2
from sklearn.cluster import KMeans
from fpdf import FPDF

# ---------------------------
# CONFIG / TIP: agrega .streamlit/config.toml en tu repo con:
# [server]
# maxUploadSize = 150
# ---------------------------

# evita warnings con imágenes muy grandes
Image.MAX_IMAGE_PIXELS = 200000000

st.set_page_config(page_title="Organizador Fine Art (Color + Lum)", layout="wide")
st.title("🎨 Organizador Fine Art — Color + Luminosidad")
st.markdown("Subí fotos, ajustá parámetros y organiza un feed armónico. Usa *Reorganizar* para volver a aplicar parámetros sin subir archivos otra vez.")

# ---------------------------
# SIDEBAR: Sliders y controles (visibles antes de subir)
# ---------------------------
st.sidebar.header("Controles (ajustá antes de procesar)")

num_clusters = st.sidebar.slider("Número de clusters (paleta por imagen)", 2, 8, 3)
reduccion = st.sidebar.slider("Reducción para análisis (px)", 100, 800, 300, step=50)
peso_color = st.sidebar.slider("Peso del color (0..1)", 0.0, 1.0, 0.6, step=0.05)
peso_lum = st.sidebar.slider("Peso de la luminosidad (0..1)", 0.0, 1.0, 0.4, step=0.05)
umbral_similitud = st.sidebar.slider("Umbral similitud color (menor = más estricto)", 10, 300, 100)

modo_bn = st.sidebar.checkbox("Usar blanco y negro para cálculo de luminosidad (suaviza gradiente)", value=False)
invertir = st.sidebar.checkbox("Invertir orden final", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Métodos de organización (botones en la interfaz principal):")
st.sidebar.write("- Clásico: color promedio + luminosidad\n- Paletas: colorgram (paleta dominante) + luminosidad (pipeline)")

# ---------------------------
# HELPERS: funciones utilitarias y caches
# ---------------------------

def pil_to_bytes(pil_img, fmt="JPEG", quality=85):
    """Convierte PIL -> bytes (para cache o colorgram)."""
    b = io.BytesIO()
    pil_img.save(b, format=fmt, quality=quality)
    b.seek(0)
    return b.getvalue()

@st.cache_data
def compute_luminosity_from_bytes(img_bytes, use_bn):
    """
    Cachea el cálculo de luminosidad a partir de bytes (evita recalculo).
    use_bn: si True convierte a L antes de analizar.
    """
    try:
        img = Image.open(io.BytesIO(img_bytes))
        if use_bn:
            img = img.convert("L")
        # convertimos a tamaño reducido para ahorrar CPU
        arr = np.array(img.convert("L").resize((64, 64)))
        return float(np.mean(arr))
    except Exception:
        return 0.0

@st.cache_data
def compute_colorgram_pallete_from_bytes(img_bytes, n_colors=3):
    """
    Extrae paleta con colorgram a partir de bytes y cachea.
    Devuelve lista de tuplas RGB (int,int,int).
    """
    try:
        # colorgram can read file-like objects too: use BytesIO
        buffer = io.BytesIO(img_bytes)
        colors = colorgram.extract(buffer, n_colors)
        pal = [(int(c.rgb.r), int(c.rgb.g), int(c.rgb.b)) for c in colors]
        return pal
    except Exception:
        return []

def rgb_to_lab(rgb):
    """Convierte color RGB (tuple) a Lab (float array) usando OpenCV."""
    arr = np.uint8([[[rgb[0], rgb[1], rgb[2]]]])  # shape (1,1,3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    return lab[0,0].astype(float)  # retorna (L,a,b)

def lab_delta(c1_lab, c2_lab):
    """Distancia euclidiana simple en Lab (proxy ΔE)."""
    return float(np.linalg.norm(np.array(c1_lab) - np.array(c2_lab)))

def create_color_bar(colores, ancho=300, alto=30):
    """Crea una barra PIL con los colores listados (RGB tuples)."""
    if not colores:
        return None
    barra = Image.new("RGB", (ancho, alto), color=(255,255,255))
    draw = ImageDraw.Draw(barra)
    n = len(colores)
    w = max(1, ancho // n)
    for i, c in enumerate(colores):
        draw.rectangle([i*w, 0, (i+1)*w, alto], fill=c)
    return barra

def create_gradient_overview(colores_seq, ancho_por_color=30, alto=50):
    """Crea una imagen numpy con tiras de los colores secuenciales (para la barra horizontal)."""
    if not colores_seq:
        return None
    n = len(colores_seq)
    img = np.zeros((alto, n*ancho_por_color, 3), dtype=np.uint8)
    for i, c in enumerate(colores_seq):
        img[:, i*ancho_por_color:(i+1)*ancho_por_color, :] = c
    return img

# ---------------------------
# UPLOADER: cargar pero NO procesar automáticamente
# - Al cargar, guardamos versiones reducidas (thumbnail) en session_state
# ---------------------------
uploaded_files = st.file_uploader("📁 Subí tus fotos (JPG/PNG). Ajustá sliders antes y presioná uno de los botones de organización.", accept_multiple_files=True, type=["jpg","jpeg","png"])

# inicializar session_state contenedores
if "images_reduced" not in st.session_state:
    st.session_state["images_reduced"] = []  # list of dicts: {name, bytes, pil}
if "ordered" not in st.session_state:
    st.session_state["ordered"] = []  # final ordered list of dicts

# Si hay nuevos archivos, re-cargar reduced versions (pero no ordenar autom.)
if uploaded_files:
    # Reseteamos listas (para evitar mezclar con previas)
    st.session_state["images_reduced"] = []
    for f in uploaded_files:
        try:
            # Leemos bytes crudos (para cache)
            raw = f.read()
            pil = Image.open(io.BytesIO(raw))
            # ======== REDUCCIÓN ANTES DE CARGAR =========
            # draft + thumbnail para evitar explotar memoria (Pillow)
            try:
                pil.draft("RGB", (reduccion, reduccion))
            except Exception:
                pass  # draft puede no soportar algunos formatos, no crítico
            pil = pil.convert("RGB")
            pil.thumbnail((reduccion, reduccion))  # reduce manteniendo proporción
            # guardamos bytes reducidos (para cache) y PIL reducido para preview
            b = pil_to_bytes(pil, fmt="JPEG", quality=85)
            st.session_state["images_reduced"].append({
                "name": f.name,
                "bytes": b,
                "pil": pil
            })
        except Exception as e:
            st.error(f"Error cargando {f.name}: {e}")

    st.success(f"{len(st.session_state['images_reduced'])} imágenes preparadas para análisis (reducidas).")

# ---------------------------
# BOTONES PRINCIPALES (interfaz)
# - botón Clásico (color promedio + lum)
# - botón Paleta (colorgram pipeline: lum -> paleta)
# - botón Reorganizar que reaplica parámetros si ya hay imágenes cargadas
# ---------------------------
col1, col2, col3 = st.columns([1,1,1])

with col1:
    btn_classic = st.button("🎨 Organizar: Color promedio + Luminosidad (clásico)")
with col2:
    btn_palette = st.button("🌈 Organizar: Paletas dominantes (colorgram) + Lum")
with col3:
    btn_reorder = st.button("🔁 Reorganizar (usar parámetros actuales)")

# If user clicks any button or previously ordered and wants reorder, compute ordering
def compute_classic_order(images_info):
    """
    Orden clásico:
      - para cada imagen: color promedio (RGB) y luminosidad
      - combinar por pesos (peso_color, peso_lum)
      - ordenar ascendente por score
    images_info: list of dicts with keys name, bytes, pil
    returns: ordered list of dicts with extra keys
    """
    results = []
    for it in images_info:
        # get bytes for cacheable funcs
        b = it["bytes"]
        # luminosity (cached)
        lum = compute_luminosity_from_bytes(b, use_bn=modo_bn)
        # compute color average on the reduced PIL (fast)
        try:
            arr = np.array(it["pil"]).reshape(-1,3).astype(float)
            color_mean = np.mean(arr, axis=0)
        except Exception:
            color_mean = np.array([0.,0.,0.])
        # build score
        color_norm = np.linalg.norm(color_mean)
        score = peso_color * color_norm + peso_lum * lum
        results.append({
            "name": it["name"],
            "pil": it["pil"],
            "bytes": b,
            "lum": lum,
            "color_mean": tuple(map(int, color_mean)),
            "score": score
        })
    # sort by score ascending
    results_sorted = sorted(results, key=lambda x: x["score"])
    return results_sorted

def compute_palette_pipeline_order(images_info):
    """
    Pipeline por pasos:
     1) ordenar por luminosidad
     2) dentro de ese orden, aplicar agrupamiento/ajuste por paleta dominante (colorgram)
     3) luego suavizado greedy (nearest neighbor en Lab) para minimizar saltos
    """
    results = []
    # paso 1: extraer paleta y lum
    for it in images_info:
        b = it["bytes"]
        lum = compute_luminosity_from_bytes(b, use_bn=modo_bn)
        pal = compute_colorgram_pallete_from_bytes(b, n_colors=num_clusters)
        # color dominante representativo: promedio de la paleta
        if pal:
            dom = np.mean(np.array(pal), axis=0)
        else:
            # fallback a color medio
            arr = np.array(it["pil"]).reshape(-1,3).astype(float)
            dom = np.mean(arr, axis=0)
        results.append({
            "name": it["name"],
            "pil": it["pil"],
            "bytes": b,
            "lum": lum,
            "palette": pal,
            "dominant_rgb": tuple(map(int, dom))
        })
    # paso 2: ordenar por luminosidad asc
    results.sort(key=lambda x: x["lum"])
    # paso 3: dentro del orden, ajustar por color: construimos bloques similares (greedy)
    # convertimos dominant rgb a Lab para distancias
    for r in results:
        r["dominant_lab"] = rgb_to_lab(r["dominant_rgb"])
    # ahora suavizado greedy: empezamos por la secuencia ordenada por lum y reordenamos localmente
    ordered = []
    remaining = results.copy()
    if not remaining:
        return []
    ordered.append(remaining.pop(0))
    while remaining:
        last_lab = ordered[-1]["dominant_lab"]
        dists = [lab_delta(last_lab, x["dominant_lab"]) for x in remaining]
        idx = int(np.argmin(dists))
        # si la distancia es mayor al umbral (en Lab), podemos buscar la más cercana global o aceptarla de todas formas.
        ordered.append(remaining.pop(idx))
    # paso 4: (opcional) ajustar dentro de la secuencia según umbral_similitud agrupando cercanos
    # devolvemos ordered con palette info
    return ordered

# Ejecutar acciones según botones
try:
    if btn_classic and st.session_state.get("images_reduced"):
        with st.spinner("Organizando (método clásico)..."):
            st.session_state["ordered"] = compute_classic_order(st.session_state["images_reduced"])
            if invertir:
                st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
        st.success("Organización clásica completada.")

    if btn_palette and st.session_state.get("images_reduced"):
        with st.spinner("Organizando (pipeline paletas + luminosidad)..."):
            st.session_state["ordered"] = compute_palette_pipeline_order(st.session_state["images_reduced"])
            if invertir:
                st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
        st.success("Organización por paletas completada.")

    # Reordenar reaplica el método anterior con parámetros actuales:
    if btn_reorder and st.session_state.get("images_reduced"):
        # detectamos qué método usar según si last ordered items have 'score' (classico) or 'palette' (paleta)
        prev = st.session_state.get("ordered", [])
        if prev and "score" in prev[0]:
            # re-aplicar clásico
            with st.spinner("Reaplicando clasificación clásica con parámetros actuales..."):
                st.session_state["ordered"] = compute_classic_order(st.session_state["images_reduced"])
                if invertir:
                    st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
            st.success("Reorganizado (clásico).")
        else:
            # re-aplicar paleta pipeline
            with st.spinner("Reaplicando pipeline de paletas con parámetros actuales..."):
                st.session_state["ordered"] = compute_palette_pipeline_order(st.session_state["images_reduced"])
                if invertir:
                    st.session_state["ordered"] = list(reversed(st.session_state["ordered"]))
            st.success("Reorganizado (paletas).")
except MemoryError as e:
    st.error("MemoryError durante el procesamiento. Baja el valor de 'reducción' o subí menos imágenes. " + str(e))
except Exception as e:
    st.error(f"Error inesperado al organizar: {e}")

# ---------------------------
# Si hay orden, mostramos gradiente, preview y PDF
# ---------------------------
if st.session_state.get("ordered"):
    ordered = st.session_state["ordered"]

    # Extraer colores representativos para la barra de gradiente.
    # Para clásico: usaremos 'color_mean' si existe; para paletas, el 'dominant_rgb'.
    colores_seq = []
    for item in ordered:
        if "color_mean" in item:
            colores_seq.append(tuple(map(int, item["color_mean"])))
        elif "dominant_rgb" in item:
            colores_seq.append(tuple(map(int, item["dominant_rgb"])))
        elif "palette" in item and item["palette"]:
            colores_seq.append(tuple(map(int, item["palette"][0])))
        else:
            colores_seq.append((200,200,200))

    # Mostrar barra horizontal de gradiente
    st.subheader("🎨 Gradiente cromático (vista previa)")
    grad = create_gradient_overview(colores_seq, ancho_por_color=30, alto=50)
    if grad is not None:
        st.image(grad, use_column_width=True)

    # Previsualización en cuadrícula 3x3 (feed)
    st.subheader("🖼️ Previsualización (feed 3x3)")
    n = len(ordered)
    # Mostrar en filas de 3
    for i in range(0, n, 3):
        cols = st.columns(3)
        for j, col in enumerate(cols):
            idx = i + j
            if idx < n:
                item = ordered[idx]
                # item puede tener distintas claves según método
                pil_img = item.get("pil") or item.get("pil")
                caption = item.get("name", f"{idx+1}")
                with col:
                    st.image(pil_img, caption=caption, use_column_width=True)
                    # barra de color bajo cada imagen (usamos palette if available)
                    if "palette" in item and item["palette"]:
                        barra = create_color_bar(item["palette"], ancho=300, alto=30)
                        st.image(barra, use_column_width=True)
                    elif "color_mean" in item:
                        barra = create_color_bar([tuple(map(int, item["color_mean"]))], ancho=300, alto=30)
                        st.image(barra, use_container_width=True)
                    else:
                        # fallback: color gris
                        barra = create_color_bar([(200,200,200)], ancho=300, alto=30)
                        st.image(barra, use_container_width=True)

    # ---------------------------
    # EXPORTAR A PDF (feed 3x3) con barras de color
    # ---------------------------
    st.subheader("📄 Exportar")
    col_pdf1, col_pdf2 = st.columns([1,1])
    with col_pdf1:
        if st.button("🧾 Generar PDF (feed 3x3)"):
            try:
                with st.spinner("Generando PDF (esto puede tardar unos segundos)..."):
                    # Creamos PDF temporal
                    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    pdf = FPDF(unit="mm", format="A4")
                    pdf.set_auto_page_break(False)
                    page_w, page_h = 210, 297
                    margin = 10
                    cell_w = (page_w - 2 * margin) / 3.0
                    cell_h = cell_w  # imagen cuadrada

                    # iterar bloques de 9
                    for i in range(0, len(ordered), 9):
                        pdf.add_page()
                        block = ordered[i:i+9]
                        for idx, item in enumerate(block):
                            r = idx // 3
                            c = idx % 3
                            x_mm = margin + c * cell_w
                            y_mm = margin + r * (cell_h + 15)  # dejar espacio para barra color debajo
                            # preparar imagen temporal en tamaño razonable
                            pil_img = item.get("pil")
                            if pil_img is None:
                                pil_img = Image.open(io.BytesIO(item["bytes"])) if item.get("bytes") else Image.new("RGB", (300,300),(240,240,240))
                            # reducir antes de guardar
                            im_copy = pil_img.copy()
                            im_copy.thumbnail((int(cell_w*10), int(cell_h*10)))
                            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                            im_copy.save(tmp_img.name, "JPEG", quality=70)
                            # draw image
                            pdf.image(tmp_img.name, x_mm, page_h - y_mm - cell_h, w=cell_w, h=cell_h)
                            # draw color bar under it if exists
                            bar_h_mm = 8
                            bar_w_mm = cell_w
                            # build bar image
                            bar_img = None
                            if "palette" in item and item["palette"]:
                                bar_img = create_color_bar(item["palette"], ancho=300, alto=30)
                            elif "color_mean" in item:
                                bar_img = create_color_bar([tuple(map(int, item["color_mean"]))], ancho=300, alto=30)
                            if bar_img:
                                tmp_bar = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                                bar_img.save(tmp_bar.name, "JPEG", quality=75)
                                # position bar just below image
                                pdf.image(tmp_bar.name, x_mm, page_h - y_mm - cell_h + cell_h - bar_h_mm - 2, w=bar_w_mm, h=bar_h_mm)
                                try:
                                    os.remove(tmp_bar.name)
                                except:
                                    pass
                            # limpiar tmp_img file
                            try:
                                tmp_img.close()
                                os.remove(tmp_img.name)
                            except:
                                pass
                    pdf.output(tmp_pdf.name)
                    # enviar al usuario
                    with open(tmp_pdf.name, "rb") as f:
                        st.download_button("⬇️ Descargar PDF final", f, file_name="feed_instagram.pdf")
                    st.success("PDF generado ✔")
                    # cleanup
                    try:
                        tmp_pdf.close()
                        os.remove(tmp_pdf.name)
                    except:
                        pass
            except MemoryError:
                st.error("MemoryError: reduce 'reducción' o subí menos imágenes antes de generar el PDF.")
            except Exception as e:
                st.error(f"Error generando PDF: {e}")

    with col_pdf2:
        if st.button("🔄 Limpiar sesión (borrar imágenes cargadas)"):
            # limpiar sesión (no recargamos el script entero)
            st.session_state["images_reduced"] = []
            st.session_state["ordered"] = []
            st.success("Sesión reiniciada. Subí nuevas imágenes cuando quieras.")

else:
    st.info("Subí imágenes y elige un método de organización (botones arriba).")

# ---------------------------
# FIN
# ---------------------------
