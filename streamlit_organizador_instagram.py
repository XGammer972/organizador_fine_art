# ===============================================================
# 🖼️ ORGANIZADOR FINE ART PARA INSTAGRAM
# Autor: Naim + ChatGPT
# Versión optimizada (RAM reducida, caching, PDF opcional)
# ===============================================================

import streamlit as st
from PIL import Image
import numpy as np
import io
import colorgram
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile
import os

# ===============================================================
# 🧠 CONFIGURACIÓN INICIAL
# ===============================================================

# Evita advertencias de Pillow con imágenes grandes
Image.MAX_IMAGE_PIXELS = 20000000

st.set_page_config(page_title="Organizador Fine Art", layout="wide")

st.title("🎨 Organizador Estético para Instagram")
st.write("Subí tus fotos y ajustá los parámetros para ordenarlas por **color y luminosidad** de manera armoniosa.")

# ===============================================================
# ⚙️ PANEL DE SLIDERS (visible desde el inicio)
# ===============================================================

st.sidebar.header("Ajustes de organización")

# Parámetros ajustables por el usuario
num_clusters = st.sidebar.slider("Número de clusters de color", 2, 10, 4)
reduccion = st.sidebar.slider("Tamaño de reducción para análisis", 200, 800, 400, step=100)
peso_color = st.sidebar.slider("Peso del color", 0.0, 1.0, 0.5, step=0.1)
peso_luminosidad = st.sidebar.slider("Peso de la luminosidad", 0.0, 1.0, 0.5, step=0.1)
modo_bn = st.sidebar.checkbox("🔲 Usar blanco y negro para suavizar gradiente", value=False)

# ===============================================================
# 📤 SUBIDA DE ARCHIVOS
# ===============================================================

uploaded_files = st.file_uploader(
    "Subí tus fotos (máximo 150 MB en total)", 
    accept_multiple_files=True, 
    type=["png", "jpg", "jpeg"]
)

# ===============================================================
# 🔧 FUNCIONES CON CACHE Y OPTIMIZACIONES DE MEMORIA
# ===============================================================

@st.cache_data
def calcular_luminosidad(img):
    """Convierte a escala de grises y obtiene promedio de luminosidad."""
    if img.mode != "L":
        img = img.convert("L")
    return np.mean(np.array(img))

@st.cache_data
def obtener_colores_dominantes(img, cantidad=3):
    """Extrae colores dominantes usando colorgram (ya optimizado)."""
    colores = colorgram.extract(img, cantidad)
    return [(c.rgb.r, c.rgb.g, c.rgb.b) for c in colores]

def reducir_imagen(img, max_size=(800, 800)):
    """
    Reduce el tamaño de la imagen antes de procesarla para ahorrar memoria.
    Se usa 'draft' + 'thumbnail' (no deforma la imagen).
    """
    img.draft("RGB", max_size)
    img = img.convert("RGB")
    img.thumbnail(max_size)
    return img

# ===============================================================
# 🔄 PROCESAMIENTO PRINCIPAL
# ===============================================================

if uploaded_files:
    st.write("📸 **Procesando tus imágenes...** Esto puede tardar unos segundos.")
    
    imagenes_info = []

    for file in uploaded_files:
        try:
            img = Image.open(file)

            # ⚙️ Reducción temprana para ahorrar RAM
            img = reducir_imagen(img, (reduccion, reduccion))

            # Si se seleccionó el modo blanco y negro
            img_proc = img.convert("L") if modo_bn else img

            # 🧮 Cálculo de luminosidad y colores dominantes
            lum = calcular_luminosidad(img_proc)
            colores = obtener_colores_dominantes(img_proc, num_clusters)

            # Guardamos la información para ordenar después
            imagenes_info.append((file.name, img, lum, colores))

            # 🧠 Libera memoria de la imagen procesada
            del img_proc

        except Exception as e:
            st.error(f"Error al procesar {file.name}: {e}")

    # ===============================================================
    # 🎨 ORGANIZACIÓN DE LAS IMÁGENES
    # ===============================================================

    if imagenes_info:
        st.write("🔢 **Organizando según parámetros seleccionados...**")

        # Calculamos el color medio ponderado de cada imagen
        datos_orden = []
        for nombre, img, lum, colores in imagenes_info:
            if len(colores) > 0:
                color_medio = np.mean(colores, axis=0)
                valor_orden = (
                    peso_color * np.linalg.norm(color_medio) +
                    peso_luminosidad * lum
                )
                datos_orden.append((nombre, img, valor_orden))

        # Ordena según el valor combinado (color + luminosidad)
        imagenes_ordenadas = sorted(datos_orden, key=lambda x: x[2])

        # ===============================================================
        # 🖼️ PREVISUALIZACIÓN EN CUADRÍCULA 3x3 (feed estilo Instagram)
        # ===============================================================

        st.subheader("🧩 Previsualización del orden final")

        columnas = st.columns(3)
        for i, (nombre, img, _) in enumerate(imagenes_ordenadas):
            with columnas[i % 3]:
                st.image(img, caption=nombre, use_container_width=True)

        # ===============================================================
        # 📄 GENERAR PDF EN FORMATO FEED
        # ===============================================================

        if st.button("📥 Generar PDF del orden final"):
            pdf_path = os.path.join(tempfile.gettempdir(), "feed_instagram.pdf")
            c = canvas.Canvas(pdf_path, pagesize=A4)
            ancho, alto = A4
            margen = 30
            tamaño_img = (ancho - 4 * margen) / 3

            x = margen
            y = alto - tamaño_img - margen
            contador = 0

            for _, img, _ in imagenes_ordenadas:
                # Reducimos la imagen antes de exportar para optimizar PDF
                img_reducida = img.copy()
                img_reducida.thumbnail((600, 600))
                temp_path = os.path.join(tempfile.gettempdir(), "temp_img.jpg")
                img_reducida.save(temp_path, "JPEG", quality=70)
                c.drawImage(temp_path, x, y, tamaño_img, tamaño_img)

                x += tamaño_img + margen
                contador += 1

                # Cada 3 imágenes cambiamos de fila
                if contador % 3 == 0:
                    x = margen
                    y -= tamaño_img + margen
                    if y < margen:
                        c.showPage()
                        y = alto - tamaño_img - margen

            c.save()
            st.success("✅ PDF generado correctamente.")
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Descargar PDF", f, file_name="feed_instagram.pdf")

else:
    st.info("⬆️ Esperando que subas tus imágenes para comenzar...")
