import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import colorgram
import io
from fpdf import FPDF

# =====================================
# ⚙️ CONFIGURACIÓN INICIAL DE LA APP
# =====================================
st.set_page_config(page_title="🎨 Organizador Fine Art", layout="wide")
st.title("🎨 Organizador Fine Art por Color y Luminosidad")
st.write("Ajustá los parámetros, subí tus fotos y obtené una organización estética por gradiente de color y luz.")

# =====================================
# 🎚️ PANEL LATERAL DE CONTROLES
# =====================================
st.sidebar.header("⚙️ Ajustes del Organizador")

# Parámetros de análisis
num_clusters = st.sidebar.slider("Número de clusters (para agrupar tonos)", 2, 10, 4)
reduccion = st.sidebar.slider("Reducción de tamaño para análisis", 100, 1000, 300)
peso_color = st.sidebar.slider("Peso del color 🎨", 0.0, 1.0, 0.5)
peso_luz = st.sidebar.slider("Peso de la luminosidad 💡", 0.0, 1.0, 0.5)
umbral_color = st.sidebar.slider("Nivel de similitud de color", 10, 300, 100)

# Modo general
modo = st.sidebar.selectbox(
    "Modo de organización inicial",
    ["Luminosidad primero", "Color primero", "Mixto (ambos equilibrados)"]
)

# =====================================
# 🧩 FUNCIONES AUXILIARES
# =====================================

def calcular_luminosidad(img):
    """Calcula la luminosidad promedio de la imagen."""
    img = img.convert("RGB")
    np_img = np.array(img.resize((reduccion, reduccion)))
    r, g, b = np_img[:,:,0], np_img[:,:,1], np_img[:,:,2]
    luminancia = 0.2126*r + 0.7152*g + 0.0722*b
    return np.mean(luminancia)

def obtener_colores_dominantes(img, cantidad=3):
    """Extrae los colores dominantes con colorgram."""
    colores = colorgram.extract(img, cantidad)
    return [(c.rgb.r, c.rgb.g, c.rgb.b) for c in colores]

def similitud_paleta(p1, p2):
    """Distancia promedio entre dos paletas RGB."""
    if not p1 or not p2:
        return float("inf")
    distancias = [np.linalg.norm(np.array(c1)-np.array(c2)) for c1 in p1 for c2 in p2]
    return np.mean(distancias)

def organizar_por_luminosidad(lista_imgs):
    """Ordena por luminosidad."""
    return sorted(lista_imgs, key=lambda x: x["luminosidad"])

def agrupar_por_color(lista_imgs, umbral):
    """Agrupa por similitud de paleta."""
    agrupadas = [lista_imgs[0]]
    for img in lista_imgs[1:]:
        ultima = agrupadas[-1]
        dist = similitud_paleta(ultima["colores"], img["colores"])
        if dist < umbral:
            agrupadas.append(img)
        else:
            agrupadas.append(img)
    return agrupadas

def crear_barra_colores(colores, ancho=300, alto=40):
    """Crea una barra visual de colores dominantes."""
    barra = Image.new("RGB", (ancho, alto))
    draw = ImageDraw.Draw(barra)
    ancho_color = ancho // len(colores)
    for i, c in enumerate(colores):
        draw.rectangle([i*ancho_color, 0, (i+1)*ancho_color, alto], fill=c)
    return barra

def crear_pdf(imagenes):
    """Genera PDF con las fotos organizadas."""
    pdf = FPDF()
    for img_info in imagenes:
        img = img_info["imagen"].resize((600, 600))
        temp = io.BytesIO()
        img.save(temp, format="JPEG")
        pdf.add_page()
        pdf.image(temp, x=10, y=10, w=180)
    buffer = io.BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# =====================================
# 📂 SUBIDA DE ARCHIVOS
# =====================================
uploaded_files = st.file_uploader("📸 Subí tus fotos (JPG o PNG)", accept_multiple_files=True, type=["jpg", "jpeg", "png"])

# =====================================
# 🔄 PROCESAMIENTO PRINCIPAL
# =====================================
if uploaded_files:
    st.success(f"Se subieron {len(uploaded_files)} fotos correctamente ✅")
    fotos_data = []

    for file in uploaded_files:
        img = Image.open(file)
        # Reducimos tamaño temporal para optimizar recursos
        img_reducida = img.resize((reduccion, reduccion))
        luminosidad = calcular_luminosidad(img_reducida)
        colores = obtener_colores_dominantes(img_reducida, cantidad=num_clusters)
        fotos_data.append({
            "nombre": file.name,
            "imagen": img,
            "luminosidad": luminosidad,
            "colores": colores
        })

    # Botón para ejecutar la organización
    if st.button("🪄 Organizar según configuración"):
        st.info(f"Procesando con modo '{modo}'...")

        # Modo luminosidad primero
        if modo == "Luminosidad primero":
            fotos_data = organizar_por_luminosidad(fotos_data)
            fotos_data = agrupar_por_color(fotos_data, umbral_color)

        # Modo color primero
        elif modo == "Color primero":
            fotos_data.sort(key=lambda x: np.mean([np.mean(c) for c in x["colores"]]))
            fotos_data = agrupar_por_color(fotos_data, umbral_color)

        # Modo mixto: equilibrio entre color y luminosidad
        else:
            fotos_data.sort(key=lambda x: x["luminosidad"] * peso_luz + np.mean([np.mean(c) for c in x["colores"]]) * peso_color)
            fotos_data = agrupar_por_color(fotos_data, umbral_color)

        st.session_state["fotos_final"] = fotos_data
        st.success("✅ Organización completada.")

    # =====================================
    # 👀 VISTA PREVIA
    # =====================================
    if "fotos_final" in st.session_state:
        st.subheader("🔍 Vista previa del orden final")

        cols = st.columns(5)
        for i, foto in enumerate(st.session_state["fotos_final"]):
            with cols[i % 5]:
                st.image(foto["imagen"], caption=foto["nombre"], use_container_width=True)
                barra = crear_barra_colores(foto["colores"])
                st.image(barra, use_container_width=True)

        # Generar PDF
        if st.button("📄 Generar PDF final"):
            pdf_bytes = crear_pdf(st.session_state["fotos_final"])
            st.download_button(
                label="⬇️ Descargar PDF organizado",
                data=pdf_bytes,
                file_name="organizacion_fine_art.pdf",
                mime="application/pdf"
            )

else:
    st.warning("Subí tus imágenes para comenzar la organización.")
