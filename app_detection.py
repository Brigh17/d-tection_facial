import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

st.set_page_config(page_title="Détection Faciale", layout="centered")

# Créer le dossier pour les captures si nécessaire
os.makedirs("captures", exist_ok=True)

st.title("📸 Détection Faciale + Galerie avec Suppression Sécurisée")

st.markdown("""
Cette application vous permet de :
- 📷 Capturer des visages avec votre webcam.
- 💾 Enregistrer les images capturées avec un nom.
- 🖼️ Visualiser une galerie des images.
- 🗑️ Supprimer plusieurs images avec confirmation.

### 🧭 Instructions :
1. Activez la webcam et capturez une image.
2. Donnez un nom à l’image avant de l’enregistrer.
3. Supprimez des images en les sélectionnant et en confirmant.
""")

# Paramètres de détection
scale_factor = st.slider("🔍 scaleFactor", 1.05, 1.5, 1.1, 0.05)
min_neighbors = st.slider("👥 minNeighbors", 3, 10, 5)
color = st.color_picker("🎨 Couleur du rectangle", "#00FF00")
bgr_color = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))  # hex vers BGR

captured_frame = None

# Capture webcam
if st.checkbox("📹 Activer la webcam et capturer une image"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Erreur : impossible d'accéder à la webcam.")
    else:
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Erreur lors de la capture de l'image.")
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

            captured_frame = frame.copy()
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="🖼️ Image capturée", channels="RGB")

# Enregistrement
if captured_frame is not None:
    filename_input = st.text_input("✍️ Nom du fichier", value=f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if st.button("💾 Enregistrer l'image"):
        path = os.path.join("captures", f"{filename_input}.png")
        cv2.imwrite(path, captured_frame)
        st.success(f"Image enregistrée : {path}")
        st.rerun()  # ✅ Correction ici

# Galerie
st.markdown("---")
st.header("🖼️ Galerie")

image_files = sorted([f for f in os.listdir("captures") if f.endswith(".png")], reverse=True)

if image_files:
    selected_images = st.multiselect("📌 Sélectionnez les images à supprimer :", image_files)

    if selected_images:
        confirm = st.checkbox("⚠️ Je confirme vouloir supprimer les images sélectionnées")
        if st.button("🗑️ Supprimer les images sélectionnées") and confirm:
            for img in selected_images:
                os.remove(os.path.join("captures", img))
            st.success(f"{len(selected_images)} image(s) supprimée(s).")
            st.rerun()  # ✅ Correction ici

    # Affichage en grille
    cols = st.columns(3)
    for idx, img_file in enumerate(image_files):
        img_path = os.path.join("captures", img_file)
        with cols[idx % 3]:
            st.image(img_path, caption=img_file, use_column_width=True)
else:
    st.info("Aucune image enregistrée.")
