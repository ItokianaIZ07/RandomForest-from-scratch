import streamlit as st
from PIL import Image
import pandas as pd
import os

st.set_page_config(
    page_title="Historique",
    page_icon="📂"
)

st.title("📂 Historique des prédictions")

st.page_link(
    "main.py",
    label="⬅ Retour à la prédiction",
    icon="🏠"
)

st.divider()

DOSSIER = "uploads"
HISTORIQUE = "historique.csv"

if not os.path.exists(HISTORIQUE):
    st.warning("Aucun historique trouvé.")
    st.stop()

if not os.path.exists(DOSSIER):
    st.warning("Le dossier des images n'existe pas.")
    st.stop()

historique = pd.read_csv(HISTORIQUE)

if historique.empty:
    st.info("Aucune prédiction enregistrée.")
    st.stop()

colonnes = st.columns(3)

for i, ligne in historique.iterrows():

    nom_image = ligne["image"]
    diagnostic = ligne["diagnostic"]

    chemin = os.path.join(DOSSIER, nom_image)

    if not os.path.exists(chemin):
        continue

    image = Image.open(chemin)

    with colonnes[i % 3]:

        st.image(
            image,
            use_container_width=True
        )

        st.caption(nom_image)

        if "malade" in diagnostic.lower():
            st.error(diagnostic)
        else:
            st.success(diagnostic)