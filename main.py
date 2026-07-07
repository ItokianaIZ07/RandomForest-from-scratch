import streamlit as st
from PIL import Image
import os
from datetime import datetime
import csv
from processing.process import DataProcessing
from model.forest import RandomForest
from model.tree import DecisionTree
import joblib

dataProcessing = DataProcessing()
model = joblib.load("./model/modele_foret.pkl")

st.set_page_config(
    page_title="Détection des maladies du maïs",
    page_icon="🌽",
    layout="centered"
)

st.title("🌽 Détection des maladies du maïs")

st.page_link(
    "pages/1_Historique.py",
    label="Voir l'historique des images",
    icon="📂"
)

st.divider()

st.write(
    "Sélectionnez une image d'une feuille de maïs puis cliquez sur **Prédire**."
)

DOSSIER_PREDICTION = "uploads"
HISTORIQUE = "historique.csv"

os.makedirs(DOSSIER_PREDICTION, exist_ok=True)

if not os.path.exists(HISTORIQUE):
    with open(HISTORIQUE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "diagnostic"])


def getPredictionResult(model, data):
    if isinstance(model, DecisionTree):
        prediction = model.predict(data)
        return 1 if prediction.count(1) > prediction.count(0) else 0

    elif isinstance(model, RandomForest):
        prediction = model.predict(data).values
        return 1 if len(prediction[prediction == 1]) > len(prediction[prediction == 0]) else 0


def predire_maladie(img):

    print("Image reçue :", img)

    data = dataProcessing.numeriseDonnee(img)
    prediction = getPredictionResult(model, data)

    return "Cette plante est malade" if prediction == 1 else "Cette plante est saine"


uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Image sélectionnée",
        use_container_width=True
    )

    if st.button("Prédire"):

        extension = uploaded_file.name.split(".")[-1]

        nom_image = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "."
            + extension
        )

        chemin_image = os.path.join(
            DOSSIER_PREDICTION,
            nom_image
        )

        image.save(chemin_image)

        resultat = predire_maladie(chemin_image)

        with open(HISTORIQUE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([nom_image, resultat])

        st.divider()

        st.subheader("Résultat")

        if "malade" in resultat.lower():
            st.error(resultat)

        elif "saine" in resultat.lower():
            st.success(resultat)