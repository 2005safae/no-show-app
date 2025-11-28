import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# 1) Charger le modèle
# ---------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------------------
# 2) Charger le dataset du jour
# ---------------------------
st.title("🎯 Prédiction des Rendez-vous Médicaux")
st.write("Prédire combien de personnes vont assister aujourd’hui.")

uploaded_file = st.file_uploader("📄 Charger le fichier des rendez-vous d'aujourd'hui (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Aperçu des données")
    st.dataframe(data.head())

    # ---------------------------
    # 3) Prédire pour chaque patient
    # ---------------------------
    st.subheader("🤖 Prédictions du modèle")

    # IMPORTANT : garder uniquement les mêmes colonnes que lors du training
    FEATURES = [col for col in data.columns if col != "No-show"]  # adapter selon ton dataset

    X = data[FEATURES]

    predictions = model.predict(X)
    data["Prediction"] = predictions

    st.dataframe(data)

    # ---------------------------
    # 4) Résumer
    # ---------------------------
    total_rdvs = len(data)
    total_present = data["Prediction"].sum()
    total_absent = total_rdvs - total_present

    st.subheader("📊 Résumé du jour")
    st.write(f"*Rendez-vous total :* {total_rdvs}")
    st.write(f"*Personnes attendues (prédites) :* {total_present}")
    st.write(f"*Personnes absentes (prédites) :* {total_absent}")

    # ---------------------------
    # 5) Capacité maximale du docteur
    # ---------------------------
    st.subheader("⚙ Capacité du docteur")
    max_capacity = st.number_input("Nombre maximum de patients par jour", min_value=1, max_value=100, value=20)

    free_slots = max_capacity - total_present

    if free_slots > 0:
        st.success(f"👍 Il reste *{free_slots} places* aujourd’hui.")
    else:
        st.error("❌ Le planning est complet aujourd’hui.")

    # ---------------------------
    # 6) Ajouter un nouveau patient
    # ---------------------------
    st.subheader("➕ Ajouter un patient et vérifier si on peut lui donner un RDV")

    with st.form("new_patient_form"):
        age = st.number_input("Âge", min_value=0, max_value=120)
        gender = st.selectbox("Genre", ["M", "F"])
        sms = st.selectbox("SMS reçu ?", ["Oui", "Non"])
        hypertension = st.selectbox("Hypertension ?", ["Oui", "Non"])
        diabetes = st.selectbox("Diabète ?", ["Oui", "Non"])

        submitted = st.form_submit_button("Prédire la présence")

    if submitted:

        # Construit une ligne de données pour le modèle
        new_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "SMS_received": 1 if sms == "Oui" else 0,
            "Hypertension": 1 if hypertension == "Oui" else 0,
            "Diabetes": 1 if diabetes == "Oui" else 0
        }])

        # Encoder comme lors du training
        new_pred = model.predict(new_data)[0]

        if new_pred == 1:
            st.info("📌 Le modèle prédit : cette personne *viendra*.")
        else:
            st.warning("📌 Le modèle prédit : cette personne *ne viendra pas*.")

        # Vérifier la capacité
        if free_slots > 0:
            st.success("👍 Un rendez-vous peut être donné (place disponible).")
        else:
            st.error("❌ Impossible : planning complet.")
