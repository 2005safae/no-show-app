import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ---------------------------
# 1) Charger le modèle
# ---------------------------
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# Liste exacte des features utilisées lors du training
FEATURE_NAMES = ['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 
                 'Alcoholism', 'Handcap', 'SMS_received', 'WaitingDays', 
                 'Day of Week', 'IsWeekend', 'Age group_13-19', 'Age group_20-39', 
                 'Age group_40-59', 'Age group_60+']

# ---------------------------
# 2) Configuration de la page
# ---------------------------
st.title("🎯 Prédiction des Rendez-vous Médicaux")
st.write("Prédire combien de personnes vont assister aujourd'hui.")

uploaded_file = st.file_uploader("📄 Charger le fichier des rendez-vous d'aujourd'hui (CSV)", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("🔍 Aperçu des données")
    st.dataframe(data.head())

    # ---------------------------
    # 3) Prédire pour chaque patient
    # ---------------------------
    st.subheader("🤖 Prédictions du modèle")

    # Vérifier que toutes les colonnes nécessaires sont présentes
    missing_cols = [col for col in FEATURE_NAMES if col not in data.columns]
    
    if missing_cols:
        st.error(f"❌ Colonnes manquantes dans le fichier CSV: {missing_cols}")
        st.info("Assurez-vous que votre fichier CSV contient toutes ces colonnes avec les mêmes noms.")
    else:
        try:
            X = data[FEATURE_NAMES]
            predictions = model.predict(X)
            
            data["Prediction"] = predictions
            data["Prediction_Label"] = data["Prediction"].map({0: "Absent", 1: "Présent"})

            st.dataframe(data)

            # ---------------------------
            # 4) Résumer
            # ---------------------------
            total_rdvs = len(data)
            total_present = (predictions == 1).sum()
            total_absent = total_rdvs - total_present

            st.subheader("📊 Résumé du jour")
            col1, col2, col3 = st.columns(3)
            col1.metric("📅 Rendez-vous total", total_rdvs)
            col2.metric("✅ Présents (prédits)", int(total_present))
            col3.metric("❌ Absents (prédits)", int(total_absent))

            # ---------------------------
            # 5) Capacité maximale du docteur
            # ---------------------------
            st.subheader("⚙ Capacité du docteur")
            max_capacity = st.number_input("Nombre maximum de patients par jour", 
                                          min_value=1, max_value=100, value=20)

            free_slots = max_capacity - total_present

            if free_slots > 0:
                st.success(f"👍 Il reste *{int(free_slots)} places* aujourd'hui.")
            else:
                st.error("❌ Le planning est complet aujourd'hui.")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")

# ---------------------------
# 6) Ajouter un nouveau patient
# ---------------------------
st.markdown("---")
st.subheader("➕ Ajouter un nouveau patient")

with st.form("new_patient_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Âge", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Genre", ["Femme (F)", "Homme (M)"])
        scholarship = st.selectbox("Bénéficiaire d'aide sociale", ["Non", "Oui"])
        hypertension = st.selectbox("Hypertension", ["Non", "Oui"])
        diabetes = st.selectbox("Diabète", ["Non", "Oui"])
        alcoholism = st.selectbox("Alcoolisme", ["Non", "Oui"])
    
    with col2:
        handcap = st.number_input("Niveau de handicap", min_value=0, max_value=4, value=0)
        sms = st.selectbox("SMS de rappel reçu", ["Non", "Oui"])
        waiting_days = st.number_input("Jours d'attente avant le RDV", min_value=0, max_value=365, value=7)
        day_of_week = st.selectbox("Jour de la semaine du RDV", 
                                   ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"])

    submitted = st.form_submit_button("🔮 Prédire la présence", type="primary")

if submitted:
    # Mapper le jour de la semaine en numéro (0=Lundi, 6=Dimanche)
    day_mapping = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, 
                   "Vendredi": 4, "Samedi": 5, "Dimanche": 6}
    day_num = day_mapping[day_of_week]
    
    # Déterminer si c'est le weekend
    is_weekend = 1 if day_num >= 5 else 0
    
    # Déterminer le groupe d'âge et créer les variables one-hot
    age_group_13_19 = 1 if 13 <= age <= 19 else 0
    age_group_20_39 = 1 if 20 <= age <= 39 else 0
    age_group_40_59 = 1 if 40 <= age <= 59 else 0
    age_group_60_plus = 1 if age >= 60 else 0
    
    # Construire les données exactement comme pendant le training
    new_data = pd.DataFrame([{
        'Gender': 1 if "Homme" in gender else 0,
        'Age': age,
        'Scholarship': 1 if scholarship == "Oui" else 0,
        'Hipertension': 1 if hypertension == "Oui" else 0,
        'Diabetes': 1 if diabetes == "Oui" else 0,
        'Alcoholism': 1 if alcoholism == "Oui" else 0,
        'Handcap': handcap,
        'SMS_received': 1 if sms == "Oui" else 0,
        'WaitingDays': waiting_days,
        'Day of Week': day_num,
        'IsWeekend': is_weekend,
        'Age group_13-19': age_group_13_19,
        'Age group_20-39': age_group_20_39,
        'Age group_40-59': age_group_40_59,
        'Age group_60+': age_group_60_plus
    }])

    try:
        # Prédiction
        new_pred = model.predict(new_data)[0]
        proba = model.predict_proba(new_data)[0]

        st.markdown("---")
        st.header("📊 Résultat de la Prédiction")
        
        if new_pred == 1:
            st.success(f"✅ *Le patient viendra probablement* (Probabilité: {proba[1]*100:.1f}%)")
        else:
            st.warning(f"⚠ *Risque de NO-SHOW* (Probabilité d'absence: {proba[0]*100:.1f}%)")

        # Vérifier la capacité (si un fichier a été uploadé)
        if uploaded_file and 'free_slots' in locals():
            st.markdown("---")
            if free_slots > 0:
                st.info(f"👍 Un rendez-vous peut être donné ({int(free_slots)} places disponibles**).")
            else:
                st.error("❌ Impossible de donner un rendez-vous : planning complet.")
    
    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")
        st.info("Vérifiez que le modèle a été correctement entraîné avec les bonnes colonnes.")

st.markdown("---")
st.caption("🏥 Application de prédiction de No-Show | Modèle: Régression Logistique")
