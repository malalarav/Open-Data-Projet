import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import pydeck as pdk  # NOUVEL IMPORT pour la carte

# --- Configuration de la page ---
st.set_page_config(
    page_title="Dashboard Churn Telco",
    layout="wide"
)

# --- Chargement et Nettoyage des données ---
@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path, sep=';', encoding='latin1')
    except Exception as e:
        data = pd.read_csv(path, sep=';')

    # CORRECTION DES DÉCIMAUX (virgule -> point)
    cols_to_convert = ['Total Charges', 'Monthly Charges', 'Latitude', 'Longitude']
    for col in cols_to_convert:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '.')
        else:
            st.error(f"Colonne attendue '{col}' introuvable lors du chargement.")
            st.stop()
            
    # Conversion numérique
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
    data['Monthly Charges'] = pd.to_numeric(data['Monthly Charges'], errors='coerce')
    data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
    data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')

    data = data.dropna(subset=['Total Charges', 'Latitude', 'Longitude'])
    
    data['Senior Citizen'] = data['Senior Citizen'].astype(str)
    
    return data

# --- Chargement du modèle ---
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None

# Charger les données et le modèle
data = load_data("Telco_customer_churn.csv")
model = load_model("churn_model.pkl")

# --- Titre principal du Dashboard ---
st.title(" Dashboard de Prédiction du Churn Client (IBM Telco)")

# --- Création des onglets ---
tab_analyse, tab_prediction, tab_carte = st.tabs([
    "1. Analyse des Comportements",
    "2. Prédiction Interactive",
    "3. Carte Géographique"
])

# --- Contenu de l'onglet 1 : Analyse (Identique) ---
with tab_analyse:
    st.header("Analyse Exploratoire des Comportements Clients")
    st.info("Objectif : Comprendre les facteurs qui influencent le désabonnement (Churn).")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Impact du Type de Contrat")
        fig_contract = px.histogram(data, x='Contract', color='Churn Label', barmode='group',
                                    title="Taux de Churn par Type de Contrat",
                                    color_discrete_map={'No':'#636EFA', 'Yes':'#EF553B'})
        st.plotly_chart(fig_contract, use_container_width=True)
    with col2:
        st.subheader("Impact du Service Internet")
        fig_internet = px.histogram(data, x='Internet Service', color='Churn Label', barmode='group',
                                    title="Taux de Churn par Service Internet",
                                    color_discrete_map={'No':'#636EFA', 'Yes':'#EF553B'})
        st.plotly_chart(fig_internet, use_container_width=True)
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Distribution de l'Ancienneté (Tenure)")
        fig_tenure = px.box(data, x='Churn Label', y='Tenure Months', color='Churn Label',
                            title="Distribution de l'Ancienneté vs. Churn",
                            color_discrete_map={'No':'#636EFA', 'Yes':'#EF553B'})
        st.plotly_chart(fig_tenure, use_container_width=True)
    with col4:
        st.subheader("Distribution des Frais Mensuels")
        fig_monthly = px.box(data, x='Churn Label', y='Monthly Charges', color='Churn Label',
                             title="Distribution des Frais Mensuels vs. Churn",
                             color_discrete_map={'No':'#636EFA', 'Yes':'#EF553B'})
        st.plotly_chart(fig_monthly, use_container_width=True)


# --- Contenu de l'onglet 2 : Prédiction (Identique) ---
with tab_prediction:
    st.header("Prédiction de la Probabilité de Désabonnement")
    
    if model is None:
        st.error(
            "Erreur : Modèle 'churn_model.pkl' introuvable. "
            "Veuillez d'abord exécuter le script 'train_model.py'."
        )
    else:
        st.info(
            "Utilisez les options dans la barre latérale (gauche) "
            "pour définir le profil d'un client et prédire sa probabilité de départ."
        )

        # --- Barre Latérale (Sidebar) pour les inputs ---
        st.sidebar.header("Tester un Profil Client")
        
        tenure = st.sidebar.slider("Ancienneté (Mois)", 0, 72, 12)
        monthly_charges = st.sidebar.slider("Frais Mensuels", 18.0, 120.0, 70.0)
        total_charges = st.sidebar.slider("Frais Totaux", 0.0, 9000.0, float(monthly_charges * tenure))
        contract = st.sidebar.selectbox("Type de Contrat", data['Contract'].unique())
        internet_service = st.sidebar.selectbox("Service Internet", data['Internet Service'].unique())
        payment_method = st.sidebar.selectbox("Méthode de Paiement", data['Payment Method'].unique())
        gender = st.sidebar.selectbox("Genre", data['Gender'].unique())
        senior_citizen = st.sidebar.selectbox("Senior Citizen", data['Senior Citizen'].unique())
        partner = st.sidebar.selectbox("Partenaire", data['Partner'].unique())
        dependents = st.sidebar.selectbox("Personnes à charge", data['Dependents'].unique())
        phone_service = st.sidebar.selectbox("Service Téléphonique", data['Phone Service'].unique())
        online_security = st.sidebar.selectbox("Sécurité en ligne", data['Online Security'].unique())
        tech_support = st.sidebar.selectbox("Support Technique", data['Tech Support'].unique())
        
        if st.sidebar.button("🔮 Prédire le Churn"):
            input_data = pd.DataFrame({
                'Tenure Months': [tenure], 'Monthly Charges': [monthly_charges],
                'Total Charges': [total_charges], 'Gender': [gender],
                'Senior Citizen': [senior_citizen], 'Partner': [partner],
                'Dependents': [dependents], 'Phone Service': [phone_service],
                'Internet Service': [internet_service], 'Contract': [contract],
                'Payment Method': [payment_method], 'Online Security': [online_security],
                'Tech Support': [tech_support]
            })
            
            try:
                prediction_proba = model.predict_proba(input_data)[0][1]
                st.subheader("Résultat de la Prédiction")
                st.metric(label="Probabilité de Désabonnement (Churn)", 
                          value=f"{prediction_proba * 100:.2f} %")
                st.progress(prediction_proba)
                
                if prediction_proba > 0.5:
                    st.error("Risque de Churn ÉLEVÉ 🚨")
                elif prediction_proba > 0.25:
                    st.warning("Risque de Churn MODÉRÉ ⚠️")
                else:
                    st.success("Risque de Churn FAIBLE ✅")
            
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")

# --- Contenu de l'onglet 3 : Carte (CORRIGÉ) ---
with tab_carte:
    st.header("Carte Géographique du Churn")
    st.info("Visualisation des clients ayant résilié (Rouge) vs. les clients actifs (Vert).")

    # Filtre interactif par ville
    top_cities = data['City'].value_counts().nlargest(50).index
    selected_city = st.selectbox("Sélectionnez une ville (Top 50)", top_cities)

    # Filtrer les données pour la carte
    map_data = data[data['City'] == selected_city].copy()

    if map_data.empty:
        st.warning(f"Aucune donnée disponible pour la ville : {selected_city}")
    else:
        # --- LA CORRECTION EST ICI ---
        # Créer une colonne de couleur en utilisant .apply()
        map_data['color'] = map_data['Churn Label'].apply(
            lambda x: [255, 0, 0, 160] if x == 'Yes' else [0, 128, 0, 160]
        )
        
        # Définir le point central de la carte
        mid_lat = map_data['Latitude'].mean()
        mid_lon = map_data['Longitude'].mean()

        # Configurer la vue initiale de la carte
        view_state = pdk.ViewState(
            latitude=mid_lat,
            longitude=mid_lon,
            zoom=11,
            pitch=50
        )

        # Définir la couche (layer) de points
        layer = pdk.Layer(
            'ScatterplotLayer',
            data=map_data,
            get_position='[Longitude, Latitude]',
            get_fill_color='color', # Utilise notre colonne 'color'
            get_radius=50,
            pickable=True,
            auto_highlight=True
        )

        # Définir l'info-bulle (tooltip) au survol
        tooltip = {
            "html": "<b>Client:</b> {CustomerID}<br/>"
                    "<b>Frais Mensuels:</b> {Monthly Charges} $<br/>"
                    "<b>Churn:</b> {Churn Label}",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        # Créer la carte Pydeck
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style='mapbox://styles/mapbox/light-v9',
            tooltip=tooltip
        )
        
        # Afficher la carte dans Streamlit
        st.pydeck_chart(r, use_container_width=True)