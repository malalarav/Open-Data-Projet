import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import pydeck as pdk

# --- Configuration de la page ---
st.set_page_config(
    page_title="Dashboard Churn Telco",
    layout="wide"
)

# Chargement des données 
@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path, sep=';', encoding='utf-8')
    except FileNotFoundError:
        st.error(f"Erreur : Fichier '{path}' introuvable. Assurez-vous qu'il est dans le même répertoire.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur de chargement du fichier nettoyé. Vérifiez le séparateur (';') et l'encodage. Détail : {e}")
        st.stop()

    # Les conversions et nettoyages ont déjà été effectués, nous assurons juste les types
    # pour garantir la compatibilité avec les graphiques et le modèle.
    try:
        data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
        data['Monthly Charges'] = pd.to_numeric(data['Monthly Charges'], errors='coerce')
        data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
        # S'assurer que Senior Citizen est bien une chaîne ('Yes'/'No' après nettoyage)
        data['Senior Citizen'] = data['Senior Citizen'].astype(str)
        # Supprimer les éventuels NaN restants par sécurité
        data.dropna(subset=['Total Charges', 'Latitude', 'Longitude'], inplace=True)
    except Exception as e:
        st.error(f"Erreur de conversion de types après chargement : {e}")
        st.stop()
    
    return data

# Chargement du modèle 
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        return None

data = load_data("telco_churn_cleaned.csv") 
model = load_model("churn_model.pkl")

st.title(" Dashboard de Prédiction du Churn Client (IBM Telco)")
st.markdown("---")

# Création des onglets
tab_presentation, tab_analyse, tab_prediction, tab_carte = st.tabs([
    "0. Présentation et Contexte", 
    "1. Analyse des Comportements",
    "2. Prédiction Interactive",
    "3. Carte Géographique"
])

# PRÉSENTATION
with tab_presentation:
    st.header("Bienvenue : Contexte et Objectifs du Projet")
    st.info(
        "Ce tableau de bord est conçu pour analyser et prédire le taux de désabonnement "
        "(Churn) des clients de l'entreprise de télécommunications Telco (fictif)."
    )

    st.subheader("1. L'Enjeu du Churn")
    st.write(
        "Le **Churn** représente la perte de clients. Dans le secteur des télécommunications, "
        "comprendre et prévenir le Churn est crucial, car **acquérir un nouveau client coûte "
        "souvent beaucoup plus cher que de retenir un client existant**."
    )
    
    st.subheader("2. Les Données Utilisées")
    st.markdown("""
    * **Source :** Jeu de données **Telco Customer Churn** de l'entreprise IBM (version nettoyée).
    * **Échantillon :** **7032 observations** (clients) après nettoyage initial.
    * **Variables Clés :** Le jeu de données couvre quatre axes principaux :
        * **Démographie :** `Gender`, `Senior Citizen`, `Partner`, `Dependents`.
        * **Services :** `Internet Service`, `Phone Service`, `Tech Support`, `Online Security`.
        * **Finances :** `Monthly Charges`, `Total Charges`, `Contract`, `Payment Method`.
        * **Cible :** `Churn Label` (Oui/Non), `Churn Score`, et la **raison de départ** (`Churn Reason`).
    """)

    st.subheader("3. Objectifs du Dashboard")
    st.markdown("""
    Ce dashboard permet de :
    * **Visualiser** les tendances et les corrélations clés dans l'onglet **Analyse des Comportements**.
    * **Tester** l'impact de différents profils clients sur le risque de départ dans l'onglet **Prédiction Interactive**.
    * **Localiser** géographiquement les zones à fort risque de Churn dans l'onglet **Carte Géographique**.
    """)
    st.markdown("---")
    st.success("Commencez par explorer l'Analyse des Comportements pour identifier les tendances majeures.")


# Analyse 
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


# Prédiction
with tab_prediction:
    st.header("Prédiction de la Probabilité de Désabonnement")
    st.info("Utilisez les options dans la barre latérale (gauche) pour définir le profil d'un client et prédire sa probabilité de départ.")
    
    if model is None:
        st.error(
            "Erreur : Modèle 'churn_model.pkl' introuvable. "
            "Veuillez d'abord exécuter le script 'train_model.py'."
        )
    else:
        st.sidebar.header("Tester un Profil Client")
        
        tenure = st.sidebar.slider("Ancienneté (Mois)", 0, 72, 12)
        monthly_charges = st.sidebar.slider("Frais Mensuels", 18.0, 120.0, 70.0)
        
        min_total = float(monthly_charges * tenure)
        max_total = 9000.0
        default_total = min(max(min_total, 0.0), max_total) 
        total_charges = st.sidebar.slider("Frais Totaux", 0.0, max_total, default_total)

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
                    st.error("Risque de Churn ÉLEVÉ 🚨 : Des actions de rétention sont urgentes.")
                elif prediction_proba > 0.25:
                    st.warning("Risque de Churn MODÉRÉ ⚠️ : Surveillance et offres personnalisées recommandées.")
                else:
                    st.success("Risque de Churn FAIBLE ✅ : Client stable.")
            
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : Assurez-vous que le modèle est compatible et que les inputs correspondent. Détail: {e}")

# Carte Géographique
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
            map_style='light',
            tooltip=tooltip
        )
        
        # Afficher la carte dans Streamlit
        st.pydeck_chart(r, use_container_width=True)