import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np
import pydeck as pdk

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Churn Telco",
    layout="wide"
)

# --- Chargement des données ---
@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path, sep=';', encoding='utf-8')
    except FileNotFoundError:
        st.error(f"Erreur : Fichier '{path}' introuvable.")
        st.stop()
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        st.stop()

    try:
        # Conversions numériques
        data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
        data['Monthly Charges'] = pd.to_numeric(data['Monthly Charges'], errors='coerce')
        data['Latitude'] = pd.to_numeric(data['Latitude'], errors='coerce')
        data['Longitude'] = pd.to_numeric(data['Longitude'], errors='coerce')
        data['Senior Citizen'] = data['Senior Citizen'].astype(str)
        
        # Création d'une colonne binaire pour les calculs (1 = Churn, 0 = Non)
        data['Churn Binary'] = data['Churn Label'].apply(lambda x: 1 if x == 'Yes' else 0)
        
        data.dropna(subset=['Total Charges', 'Latitude', 'Longitude'], inplace=True)
    except Exception as e:
        st.error(f"Erreur de conversion : {e}")
        st.stop()
    
    return data

# --- Chargement du modèle ---
@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except:
        return None

# --- Initialisation ---
data = load_data("telco_churn_cleaned.csv") 
model = load_model("churn_model.pkl")

st.title("Dashboard de Prédiction du Churn Client (IBM Telco)")
st.markdown("---")

tab_presentation, tab_analyse, tab_prediction, tab_carte = st.tabs([
    "0. Présentation", 
    "1. Analyse Stratégique",
    "2. Prédiction Interactive",
    "3. Carte Géographique"
])

# --- Onglet 0 : Présentation ---
with tab_presentation:
    st.header("Contexte du Projet")
    st.info("Outil d'aide à la décision pour réduire le taux d'attrition (Churn).")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Le Problème")
        st.write("Le marché des télécoms est saturé. Acquérir un nouveau client coûte 5x plus cher que d'en garder un.")
    with col2:
        st.subheader("La Solution")
        st.write("Ce dashboard identifie les clients à risque, les raisons de leur départ et localise les zones critiques.")

# --- Onglet 1 : Analyse (Complète) ---
with tab_analyse:
    st.header("Analyse Exploratoire")
    
    # 1. KPIs
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    churn_rate = (data['Churn Label'].value_counts(normalize=True).get('Yes', 0) * 100).round(2)
    
    col_kpi1.metric("Taux de Churn Global", f"{churn_rate} %")
    col_kpi2.metric("Revenu Mensuel Moyen", f"{data['Monthly Charges'].mean():.2f} $")
    col_kpi3.metric("Nombre de Clients", f"{len(data)}")
    col_kpi4.metric("Perte Totale (Churn)", f"{data[data['Churn Label']=='Yes']['Total Charges'].sum():,.0f} $")
    
    st.divider()

    # 2. Les Raisons du départ
    if 'Churn Reason' in data.columns:
        st.subheader("Pourquoi les clients partent-ils ?")
        col_reason1, col_reason2 = st.columns([2, 1])
        
        with col_reason1:
            # Calcul du top 10 des raisons
            churn_reasons = data[data['Churn Label'] == 'Yes']['Churn Reason'].value_counts().reset_index()
            churn_reasons.columns = ['Reason', 'Count']
            
            fig_reason = px.bar(
                churn_reasons.head(10), x='Count', y='Reason', orientation='h',
                title="Top 10 des motifs de résiliation",
                color='Count', color_continuous_scale='Reds'
            )
            fig_reason.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_reason, use_container_width=True)
            
        with col_reason2:
            st.info(
                "**Analyse :**\n\n"
                "Ce graphique isole les motifs précis déclarés par les clients lors de la résiliation. "
                "C'est l'indicateur clé pour orienter les actions correctives."
            )

    st.divider()

    # 3. Visualisations Temporelles et Services
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader("Quand le risque est-il le plus fort ?")
        # Courbe de survie (Tenure vs Churn Rate)
        churn_by_tenure = data.groupby('Tenure Months')['Churn Binary'].mean().reset_index()
        churn_by_tenure['Churn Rate %'] = (churn_by_tenure['Churn Binary'] * 100).round(1)
        
        fig_line = px.line(
            churn_by_tenure, 
            x='Tenure Months', 
            y='Churn Rate %',
            title="Évolution du risque selon l'ancienneté",
            markers=True
        )
        fig_line.add_vrect(x0=0, x1=12, fillcolor="red", opacity=0.1, annotation_text="Danger Immédiat")
        st.plotly_chart(fig_line, use_container_width=True)

    with col_g2:
        st.subheader("Impact du Service Internet")
        churn_internet = data.groupby('Internet Service')['Churn Binary'].mean().reset_index()
        churn_internet['Churn Rate %'] = (churn_internet['Churn Binary'] * 100).round(1)
        
        fig_bar = px.bar(
            churn_internet, 
            x='Internet Service', 
            y='Churn Rate %',
            color='Churn Rate %',
            title="Taux de désabonnement par Technologie",
            color_continuous_scale='Reds',
            text_auto=True
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 4. Exploration Dynamique (Classique)
    st.subheader("Comparateur de Facteurs")
    feature_options = ['Contract', 'Payment Method', 'Tech Support', 'Gender', 'Senior Citizen']
    selected_feature = st.selectbox("Comparer le churn selon :", feature_options)

    fig_hist = px.histogram(
        data, x=selected_feature, color='Churn Label', 
        barmode='group',
        title=f"Distribution du Churn par {selected_feature}",
        color_discrete_map={'No':'#636EFA', 'Yes':'#EF553B'},
        text_auto=True
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # --- AJOUT DES NOUVELLES VISUALISATIONS ICI ---
    st.divider()
    
    st.subheader("Impact Financier (Distribution des Factures)")
    col_box1, col_box2 = st.columns([2, 1])
    
    with col_box1:
        fig_box = px.box(
            data, 
            x='Churn Label', 
            y='Monthly Charges', 
            color='Churn Label',
            title="Comparaison des factures mensuelles (Fidèles vs Partis)",
            points="outliers", 
            color_discrete_map={'No': '#636EFA', 'Yes': '#EF553B'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        
    with col_box2:
        st.info(
            "**Lecture :**\n"
            "Si la boîte rouge est plus haute que la bleue, cela confirme que "
            "les clients qui partent ont généralement des factures plus élevées."
        )

    st.divider()

    st.subheader("Matrice de Corrélation")
    st.write("Analyse statistique des liens entre les variables numériques.")
    
    # Sélection des colonnes numériques
    numeric_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges', 'Churn Binary']
    if 'CLTV' in data.columns:
        numeric_cols.append('CLTV')

    corr_matrix = data[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix, 
        text_auto=".2f",
        aspect="auto",
        title="Carte de chaleur des corrélations",
        color_continuous_scale='RdBu_r',
        origin='lower'
    )
    st.plotly_chart(fig_corr, use_container_width=True)


# --- Onglet 2 : Prédiction ---
with tab_prediction:
    st.header("Simulateur de Churn & Recherche de Profils")
    st.info("Modifiez les paramètres pour estimer le risque et trouver des clients similaires.")
    
    if model:
        col_input, col_res = st.columns([1, 2])
        
        with col_input:
            st.markdown("### Profil Simulé")
            # Ajout de clés uniques pour éviter les conflits d'interface
            tenure = st.slider("Ancienneté (Mois)", 0, 72, 12, key='tenure_pred')
            monthly = st.slider("Facture Mensuelle ($)", 18.0, 120.0, 70.0, key='monthly_pred')
            contract = st.selectbox("Type de Contrat", data['Contract'].unique(), key='contract_pred')
            internet = st.selectbox("Service Internet", data['Internet Service'].unique(), key='internet_pred')
            tech_support = st.selectbox("Support Technique", data['Tech Support'].unique(), key='tech_pred')
            payment = st.selectbox("Moyen de Paiement", data['Payment Method'].unique(), key='payment_pred')

            # Création du DataFrame pour le modèle
            # Note : Pour les variables non demandées à l'utilisateur, on prend le mode (valeur la plus fréquente)
            input_df = pd.DataFrame({
                'Tenure Months': [tenure], 'Monthly Charges': [monthly],
                'Total Charges': [monthly * tenure], 
                'Gender': [data['Gender'].mode()[0]],
                'Senior Citizen': [data['Senior Citizen'].mode()[0]], 
                'Partner': [data['Partner'].mode()[0]],
                'Dependents': [data['Dependents'].mode()[0]], 
                'Phone Service': [data['Phone Service'].mode()[0]],
                'Internet Service': [internet], 
                'Contract': [contract],
                'Payment Method': [payment], 
                'Online Security': [data['Online Security'].mode()[0]],
                'Tech Support': [tech_support]
            })

            predict_btn = st.button("Analyser ce profil", type="primary")

        with col_res:
            st.markdown("### 1. Prédiction du Modèle")
            if predict_btn:
                # --- Partie Prédiction ---
                proba = model.predict_proba(input_df)[0][1]
                st.metric("Probabilité de départ estimée", f"{proba*100:.1f} %")
                
                if proba > 0.5:
                    st.error("🚨 Client à HAUT RISQUE")
                elif proba > 0.3:
                    st.warning("⚠️ Client à RISQUE MODÉRÉ")
                else:
                    st.success("✅ Client FIDÈLE")
                
                st.divider()

                # --- Partie Recherche de Similaires (NOUVEAU) ---
                st.markdown("### 2. Clients réels similaires dans la base")
                st.write(f"Recherche de clients avec : **{contract}**, **{internet}**, **{payment}**...")
                
                # Filtrage Intelligent (avec tolérance pour les chiffres)
                tolerance_tenure = 6  # +/- 6 mois
                tolerance_price = 10  # +/- 10 dollars

                similar_clients = data[
                    (data['Contract'] == contract) &
                    (data['Internet Service'] == internet) &
                    (data['Payment Method'] == payment) &
                    (data['Tech Support'] == tech_support) &
                    (data['Tenure Months'].between(tenure - tolerance_tenure, tenure + tolerance_tenure)) &
                    (data['Monthly Charges'].between(monthly - tolerance_price, monthly + tolerance_price))
                ]

                count_sim = len(similar_clients)
                
                if count_sim > 0:
                    churn_real_rate = (similar_clients['Churn Binary'].mean() * 100)
                    
                    st.write(f"🔎 **{count_sim} clients trouvés** correspondant à ces critères (à ±{tolerance_tenure} mois et ±{tolerance_price}$).")
                    
                    col_metric1, col_metric2 = st.columns(2)
                    col_metric1.metric("Taux de Churn réel observé", f"{churn_real_rate:.1f} %")
                    col_metric2.metric("Facture Moyenne du groupe", f"{similar_clients['Monthly Charges'].mean():.2f} $")
                    
                    with st.expander("Voir la liste détaillée des clients"):
                        # On affiche les colonnes les plus pertinentes
                        cols_to_show = ['CustomerID', 'Churn Label', 'Tenure Months', 'Monthly Charges', 'City', 'Total Charges']
                        st.dataframe(
                            similar_clients[cols_to_show].sort_values(by='Churn Label', ascending=False),
                            hide_index=True,
                            use_container_width=True
                        )
                else:
                    st.warning("Aucun client exact trouvé. Essayez d'élargir les critères ou de changer le type de contrat.")

            else:
                st.write("Cliquez sur 'Analyser ce profil' pour voir la prédiction et les données réelles.")
    else:
        st.warning("Modèle non chargé.")

# --- Onglet 3 : Carte Géographique ---
with tab_carte:
    st.header("Géographie des Départs")
    
    col_controls, col_map = st.columns([1, 3])
    
    with col_controls:
        st.markdown("### Filtres Carte")
        map_style = st.radio(
            "Style de visualisation", 
            ["Points (Fiche Client)", "Zones de Chaleur (Heatmap)", "Colonnes 3D (Volume)"]
        )
        show_churn_only = st.checkbox("Voir seulement les départs", value=True)
        
        all_cities = ["Toutes les villes"] + list(data['City'].unique())
        city_filter = st.selectbox("Filtrer par Ville", all_cities[:50])

    # Filtrage des données
    map_data = data.copy()
    if show_churn_only:
        map_data = map_data[map_data['Churn Label'] == 'Yes']
    
    if city_filter != "Toutes les villes":
        map_data = map_data[map_data['City'] == city_filter]
        zoom_level = 11
    else:
        zoom_level = 5

    # Centrage de la carte
    mid_lat = map_data['Latitude'].mean() if not map_data.empty else 37
    mid_lon = map_data['Longitude'].mean() if not map_data.empty else -122
    
    initial_view = pdk.ViewState(
        latitude=mid_lat, longitude=mid_lon,
        zoom=zoom_level, pitch=45
    )

    layers = []
    tooltip = None # Par défaut

    # Configuration des couches (Layers)
    if map_style == "Points (Fiche Client)":
        # Couleur : Rouge si Churn, Vert sinon
        map_data['color'] = map_data['Churn Label'].apply(lambda x: [200, 30, 30, 200] if x == 'Yes' else [30, 200, 30, 200])
        
        layers.append(pdk.Layer(
            'ScatterplotLayer', 
            data=map_data,
            get_position='[Longitude, Latitude]',
            get_fill_color='color', 
            get_radius=200, 
            pickable=True,       # INDISPENSABLE pour l'interactivité
            auto_highlight=True, # Surligne le point au survol
            opacity=0.8
        ))
        
        # LE TOOLTIP INTERACTIF (HTML)
        tooltip = {
            "html": """
                <div style='font-family: sans-serif; font-size: 12px; padding: 5px;'>
                    <b>Client ID :</b> {CustomerID} <br>
                    <b>Statut :</b> {Churn Label} <br>
                    <hr style='margin: 3px 0;'>
                    <b>Ville :</b> {City} <br>
                    <b>Facture :</b> {Monthly Charges} $ <br>
                    <b>Contrat :</b> {Contract} <br>
                    <b>Ancienneté :</b> {Tenure Months} mois
                </div>
            """,
            "style": {
                "backgroundColor": "#1e1e1e",
                "color": "white",
                "borderRadius": "5px",
                "border": "1px solid #EF553B"
            }
        }

    elif map_style == "Zones de Chaleur (Heatmap)":
        layers.append(pdk.Layer(
            "HeatmapLayer", data=map_data,
            get_position='[Longitude, Latitude]',
            opacity=0.9, threshold=0.1, radiusPixels=40
        ))
        tooltip = None

    elif map_style == "Colonnes 3D (Volume)":
        layers.append(pdk.Layer(
            "HexagonLayer", data=map_data,
            get_position='[Longitude, Latitude]',
            radius=1000, elevation_scale=50, elevation_range=[0, 3000],
            pickable=True, extruded=True
        ))
        tooltip = {"html": "<b>Volume :</b> {elevationValue} clients"}

    # Affichage
    with col_map:
        if not map_data.empty:
            st.pydeck_chart(pdk.Deck(
                layers=layers, 
                initial_view_state=initial_view,
                map_style=None,
                tooltip=tooltip # On passe le dictionnaire tooltip ici
            ))
            if map_style == "Points (Fiche Client)":
                st.info("💡 Survolez un point pour voir la fiche détaillée du client.")
        else:
            st.warning("Aucune donnée pour cette sélection.")