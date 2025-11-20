# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

print("--- Démarrage de l'entraînement du modèle ---")

# --- 1. Charger les données (même logique que app.py) ---
# LA LIGNE "@st.cache_data" A ÉTÉ SUPPRIMÉE ICI
def load_data(path):
    try:
        data = pd.read_csv(path, sep=';', encoding='latin1')
    except:
        data = pd.read_csv(path, sep=';')

    # Gérer les décimaux avec virgule
    if 'Total Charges' in data.columns:
        data['Total Charges'] = data['Total Charges'].astype(str).str.replace(',', '.')
    if 'Monthly Charges' in data.columns:
        data['Monthly Charges'] = data['Monthly Charges'].astype(str).str.replace(',', '.')

    # Conversion numérique
    data['Total Charges'] = pd.to_numeric(data['Total Charges'], errors='coerce')
    data['Monthly Charges'] = pd.to_numeric(data['Monthly Charges'], errors='coerce')
    data = data.dropna(subset=['Total Charges'])
    
    # 'Senior Citizen' est 0 ou 1, mais c'est une catégorie
    data['Senior Citizen'] = data['Senior Citizen'].astype(str)
    
    return data

data = load_data("Telco_customer_churn.csv")

# --- 2. Définir les features (X) et la cible (y) ---
# (Le reste du code est identique à avant)
target = 'Churn Value' 
numeric_features = ['Tenure Months', 'Monthly Charges', 'Total Charges']
categorical_features = [
    'Gender', 'Senior Citizen', 'Partner', 'Dependents', 
    'Phone Service', 'Internet Service', 'Contract', 'Payment Method',
    'Online Security', 'Tech Support'
]
features = numeric_features + categorical_features

X = data[features]
y = data[target]

print(f"Features sélectionnées : {features}")

# --- 3. Créer le Pipeline de Pré-traitement ---
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 4. Définir le modèle ---
model = LogisticRegression(max_iter=1000)

# --- 5. Créer le Pipeline Final (Pré-traitement + Modèle) ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# --- 6. Entraîner le modèle ---
print("Entraînement en cours...")
pipeline.fit(X, y)

# --- 7. Sauvegarder le modèle ---
file_path = 'churn_model.pkl'
joblib.dump(pipeline, file_path)

print(f"--- Modèle entraîné et sauvegardé sous '{file_path}' ! ---")