import gradio as gr
import pandas as pd
import numpy as np
from rdkit import Chem
from mordred import Calculator, descriptors
import joblib
import warnings

# Supprimer les avertissements de RDKit/Mordred pendant le chargement (optionnel)
warnings.filterwarnings("ignore")

# --- A. CHARGEMENT DES COMPOSANTS (Cache du modèle) ---
# NOTE : Les fichiers .joblib doivent être dans le même dossier
try:
    # Utilisation d'un cache pour ne charger les fichiers qu'une seule fois
    model = joblib.load('best_et_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    target_names = joblib.load('target_names.joblib')
    calc = Calculator(descriptors, ignore_3D=True)
except FileNotFoundError:
    # En cas d'échec du chargement, on affiche un message d'erreur et on quitte (critique pour le déploiement)
    print("ERREUR : Assurez-vous que tous les fichiers .joblib sont présents dans le dossier.")
    raise

def predict_activity_gradio(smiles: str) -> pd.DataFrame:
    """
    Fonction principale appelée par Gradio.
    Prend un SMILES et retourne un DataFrame formaté.
    """
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        # Retourne un DataFrame d'erreur si le SMILES est invalide
        return pd.DataFrame({'Erreur': ["Code SMILES invalide. Veuillez vérifier le format."]})
    
    # --- PRÉ-TRAITEMENT ---
    df_features_new = calc.pandas([mol])
    df_features_new = df_features_new.apply(pd.to_numeric, errors='coerce')
    df_features_new = df_features_new.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df_features_aligned = pd.DataFrame(columns=feature_names)
    df_features_aligned.loc[0] = 0 
    df_features_aligned.update(df_features_new)
    
    X_new_scaled = scaler.transform(df_features_aligned[feature_names])

    # --- PRÉDICTION ---
    prediction = model.predict(X_new_scaled)[0]
    
    # --- FORMATAGE DES RÉSULTATS ---
    results_df = pd.DataFrame({
        'Souche Bactérienne': [col.replace('Label_', '').replace('(NT', ' (NT') for col in target_names],
        'Inhibition Prédite': prediction
    })
    
    # Afficher uniquement les prédictions positives (Label = 1)
    positive_predictions = results_df[results_df['Inhibition Prédite'] == 1].copy()
    
    if not positive_predictions.empty:
        # Si actif, ne retourne que les lignes actives
        positive_predictions['Activité'] = 'OUI (Inhibiteur)'
        return positive_predictions[['Souche Bactérienne', 'Activité']]
    else:
        # Si inactif, retourne un message clair dans le format DataFrame
        return pd.DataFrame({'Souche Bactérienne': ["Aucune inhibition prédite."], 'Activité': ["INACTIF"]})


# --- C. INTERFACE GRADIO ---
# Définition de l'interface
demo = gr.Interface(
    fn=predict_activity_gradio,
    inputs=gr.Textbox(
        label="Code SMILES de la molécule :",
        placeholder="Exemple : C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O (Naproxène)"
    ),
    outputs=gr.DataFrame(
        label="Résultats de l'activité inhibitrice (Modèle AUROC 0.973)"
    ),
    title="🔬 Prédicteur d'Effets Médicamenteux sur le Microbiote",
    description="Entrez le code SMILES d'un médicament pour prédire son activité inhibitrice sur 40 souches bactériennes intestinales. Basé sur le modèle optimisé Extra Trees. **Note:** La prédiction `INACTIF` signifie que la probabilité d'inhibition est faible (p-value > 0.05).",
    examples=[
        ['C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O'] # Exemple: Naproxène
    ]
)

# Lancement du serveur Gradio
if __name__ == "__main__":
    # Pour le déploiement sur Hugging Face Spaces, Gradio utilise le mode 'default'
    # Pour le test local, vous pouvez laisser .launch()
    demo.launch()
