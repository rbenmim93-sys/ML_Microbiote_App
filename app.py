import gradio as gr
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import KekulizeException
from mordred import Calculator, descriptors
import joblib
import warnings
from rdkit import RDLogger
import pubchempy as pcp

# --- Configuration Initiale ---
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings("ignore")

calc = None 
global_importances = None # Pour stocker les importances brutes

# --- A. CHARGEMENT DES COMPOSANTS ---
try:
    model = joblib.load('best_et_model.joblib')
    scaler = joblib.load('standard_scaler.joblib')
    feature_names = joblib.load('feature_names.joblib')
    target_names = joblib.load('target_names.joblib')
    
    # --- 1. Chargement des Importances Globales ---
    if hasattr(model, 'feature_importances_'):
        global_importances = model.feature_importances_
        # Gestion du cas multi-output (moyenne des importances)
        if global_importances.ndim > 1: 
             global_importances = np.mean(global_importances, axis=0)
    else:
        # Fallback si le modèle ne donne pas d'importances (peu probable avec ExtraTrees)
        global_importances = np.ones(len(feature_names))
    # ----------------------------------------------

    try:
        calc = Calculator(descriptors, ignore_3D=True)
    except KekulizeException as e:
        print(f"ATTENTION: Erreur RDKit/Mordred à l'initialisation: {e}")
        calc = None 

except FileNotFoundError:
    print("ERREUR FATALE: Fichiers .joblib manquants.")
    raise

# --- FONCTION UTILITAIRE POUR LE NOM ---
def get_molecule_name(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles')
        if compounds:
            synonyms = compounds[0].synonyms
            if synonyms:
                return synonyms[0] 
            else:
                return compounds[0].iupac_name
        return "Nom inconnu (Absent de PubChem)"
    except Exception:
        return "Erreur PubChem"

# --- FONCTION PRINCIPALE ---
def predict_activity_gradio(smiles: str):
    
    if calc is None:
        return "Erreur Système", pd.DataFrame({'Erreur': ["Calculateur HS."]}), pd.DataFrame()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "SMILES Invalide", pd.DataFrame({'Erreur': ["Code SMILES invalide."]}), pd.DataFrame()
    
    # 1. Nom
    mol_name = get_molecule_name(smiles)
    
    # 2. Calculs Mordred
    df_features_new = calc.pandas([mol])
    df_features_new = df_features_new.apply(pd.to_numeric, errors='coerce')
    df_features_new = df_features_new.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df_features_aligned = pd.DataFrame(columns=feature_names)
    df_features_aligned.loc[0] = 0 
    df_features_aligned.update(df_features_new)
    
    # 3. Préparation des données pour le modèle
    X_new_scaled = scaler.transform(df_features_aligned[feature_names])
    
    # --- NOUVEAU : Calcul du Top 5 Dynamique (Contribution Locale) ---
    # On calcule : Valeur Scalée * Importance Globale
    # Cela nous dit : "Quelles caractéristiques de CETTE molécule sont anormalement fortes 
    # sur des critères jugés importants par le modèle ?"
    
    # On prend la valeur absolue pour voir l'impact fort (positif ou négatif)
    local_contributions = np.abs(X_new_scaled[0] * global_importances)
    
    # On récupère les indices des 5 plus grandes contributions
    top_5_indices = np.argsort(local_contributions)[::-1][:5]
    
    # On prépare les données pour l'affichage
    top_5_names = [feature_names[i] for i in top_5_indices]
    top_5_raw_values = [df_features_aligned.iloc[0, i] for i in top_5_indices] # Valeur réelle (non scalée)
    
    df_top5_display = pd.DataFrame({
        'Descripteur Clé (Spécifique)': top_5_names,
        'Valeur Mesurée': top_5_raw_values
    })
    # ----------------------------------------------------------------
    
    # 4. Prédiction
    prediction = model.predict(X_new_scaled)[0]
    
    results_df = pd.DataFrame({
        'Souche Bactérienne': [col.replace('Label_', '').replace('(NT', ' (NT') for col in target_names],
        'Inhibition Prédite': prediction
    })
    
    positive_predictions = results_df[results_df['Inhibition Prédite'] == 1].copy()
    
    if not positive_predictions.empty:
        positive_predictions['Activité'] = 'OUI (Inhibiteur)'
        final_bact_df = positive_predictions[['Souche Bactérienne', 'Activité']]
    else:
        final_bact_df = pd.DataFrame({'Souche Bactérienne': ["Aucune inhibition prédite."], 'Activité': ["INACTIF"]})

    return mol_name, final_bact_df, df_top5_display

# --- C. INTERFACE GRADIO ---
with gr.Blocks(title="Prédicteur Microbiote") as demo:
    gr.Markdown("# 🔬 Prédicteur d'Effets Médicamenteux sur le Microbiote")
    gr.Markdown("Analyse basée sur l'Acidité, la Forme et la Topologie moléculaire.")
    
    with gr.Row():
        inp = gr.Textbox(label="Code SMILES", placeholder="Ex: C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O", value='C[C@@H](C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O')
        btn = gr.Button("🔍 Analyser la Molécule", variant="primary")
    
    out_name = gr.Textbox(label="🏷️ Molécule Identifiée")
    
    with gr.Row():
        out_bact = gr.DataFrame(label="🦠 Prédictions d'Inhibition")
        out_desc = gr.DataFrame(label="⚗️ Top 5 Facteurs pour CETTE molécule")

    btn.click(fn=predict_activity_gradio, inputs=inp, outputs=[out_name, out_bact, out_desc])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
