import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


job_keywords = {
    'Engineer': ['Engineer', 'Developer', 'Architect', 'ETL', 'MLOps', 'Infrastructure'],
    'Analyst': ['Analyst', 'BI', 'Data Analytics', 'Financial'],
    'Scientist': ['Scientist', 'Research', 'Machine Learning', 'AI', 'Deep Learning'],
    'Others': ['Manager', 'Head', 'Lead', 'Consultant', 'Specialist']
}
def categorize_job(title):
    for category, keywords in job_keywords.items():
        if any(keyword.lower() in title.lower() for keyword in keywords):
            return category
    return 'Others'

def preprocessor(df, train):
    """
    Pipeline de Prétraitement des Données

    Ce pipeline effectue le prétraitement des données pour la prédiction des salaires en USD. Il fonctionne en mode entraînement (`train=True`) et en mode prédiction (`train=False`).

    Étapes du prétraitement :
        1. Vérification des colonnes : Vérifie la présence des colonnes requises et signale les éventuelles absences.
        2. Catégorisation des postes : Regroupe les intitulés de postes selon des catégories spécifiques.
        3. Imputation des valeurs manquantes :
            - Variables numériques : Remplacement par la médiane.
            - Variables catégorielles : Remplacement par la modalité la plus fréquente.
    4. Encodage :
        - Target encoding pour certaines variables catégorielles.
        - OneHotEncoding pour experience_level, 'company_size', 'employment_type' et 'job_groups'.
    5. Création de la variable 'company_job_size' : Facteur combinant taille de l'entreprise, poste, localisation et expérience.
    6. Standardisation des données avec 'MinMaxScaler'.
    7. Génération de polynomes des variables (degré 2).
    8. Sauvegarde et chargement du pipeline :
        - Lors de l'entraînement ('train=True'), les objets de transformation sont sauvegardés.
        - Lors de la prédiction ('train=False'), le pipeline est rechargé et appliqué aux nouvelles données.

    Utilisation :
        - 'train=True' : Entraîne et sauvegarde le pipeline.
        - 'train=False' : Applique le pipeline sur les nouvelles données.
    """

    # Verif des colonnes
    # Selection des var retenus pour le modèle
    # df = df[[]]
    famd = ["job_title", "employment_type", "work_year", "experience_level",
            "company_size", "company_location", "remote_ratio"]
    
    missing_cols = [col for col in famd if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Les colonnes suivantes sont manquantes : {missing_cols}")
    
    if train:
        df = df[famd + ['salary_in_usd']]
        y = df['salary_in_usd']
    else:
        df = df[famd]

    # Séparation des colonnes numériques et catégorielles
    if 'salary_in_usd' in df.columns:
        num_cols = df.drop(columns=['salary_in_usd']).select_dtypes(include=[np.number]).columns
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns

    df['job_groups'] = df['job_title'].apply(categorize_job)

    cat_cols = df.select_dtypes(include=['object']).columns
    cats = df.drop(columns=['experience_level', 'company_size', 'employment_type', 'job_groups']).select_dtypes(include=['object']).columns
    
    if train:
        std_salary_per_comp_loc = df.groupby('company_location')['salary_in_usd'].std()
        std_salary_per_job_title = df.groupby('job_title')['salary_in_usd'].std()
        std_salary_per_exp = df.groupby('experience_level')['salary_in_usd'].std()
        
        df = df.drop(columns=['salary_in_usd'])
        # Imputation des valeurs manquantes
        num_imputer = SimpleImputer(strategy='median')
        df[num_cols] = num_imputer.fit_transform(df[num_cols])

        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        imputers = (num_imputer, cat_imputer)
        
        df['company_job_size'] = (df['company_size'].map({'S': 1, 'M': 2, 'L': 3}) * 
                                  df['job_title'].map(std_salary_per_job_title) * 
                                  df['company_location'].map(std_salary_per_comp_loc) *
                                  df['experience_level'].map(std_salary_per_exp))

        bouche = df['company_job_size'].median()
        df['company_job_size'] = df['company_job_size'].fillna(bouche.mean())

        stds = (std_salary_per_job_title, std_salary_per_comp_loc, std_salary_per_exp, bouche)

        df['salary_in_usd'] = y
        # Target Encoding des variables catégorielles
        target_encoders = {}
        for col in cats:
            target_mapping = df.groupby(col)['salary_in_usd'].mean().to_dict()
            df[col] = df[col].map(target_mapping)
            target_encoders[col] = target_mapping
        
        df = df.drop(columns=['salary_in_usd'])
                
        df = pd.get_dummies(df, columns=['experience_level', 'company_size', 'employment_type', 'job_groups'], drop_first=True)
        dummies_columns = df.columns.tolist()
        
        # Standardisation
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        
        # Sauvegarde du pipeline
        pipelines = (scaler, dummies_columns, imputers, target_encoders, stds, df.columns)
        with open('./modules/preprocessors.pkl', 'wb') as f:
            pickle.dump(pipelines, f)
        
    else:
        with open('./modules/preprocessors.pkl', 'rb') as f:
            scaler, dummies_columns, imputers, target_encoders, stds, trained_columns = pickle.load(f)
        
        std_salary_per_job_title, std_salary_per_comp_loc, std_salary_per_exp, bouche = stds

        num_imputer, cat_imputer = imputers 
        df[num_cols] = num_imputer.transform(df[num_cols])
        df[cat_cols] = cat_imputer.transform(df[cat_cols])
        
        df['company_job_size'] = (df['company_size'].map({'S': 1, 'M': 2, 'L': 3}) * 
                                  df['job_title'].map(std_salary_per_job_title) * 
                                  df['company_location'].map(std_salary_per_comp_loc) *
                                  df['experience_level'].map(std_salary_per_exp))
        df['company_job_size'] = df['company_job_size'].fillna(bouche.mean())

        # Target Encoding en appliquant les mappings existants
        for col in cats:
            df[col] = df[col].map(target_encoders[col])
            df[col] = df[col].fillna(np.mean(list(target_encoders[col].values())))  # Remplacement des valeurs inconnues par la moyenne globale
        
        df = pd.get_dummies(df, columns=['experience_level', 'company_size', 'employment_type', 'job_groups'], drop_first=True)
        df = df.reindex(columns=dummies_columns, fill_value=0)
        
        # Alignement des colonnes avec l'entraînement
        df = df.reindex(columns=trained_columns, fill_value=0)
        
        scaled_data = scaler.transform(df)
    
    df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    if train:
        poly = PolynomialFeatures(degree= 2, include_bias= False)
        X_poly = poly.fit_transform(df)
        
        with open('./modules/poly.pkl', 'wb') as f:
            pickle.dump(poly, f)    
    else:
        with open('./modules/poly.pkl', 'rb') as f:
            poly = pickle.load(f)

        X_poly = poly.transform(df)
    
    df = pd.DataFrame(X_poly, columns= poly.get_feature_names_out(df.columns), index= df.index)

    print("Le prétraitement des données a été effectué avec succès !")
    return df