# Projet de Prédiction de Salaires

Ce projet vise à prédire les salaires en utilisant des modèles d’apprentissage automatique, notamment une régression linéaire (LR) et un Random Forest Regressor (RF). Le dépôt est organisé pour simplifier le prétraitement des données, l’entraînement des modèles, l’optimisation des hyperparamètres et la prédiction sur de nouvelles données.

------------------------------------------------------------------------

## **Structure du Projet**

### **Répertoires et Fichiers**

#### **`data/`**

-   `data_eval.csv` : Jeu de données qui doit être utilisé pour l’évaluation et les tests des modèles.
-   `training.csv` : Jeu de données utilisé pour l’entraînement des modèles.

#### **`modules/`**

-   `models.pkl` : Contient les modèles de régression linéaire et Random Forest sérialisés.
-   `poly.pkl` : Contient la transformation des Polynomial Features utilisée pour l’ingénierie des variables.
-   `preprocessors.pkl` : Stocke les composants du pipeline de prétraitement pour transformer les jeux de données.

#### **Scripts Python Principaux**

-   `pipeline.py` : Script pour le prétraitement des données, incluant l’ingénierie des variables, le scaling et l’encodage.
-   `post.py` : Script pour utiliser les modèles entraînés pour effectuer des prédictions sur de nouvelles données.
-   `model_implementation.py` : Contient le code pour charger et tester les modèles.
-   `tuning.py` : Script pour optimiser les hyperparamètres du modèle Random Forest et sélectionner les meilleures variables pour la régression linéaire.

#### **Autres Fichiers**

-   `res.csv` : Contient les meilleures combinaisons de variables pour la régression linéaire, basées sur l’optimisation.
-   `rapport.pdf` : Documentation de la méthodologie, des expériences et des résultats du projet.
-   `requirements.txt` : Liste des bibliothèques Python nécessaires et leurs versions.
-   `eda.ipynb` : Contient l'analyse exploratoire des données
-   `Readme.md` : Ce fichier, qui documente la structure et l’utilisation du projet.

------------------------------------------------------------------------

## **Utilisation**

### **1. Prétraitement des données**

Le pipeline de prétraitement est implémenté dans `pipeline.py`. Il gère :

-   L’encodage des variables catégoriques.
-   Le scaling des variables numériques.
-   L’ingénierie des variables (par exemple, Polynomial Features).
-   La sauvegarde du pipeline de prétraitement dans `preprocessors.pkl` pour garantir la cohérence entre les scripts.

### **2. Entraînement et Optimisation des Modèles**

-   **Random Forest et Régression Linéaire** :
    -   Entraînez les modèles avec `model_implementation.``py`.
    -   L’optimisation des hyperparamètres pour Random Forest et la sélection des variables pour la régression linéaire sont gérées dans `tuning.py`.
    -   Les meilleurs paramètres et variables sont sauvegardés pour garantir la reproductibilité.

### **3. Prédiction**

-   Utilisez `post.py` pour effectuer des prédictions sur de nouvelles données (`data_eval.csv`) avec les modèles entraînés. Le pipeline garantit que les données d’entrée sont prétraitées de manière cohérente.

### **4. Meilleures Variables et Hyperparamètres**

-   Les meilleures combinaisons de variables pour la régression linéaire sont stockées dans `res.csv`.
-   Les hyperparamètres du Random Forest sont stockés dans `models.pkl` avec le modèle entraîné.

------------------------------------------------------------------------

## **Comment Exécuter**

1.  **Installer les dépendances** :

    ``` bash
    pip install -r requirements.txt
    ```

2.  **Entraîner les modèles** : Lancez les scripts d’entraînement et d’optimisation :

    ``` bash
    python model_implementation.py
    python tuning.py
    ```

3.  **Effectuer des prédictions** : Utilisez `post.py` pour prédire de nouvelles données :

    ``` bash
    python post.py
    ```

4.  **Consulter les résultats** : Consultez `res.csv` pour les meilleures variables de la régression linéaire et `models.pkl` pour les modèles entraînés.

------------------------------------------------------------------------

## **Dépendances**

Toutes les dépendances requises et leurs versions sont listées dans `requirements.txt`. Les bibliothèques principales incluent :

-   `pandas`
-   `scikit-learn`
-   `joblib`
-   `numpy`

------------------------------------------------------------------------

## **Auteur**

Ce projet a été développé par [Aurel](https://github.com/aurvl) . Pour toute question, n’hésitez pas à me contacter ([Contacts](https://aurvl.github.io/portfolio/home_fr.html#contacts)) !
