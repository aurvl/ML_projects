# Scraping d'Informations de Films depuis AlloCiné

## Vue d'ensemble du projet
Ce projet a pour objectif d'extraire des informations de films depuis AlloCiné en scrappant des pages HTML stockées localement. Les données collectées sont ensuite stockées dans un format structuré (DataFrame Pandas) pour une analyse ultérieure. La première phase consiste à récupérer les détails des films tels que le titre, la durée, le genre, le réalisateur, les évaluations, et plus encore à partir de chaque page.

## Objectifs
1. **Scraping des données de films** : Extraire des données de 57 pages de films stockées localement.
2. **Stockage des données** : Stocker les données scrappées dans une DataFrame pour une analyse future.
3. **Analyses futures** :
   - Visualiser les données collectées à l’aide de Power BI.
   - Construire un modèle de machine learning pour prédire une des variables extraites (par exemple, la note du public, le nombre de saisons).

## Champs de données
Les informations extraites pour chaque film incluent :
- **Titre** : Titre du film
- **Statut** : Film terminé ou en cours
- **Période** : Dates de sortie et de fin
- **Durée** : Durée du film (en minutes)
- **Type/Genre** : Genre du film
- **Réalisateur** : Nom(s) du/des réalisateur(s)
- **Personnage principal** : Acteur(s) principal(aux) du film
- **Nationalité** : Pays d'origine
- **Chaîne** : Chaîne de diffusion originale
- **Note de la presse** : Évaluations fournies par la presse
- **Note du public** : Évaluations fournies par le public
- **Nombre de saisons et épisodes** : Pour les séries, le nombre total de saisons et d'épisodes
- **Description** : Courte description ou synopsis du film

## Processus étape par étape

### Étape 1 : Scraping et stockage des données
En utilisant des expressions régulières, les champs de données suivants ont été extraits des pages HTML :
- Titres, genres, évaluations, noms de réalisateurs, etc.

Ces champs ont ensuite été organisés dans une DataFrame Pandas. Voici un exemple de ligne extraite de la DataFrame :

| Titre    | Statut | Période   | Durée  | Genre   | Réalisateur     | Note public | Saisons | Épisodes | Description       |
|----------|--------|-----------|--------|---------|-----------------|-------------|---------|----------|-------------------|
| El Barco | None   | 2011-2013 | 75 min | Aventure| Iván Escobar    | 3.8         | None    | None     | None              |

### Étape 2 : Travaux futurs
Les prochaines étapes du projet incluront :
1. **Rapport Power BI** : Visualisation des données en créant un tableau de bord interactif présentant les statistiques clés et les tendances des films.
2. **Modèle de machine learning** : Utilisation du dataset pour construire un modèle de machine learning visant à prédire une des variables, comme la note du public ou le nombre de saisons.

## Outils et bibliothèques
- **Python**
  - `re` pour le scraping basé sur les expressions régulières
  - `pandas` pour la manipulation des données
  - `html` pour la gestion des caractères spéciaux dans les fichiers HTML
  - `os` pour la gestion des répertoires de fichiers
- **Power BI** (pour les analyses futures)
- **Framework de machine learning** (à déterminer dans la prochaine étape)

## Comment exécuter
1. Placez les fichiers HTML dans le répertoire `Data/Pages`.
2. Exécutez le script de scraping pour extraire les informations des films dans une DataFrame.
3. Exportez la DataFrame vers un fichier CSV pour l’utiliser dans Power BI ou pour la modélisation.

```bash
python scrape_allocine.py
```

## Plans futurs
- **Tableau de bord Power BI** : Une fois les données collectées, elles seront utilisées pour créer un rapport dynamique dans Power BI.
- **Machine Learning** : Explorer différents modèles pour prédire la note du public ou une autre variable pertinente à partir du dataset.

## Remerciements
- Données extraites de [AlloCiné](https://www.allocine.fr)
