# Mini-Projet 2 : Classification d'images de téléphones

Ce projet a été réalisé en binôme par Adam El Asrag et Lazim Sar. L'objectif est de construire un pipeline complet de classification d'images, appliqué à un jeu de données original collecté par les étudiants.

## Problème traité

Le projet cherche à distinguer automatiquement des photographies de téléphones allumés (écran éclairé, classe +1) de photographies de téléphones éteints (écran noir, classe -1). Les images ont été prises dans des conditions variées, avec différents fonds et différents modèles de téléphones.

## Structure du projet

Le projet s'organise autour des feuilles Jupyter suivantes, rédigées au format MyST Markdown et exécutables avec JupyterLab.

`0_analyse.md` est la feuille principale du projet. Elle intègre l'ensemble de la démarche : chargement des images, prétraitement, extraction d'attributs, analyse de corrélation, sélection des features et évaluation des classificateurs. C'est le seul fichier évalué par les encadrants.

`1_jeu_de_donnees.md` décrit comment le jeu de données a été constitué et comment les images sont organisées dans le dossier `data/`.

`2_extraction_d_attributs.md` présente l'extraction des attributs ad-hoc et l'analyse en composantes principales (ACP).

`3_classificateurs.md` compare plusieurs classificateurs scikit-learn et analyse les phénomènes de surapprentissage et de sous-apprentissage.

`0_diaporama.md` contient le squelette de la présentation orale (soutenance de 6 minutes en semaine 9).

`utilities.py` regroupe toutes les fonctions utilitaires réutilisées et développées pour ce projet.

## Approche technique

### Prétraitement des images

Les images brutes sont recadrées en vignettes de 90×90 pixels centrées sur l'objet. Un filtre de premier plan (`my_foreground_filter`) isole le téléphone du fond en combinant `foreground_redness_filter` avec un seuil de 0.6, une inversion conditionnelle si le fond est clair, et un lissage gaussien. Le centre de masse des pixels de premier plan est ensuite calculé pour servir de point d'ancrage au recadrage final.

### Extraction d'attributs

Vingt-trois attributs couleur sont calculés sur les pixels de premier plan via la fonction `get_colors`, couvrant les composantes RGB brutes, la chroma, les différences de canaux et la luminosité. Deux attributs spécifiques au problème ont été développés : `brightness` (moyenne de (R+G+B)/3 sur l'avant-plan) et `std_brightness` (écart-type de cette luminosité). Des filtres adaptés (Matched Filters) ont également été construits à partir de templates moyens calculés sur les exemples de chaque classe.

### Sélection des attributs

Une analyse de corrélation montre que la luminosité est le facteur le plus discriminant pour séparer les deux classes. L'attribut M (maximum RGB), `brightness` et `std_brightness` concentrent l'essentiel de l'information utile. La courbe d'apprentissage en fonction du nombre d'attributs confirme qu'au-delà de 7 features, les performances de test se dégradent en raison du surapprentissage. Les trois attributs retenus sont donc M, `brightness` et `std_brightness`.

### Classification et évaluation

Le classificateur k-NN avec k=3 est utilisé comme modèle principal, évalué par validation croisée stratifiée (10 splits, 50% test) et mesuré par la `balanced_accuracy_score` de scikit-learn. Les performances sont comparées selon les trois jeux d'attributs : 6 attributs ad-hoc initiaux, 23 attributs étendu, puis la sélection finale des 3 attributs les plus pertinents. Les résultats sont exportés dans `features_data.csv`.

## Installation et dépendances

Le projet requiert Python 3 avec les bibliothèques suivantes.

```
pip install numpy pandas scipy matplotlib seaborn Pillow scikit-learn jupyterlab jupytext
```

## Lancement

Pour ouvrir une feuille dans JupyterLab, lancez `jupyter lab` depuis la racine du projet puis ouvrez `0_analyse.md`. Les feuilles `.md` sont exécutées comme des notebooks grâce à Jupytext.


## Résultats principaux

L'analyse montre que la distinction entre téléphone allumé et éteint repose essentiellement sur la luminosité de l'écran. Trois attributs suffisent à obtenir une `balanced accuracy` proche de 0.95 sur l'ensemble de test, sans surapprentissage notable. Les autres attributs (composantes RGB individuelles, chroma, filtres adaptés) sont largement redondants avec ces trois mesures de luminosité.
