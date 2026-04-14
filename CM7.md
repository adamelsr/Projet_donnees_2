---
jupytext:
  notebook_metadata_filter: rise
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  auto_select: first
  autolaunch: false
  centered: false
  controls: false
  enable_chalkboard: true
  height: 100%
  margin: 0
  maxScale: 1
  minScale: 1
  overlay: "<div style='position: absolute; top: 0; left: 0'>Introduction \xE0 la\
    \ Science des Donn\xE9es, L1 Math-Info, Facult\xE9 des Sciences d'Orsay</div><div\
    \ style='position: absolute; top: 0; right: 0'><img src='media/logoParisSaclay.png'\
    \ width='150'></div>"
  scroll: true
  slideNumber: true
  start_slideshow_at: selected
  transition: none
  width: 90%
---

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

<center>

# Cours 7: Estimation de l'apprentissage et Deep Learning


</center>

<br>

<center>Fanny Pouyet</center>

<center>L1 Informatique</center>

<center>Janvier - Avril</center>

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

**Précédemment**

* Chaine de traitement d'analyse de données
* Application aux images : extraction de features
* Classificateurs
* Préparation des données : traitement d'images

**Cette semaine**

* Choix du classificateur
* Deep Learning et classification d'images

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

## Déclaration des classificateurs

Il existe de très nombreux classificateurs d'images dans la librairie `scikit-learn`. 

Pouvez vous en citer au moins 3 ?

Pour chaque classificateur on peut :

- définir un ensemble d'entrainement et de test
- utiliser les méthodes `.fit` et `.predict`
- calculer des performances

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

Selon les classificateurs, on n'obtient pas les mêmes resultats. Comment choisir le meilleur classificateur ?

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Notions de sous-apprentissage et sur-apprentissage

Ces deux notions sont liés à la granularité de la frontière de décision. Elles sont illustrées par les graphiques suivants :

:::{figure} https://miro.medium.com/v2/resize:fit:720/format:webp/1*lARssDbZVTvk4S-Dk1g-eA.png
:alt: "Under-fitting, optimal and Over-fitting models"
:::

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

Lorsque l'on entraîne un classificateur sur des données, deux comportements sont à éviter:

- Lorsque à la fois les performances d'entraînement et celles de test sont mauvaises, on dit que **le classificateur a sous-appris (under-fitting)**.
- Lorsque les performances d'entraînement sont bonnes mais pas celles de test, on dit alors que **le classificateur a sur-appris (over-fitting)**.

*Question* Peut-on à la fois sur-apprendre et sous-apprendre ?

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

Pour identifier les classificateurs qui ont un de ces deux comportements on doit définir ce qu'est **une mauvaise performance**.
- on peut choisir un seuil
- on peut comparer les performances entre classificateurs. **Lors du projet 2**, "mauvais" veut dire que les performances sont inférieures à la performance médiane. (*Question* Revenir sur la définition pratique de l'underfitting/overfitting dans ces conditions)

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Comité de classificateurs

Pour chaque image, les classificateurs font des prédictions. Et ils sont soient en accord soit en désaccord entre eux. On peut faire voter les classificateurs et choisir la classe finale après le vote. 

**Comité de classificateurs** : classificateur dans lequel les réponses de plusieurs classificateurs sont combinées en une seule réponse. En d'autres termes, chaque classificateur vote pour l'une ou l'autre des catégories.

+++ {"editable": true, "slideshow": {"slide_type": ""}, "tags": []}

:::{warning}
La performance du comité n'est pas forcément la meilleure solution car les classificateurs pris individuellement peuvent être meilleurs à notre tache.
:::

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Comment gérer nos données étant donné nos classificateurs?

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Gestion de l'incertitude

Selon le degré d'accord entre classificateurs on peut **quantifier l'incertitude de nos données.**

:::{figure} media/aleatoric_epistemic.png
:alt: "Under-fitting, optimal and Over-fitting models"
:::

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

**l'incertitude aléatorique** : Les classificateurs du comité sont d'accord sur des probabilités
  incertaines : chaque classificateur émet
  une probabilité proche de `[0.5, 0.5]`. Elle est lié à l'*ambiguïté* intrinsèque de la
  tâche de classification.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

**l'incertitude épistémique** : Les classificateurs du comité sont certains de leurs prédictions,
  mais ils sont en désaccord entre eux : les
  classificateurs émettent chacun une probabilité confiante mais
  différentes: `[1.,0.]`, `[0., 1.]`, `[0.9, 0.1]`, `[0.05, 0.95]`,
  etc. Elle est
  lié à l'idée de *nouveauté* dans les données. Cette incertitude
  peut être réduite en ajoutant de nouvelles données.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

:::{admonition} Remarque

Les images avec une faible incertitude épistémique
(les classificateurs sont d'accord) et une grande incertitude
aléatorique (les classificateurs sont incertains de la
prédiction) correspondent à des images se situant aux abords de la
frontière de décision entre nos deux classes.
:::

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

##### Incertitude aléatorique

Pour chaque classificateur et chaque image, on estime cette incertitude en calculant l'entropie de Shanon
qui est une mesure issue de la théorie de l'information. 


**Entropie de Shannon** : mesure la quantité d'information contenu dans une source. Ici, notre probabilité d'appartenir à une classe pour l'image X: 

$$H(X) = - \sum_ip(i)\log_2(p(i))$$

avec  $p(i)$ la probabilité de classification sur la classe $i$.

Comportement de cette fonction avec 2 classes:

:::{figure} https://upload.wikimedia.org/wikipedia/commons/2/22/Binary_entropy_plot.svg
:alt: "Shannon entropy pour 2 classes"
:::

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

#### Incertitude épistémique

*Rappel:* Les classificateurs du comité sont certains de leurs prédictions, mais ils sont en désaccord entre eux : les classificateurs émettent chacun une probabilité confiante mais différentes: [1.,0.], [0., 1.], [0.9, 0.1], [0.05, 0.95], etc. 

Pour chaque image, on mesure le désaccord entre les classifcateurs: on calcule l'incertitude moyenne d'une image en calculant l'écart-type pour une des classes entre les classificateurs.

$$ \sigma(X) = \sqrt{\frac{\sum_{c=1}^C (x_c - \mu)^2}{C}}$$

où $c$ sont les classificateurs et $x_c$ la probabilité d'appartenir à la première classe. En effet, comme nous n'avons que deux classes l'écart-type est identique pour l'une ou l'autre.

Plus les classificateurs sont en désaccord, plus cet écart-type est grand.
L’incertitude épistémique globale correspond ensuite à la moyenne de ces désaccords sur toutes les images.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

## Deep Learning ou apprentissage profond 


* L'apprentissage statistique (*Machine Learning*) : ensemble de techniques permettant d’estimer des règles à partir de données
* L'apprentissage profond (*Deep Learning*) : apprentissage statistique reposant sur des **réseaux de neurones** dont les dimensions vont apporter une plus ou moins grande complexité à l’établissement des règles.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

Quand avons nous déjà parlé d'apprentissage profond / réseau de neurones ?

+++ {"slideshow": {"slide_type": "fragment"}}

Avec le perceptron, au CM5

+++ {"slideshow": {"slide_type": "slide"}}

### Les neurones en biologie

Neurones = cellules assurant la transmission d'un signal dans notre organisme (des dentrites jusqu'aux axones terminaux). 
```{figure} https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Neuron_-_annotated.svg/1200px-Neuron_-_annotated.svg.png
---
alt: Neurone biologique
width: 500px
align: center
---
```

+++ {"slideshow": {"slide_type": "slide"}}

#### Apercu du mécanisme
* Activation (dendrites): entrée de molécules de signal
* Transformation en signal électrique et transport le long de l'axone (par une variation de la concentration en ions entre l'environnement intra et extracellulaire)
* Arrivée : interprétation en une action ou bien en un transfert vers un autre neurone (par sécretion de molécules de signal).

+++ {"slideshow": {"slide_type": "slide"}}

#### Quelques statistiques sur les neurones
* $10^{11}$ = 100 milliards de neurones *par individu* (équivalent au nombre d'étoiles dans la Voie Lactée)
* nerf sciatique = le plus long (> 1m)
* $10^{14}$ synapses, zones de contact entre neurones chez l'adulte
```{figure} media/brains.png
---
alt: media/brains.png
width: 700px
align: center
---
```

+++ {"slideshow": {"slide_type": "slide"}}

### Les premiers neurones en informatique

**McCulloch et Pitts, 1943**
Le premier neurone artificiel (modélisation mathématique des neurones biologiques)
```{figure} media/artificial_neuron.png
---
alt: media/artificial_neuron.png
width: 400px
align: center
---
```
 Pouvez vous le décrire ?

+++ {"slideshow": {"slide_type": "slide"}}

* un seul neurone
* une fonction d'activation $F$, de type "tout ou rien" (seuil d'activation)
* Un signal d'entrée binaire $x_j = 0 $ ou $1$

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

```{figure} media/artificial_neuron_lien_bio.png
---
alt: media/artificial_neuron_lien_bio.png
width: 900px
align: center
---
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

#### Perceptron (Rosenblatt, 1957)

Premier algorithme d'apprentissage supervisé dont l'objectif est de distinguer 2 classes. L'unique couche du perceptron contient un neurone et calcule une combinaison linéaire $w_0 + \sum w_j x_j$ des signaux $x_1,...x_n$. On applique alors une fonction d'activation et on transmet en sortie le résultat

```{figure} media/perceptron.png
---
alt: Perceptron (https://commons.wikimedia.org/wiki/File:Perceptron_moj.png)
width: 700px
align: center
---
```

+++ {"slideshow": {"slide_type": "slide"}}

**Les fonctions d'activation (les plus simples)**

* L'identité (pour les problèmes de régression)
* Seuil d'activation (pour les problèmes de classification)
* Intervalles d'activation (si on a plus de 2 classes)
* Autres fonctions plus complexe, on en mentionnera quelques unes un peu plus tard

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

#### Perceptron multi-classe 

Une proposition parmi d'autres si on a $C>2$ classes.

* pas 1 mais $C$ neurones dans la couche de sortie (un neurone par classe)
* les $n$ neurones de la couche d'entrée sont tous connectés à chacun des neurones de sortie
* on a donc $n\times C$ poids de connexions, notés $w_j^c$
* les classes sont supposées non chevauchantes

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

## Le deep learning, un *réseau* de neurones

**Réseau de neurones artificiel** : un ensemble de modèles (classifieurs par ex.) dont l'architecture est articulée en *couches*. Les couches sont composées de différentes transformations non linéaires: des couches de perceptron, des convolutions ou autres transformations.

Ces architectures manipulent des données
brutes, sans utiliser d'attributs: les attributs sont en quelque sorte
appris par le réseau.

**Le réseau le plus simple est un réseau entièrement connecté, mais il est difficile à optimiser !**

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

**Poids** 

*  Quand le réseau
fait une prédiction, les valeurs des données brutes (pixels) sont
propagées dans le réseau jusqu'à donner le résultat en sortie.
* Les **poids** sont les paramètres appris par le réseau, associés aux neurones et optimisés lors de l'entraînement. 
* L'entrainement se fait selon la technique de la descente de gradient (vous le verrez en IAS en L3)

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

**Spécificité des réseaux de neurones**

* les valeurs d'entrée sont les données brutes et non plus les attributs.
* les poids deviennent en quelque sorte des attributs de nos images

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Perceptron multi-couche (MLP, multi layer perceptron)

```{figure} media/MLP.png
---
alt: media/MLP.png
width: 900px
align: center
---
```

+++ {"slideshow": {"slide_type": "slide"}}

On en a utilisé un lors du CM5 et avec [Marcelle](https://demos.marcelle.dev/webcam-mlp-vs-knn).
* Les couches intermédiaires sont appelées couches cachées (*hidden layers*).
* Chaque neurone d'une couche intermédiaire ou de la couche de sortie, recoit en entrée les sorties des neurones de la couche précédente
* Il n'y a pas de cycle
* La complexité vient des fonctions d'activation qui peuvent etre complexes

+++ {"slideshow": {"slide_type": "slide"}}

###  Réseaux de neurones convolutionnels - Convolutional neural network, CNN ou ConvNet

```{figure} media/CNN.png
---
alt: media/CNN.png
width: 1000px
align: center
---
```
* utilisé pour le traitement d'images
* multiples perceptrons empilés
* chaque perceptron traite une sous-tache

+++ {"slideshow": {"slide_type": "slide"}}

#### Traitement convolutif

Définition lors du [CM6](../Semaine6/CM6.md).

#### Organisation des CNN

Succession de plusieurs type de couches:
* Couche de convolution
* Couche d'activation
* Couche de compression
* Couche entièrement connectée

+++ {"slideshow": {"slide_type": "slide"}}

##### Couche de convolution

* Neurones de traitement : analyse une sous-image selon un mode de convolution défini
* Trois hyperparamètres : la profondeur, le pas et la marge.
    * Profondeur : nombre de noyaux de convolution
    * Pas : décalage des noyaux de convolution (avec ou sans chevauchement)
    * Marge : la taille des marges de l'image pour gérer les extrémités lors des calculs de convolution
* Dans une même couche, les valeurs des paramètres des neurones sont tous identiques.

+++ {"slideshow": {"slide_type": "slide"}}

##### Couche de d'activation ou de correction

* Opère une fonction d'activation sur les sorties des neurones précédents. Par rapport aux fonctions d'activation précédentes, on notera qu'il existe aussi:

* La correction ReLU (abréviation de Unité Linéaire Rectifiée) : $f(x)=\ max(0,x)$. On est linéaire une fois activé.
* La correction par la fonction sigmoïde $f(x)=(1+e^{-x})^{-1}$.

+++ {"slideshow": {"slide_type": "slide"}}

##### Couche de compression

* Neurones de pooling : regroupent les sorties de convolution
* forment une image intermédiaire qui servira de nouvelle "image" d'analyse
* Le pooling permet de réduire la dimension des images intermédiaires
* réduit le risque de surapprentissage

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

##### Couche entièrement connectée (*FC, fully connected*)

En fin de réseau, on utilisera des couches entièrement connectées aux sorties de la couche précédente. Leurs fonctions d'activations peuvent être calculées avec une multiplication matricielle et permettent *in fine* de classifier nos images.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Exemple d'une architecture connexioniste

Exemple d'un réseau de
neurone profond, sur [le
site](https://adamharley.com/nn_vis/cnn/2d.html) créé par Adam
Harley de l'Université Carnegie Mellon. Il présente un réseau
convolutionnel pour la reconnaissance de chiffres manuscrits. 

```{figure} https://adamharley.com/nn_vis/images/convnet_flat_480.png
---
alt: convnet exemple
width: 900px
align: center
---
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Region-based CNN (2013)

On peut aussi spécialiser régionalement notre analyse. On parle alors de R-CNN (region based CNN, à ne pas confondre avec les RNN pour recursive neural networks -- hors programme).

```{figure} ../Semaine6/media/pedestrian.png
---
alt: R-CNN et voitures autonomes
width: 600px
align: center
---

```

+++ {"slideshow": {"slide_type": "slide"}}

* On identifie les régions d'interet (*ROI, regions of interest*) en utilisant un mécanisme de recherche sélective
* Chaque ROI est un ensemble de rectangles (idéalement, entourant nos objets)
* Ensuite, chaque ROI est donné comme entrée d'un réseau de neurone puis est classifié

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

**RCNN en python**

* Il existe des réseaux pré-entrainés, capables de reconnaitre de nombreux objets de la vie quotidienne.
* C'est très facile à utiliser, mais ca ne vous dit pas comment ils ont été entrainés ni construits !
* La librairie OpenCV contient le réseau YOLO qui est entrainé
* cvlib (*computer vision library*) contient aussi un réseau préentrainé

* 
```{figure} media/yolo.png
---
alt: exemple de RCNN
width: 700px
align: center
---
```

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Principales différences entre le deep et le machine learning

1) La complexité des modèles;
2) La quantité de données requise est plus importante pour le deep learning (en lien avec le point 1);
3) Ces réseaux profonds comptent beaucoup de paramètres (neurones) à
optimiser. Ils nécessitent des infrastructures spéciales pour les
entraîner: de puissantes cartes graphiques (GPU), comme celles
utilisées pour les jeux vidéos, qui permettent de paralléliser les
calculs.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Une révolution du deep learning : la vision par ordinateur

#### Jeu de données de référence : ImageNet
* Collaboration entre l’université de Stanford et celle de Princeston.
* 15 millions d’images et 22.000 catégories.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

#### ImageNet Large Scale Visual Recognition Challenge (ILSVRC) : 
* De 2010 à 2017:
    * 1 000 images par catégorie,
    * 50 000 images de validation,
    * 150 000 images de test.
* Objectif : faire avancer la recherche dans le domaine de la vision par
ordinateur
* ImageNet continue à être utilisé dans de nombreux projets comme jeu de test.

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

#### Quelques exemples connus de CNN qui ont remporté l'ILSVRC

**Inception network de Google** 
```{figure} https://www.researchgate.net/profile/Bo-Zhao-67/publication/312515254/figure/fig3/AS:489373281067012@1493687090916/nception-module-of-GoogLeNet-This-figure-is-from-the-original-paper-10.png
---
alt: https://www.researchgate.net/profile/Bo-Zhao-67/publication/312515254/figure/fig3/AS:489373281067012@1493687090916/nception-module-of-GoogLeNet-This-figure-is-from-the-original-paper-10.png
width: 600px
align: center
---
```

* nouveau concept : module Inception

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

```{figure} media/inception.png
---
alt: Module inception de google
width: 500px
align: center
---
```

* 22 couches

+++ {"slideshow": {"slide_type": "slide"}}

**ResNet de microsoft research Asia**

```{figure} media/bloc_resnet.png
---
alt: media/bloc_resnet.png
width: 400px
align: center
---
```

* nouveau concept : introduire une connexion de raccourci qui saute une ou plusieurs couches.

+++ {"slideshow": {"slide_type": "slide"}}

```{figure} media/resnet.png
---
alt: media/resnet.png
width: 1000px
align: center
---
```

* 152 couches !

+++ {"editable": true, "slideshow": {"slide_type": "slide"}, "tags": []}

### Conclusions sur le Deep Learning

Aujourd’hui, pour reconnaître des images, on n’extrait plus les caractéristiques à la main. On utilise de très grands réseaux de neurones (deep learning) qui :

- apprennent automatiquement les caractéristiques pertinentes
- contiennent parfois des millions voire milliards de paramètres
- nécessitent des GPU pour être entraînés

On peut aussi utiliser le transfert d’apprentissage : on prend un modèle déjà entraîné sur des millions d’images et on l’adapte à notre problème.

+++ {"editable": true, "heading_collapsed": true, "slideshow": {"slide_type": "slide"}, "tags": []}

## Perspectives

* **CM8**: Biais dans les données et impact écologique + sociétal de l'intelligence artificielle
