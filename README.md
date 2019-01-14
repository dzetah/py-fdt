# py-fdt
> TP de fouille de texte. M2 WIC, UE Fouille de Texte / Stanyslas Bres.

## Description du modèle

Le modèle conçu utilise deux représentations simultanées :
  1. Plongements lexicaux.
  2. Représentation creuse avec vectorisation _TfIdf_.

Ces deux représentations sont ensuite _concaténées_ puis envoyées dans une couche cachée **Dense** de 16 cellules avant de produire la sortie.

### Tokenization

J'ai récupéré une liste de stopwords français disponible en ligne, que j'ai affinée afin de conserver les négations (_ne_, _pas_, _mais_, etc.).

### Plongements lexicaux

La première branche consiste à utiliser un word embedding pré-entrainé allié à des cellules récurrentes GRU (sorte de LSTM plus rapides mais moins précises).

La tokenization utilise les mots complets (car c'est sous cette forme qu'ils sont dans l'embedding), sans la ponctuation ni les symboles et les nombres. Je filtre aussi les tokens avec ma liste de stopwords pour enlever les mots qui ne portent pas de sens.

la vectorization des plongements lexicaux consiste à renvoyer l'index du mot dans le word embedding afin que le _layer_ **Embedding** fasse correspondre cet index à la représentation vectorielle. Si le mot n'est pas présent dans le word embedding, on ne l'utilise pas.

Afin de disposer de séquences de taille fixe, on utilise l'utilitaire `pad_sequences` fourni par keras qui va créer du padding sur les séquences trop courtes si nécessaire.

Ensuite, j'utilise le layer **GRU** qui est une couche récurrente prenant en entrée chaque token vectorisé et itèrant pour toute la phrase. Cette couche à l'avantage de prendre chaque token de manière séquentielle et donc de préserver le "contexte" des tokens précédents.

### Représentation creuse

La seconde branche d'entrée utilise un **TfidfVectorizer** et la tokenization est légèrement différente car elle utilise les lemmes et non le texte brut, ceci afin de rapprocher les mots dont le sens est le même mais dont la déclinaison pourrait les différencier (verbes conjugés, féminin/masculin, etc.).

## Observations

Les hyper paramètres ayant le plus d'effet sont le nombre de couches cachées et les fonctions d'activation des couches denses (la fonction _relu_ est très efficace). Avec les plongements lexicaux, la taille maximale des séquence est un hyper paramètre important ainsi que le batch_size (appliqué entre chaque étape du RNN).

Sur une moyenne de 5 runs le modèle donne une précision d'environ 69-70% pour une durée de traitement de 500s sur un macbook pro core i5 (sans carte graphique).

Nul doute que les temps de calculs seront bien meilleurs sur un PC équipé d'une carte graphique dédiée.

## Conclusion

Outre les résultats purs du modèle que j'ai eu du mal à améliorer, j'ai pu mettre en oeuvre différentes techniques de vectorisation : **Word Embeddings**, **CountVectorizer**, **TfidfVectorizer**.

Il était intéressant de faire varier les hyper paramètres et de tenter de comprendre les causes des variations des résultats.

Ce projet m'a également permi de bien mieux comprendre le fonctionnement du machine learning et deep learning par la pratique, car les concepts associés restaient assez abstraits pour moi.

Comme améliorations possibles, il aurait été possible de rajouter un dictionnaire de mots polarisés et ajouter cette valeut de polarité comme paramètre au modèle. Et dans le même temps, vérifier que les mots polarisés ne sont pas précédés d'une négation...

## Installation

Les librairies suivantes sont nécessaires :
- python `3.6`
- keras `>= 2.2.4`
- spacy `>= 2.0.12`
- gensim `>= 3.4.0`
- scikit-learn `>= 0.20.1`
- pandas `>= 0.23.4`

Il est recommandé d'utiliser [anaconda](https://anaconda.com) pour gérer l'environnement de développement.
> Il est possible de charger cet environnement depuis le fichier `condaenv.yml` avec la commande `conda env create -f condaenv.yml`.

Il faut égallement installer un des word embeeding français depuis l'[adresse suivante](http://fauconnier.github.io/#data).

Afin de permettre à SpaCy d'analyser nos phrases en français, il faut télécharger le modèle correspondant avec la commande suivante :

```bash
$ python -m spacy download fr
```

### Modification de l'environnement

Si vous souhaitez apporter des modifications à l'environnement _Conda_, n'oubliez pas de mettre à jour le fichier `condaenv.yml` à l'aide du script `conda-export-env.sh`
