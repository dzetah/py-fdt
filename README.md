# py-fdt
> TP de fouille de texte. M2 WIC, UE Fouille de Texte / Stanyslas Bres.

## Installation

Les librairies suivantes sont nécessaires :
- python `3.6`
- keras `>= 2.2.4`
- spacy `>= 2.0.12`
- gensim `>= 3.4.0`
- scikit-learn `>= 0.20.1`
- ntlk `>= 3.3.0`
- pandas `>= 0.23.4`

Il est recommandé d'utiliser [anaconda](https://anaconda.com) pour gérer l'environnement de développement.
> Il est possible de charger cet environnement depuis le fichier `condaenv.yml` avec la commande `conda env create -f condaenv.yml`.

Il faut maintenant installer les word embeedings français depuis l'[adresse suivante](http://fauconnier.github.io/#data)

Afin de permettre à SpaCy d'analyser nos phrases en français, il faut télécharger le modèle correspondant avec la commande suivante :

```bash
$ python -m spacy download fr
```

### Modification de l'environnement

Si vous souhaitez apporter des modifications à l'environnement _Conda_,

n'oubliez pas de mettre à jour le fichier `condaenv.yml` à l'aide du script `conda-export-env.sh`
