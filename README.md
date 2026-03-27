# Projet : Federated MNIST (éducatif)

Ce dépôt contient un exemple pédagogique de Federated Learning appliqué au jeu de données MNIST. Il vise à illustrer, de façon simple et reproductible, les étapes clefs d'un pipeline FL implémenté manuellement avec PyTorch (sans frameworks FL externes).

## Explication du projet

Le but est d'apprendre les principes du Federated Learning en comparant un entraînement centralisé classique à une simulation fédérée :

- Entraînement centralisé (baseline) : un modèle est entraîné sur l'ensemble des données d'entraînement et évalué sur le jeu de test.\
- Simulation clients (IID) : le jeu d'entraînement est divisé en plusieurs sous-ensembles IID (3–5 clients). Chaque client possède localement une portion des données.\
- Entraînement local : chaque client reçoit le modèle global, effectue une (courte) mise à jour locale (SGD, CrossEntropy) puis renvoie ses poids.\
- Agrégation (FedAvg) : le serveur central agrège les poids des clients en faisant une moyenne pondérée par la taille des jeux locaux.\
- Boucle FL : répéter l'envoi, l'entraînement local et l'agrégation pendant plusieurs rounds, puis évaluer le modèle global sur le jeu de test.\

Le notebook `federated_mnist.ipynb` contient :

- Imports et préparation des données (MNIST, normalisation).\
- Définition d'un modèle léger (MLP) pour garder l'exemple rapide à exécuter.\
- Baseline centralisée pour comparaison.\
- Fonction de partition IID, routine d'entraînement local, fonction d'évaluation.\
- Implémentation manuelle de FedAvg et boucle fédérée (3–5 rounds, paramètres courts pour un runtime réduit).\

## Contenu principal

- `federated_mnist.ipynb` : Notebook Jupyter implémentant le pipeline décrit ci-dessus.\
- `requirements.txt` : Dépendances Python minimales (torch, torchvision, jupyter, matplotlib, numpy).\
- `Prop_Sujet_Vinith.pdf` : Proposition / sujet fourni.\
- `fed.pdf` : PDF additionnel présent dans le dépôt.\
- `archi.pdf` : Diagramme d'architecture (si présent). Si `archi.pdf` n'apparaît pas, copiez-le à la racine du dépôt pour l'inclure.

## Exécution rapide

1. Créez et activez un environnement virtuel (recommandé) :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```

3. Lancez Jupyter et ouvrez le notebook :

```bash
jupyter notebook
# puis ouvrez federated_mnist.ipynb dans l'interface
```

Le notebook est paramétré pour tourner rapidement sur CPU (nombre d'époques et rounds réduits). Augmentez ces paramètres ou utilisez un GPU pour des résultats plus précis.

## Notes et améliorations possibles

- Pour des expériences plus proches de la réalité : augmenter le nombre d'époques locales, le nombre de rounds, utiliser un CNN, simuler des splits non-IID, ajouter des stratégies de sélection de clients et des schémas d'optimisation plus avancés.\
- `archi.pdf` : si vous me fournissez ce fichier, je peux l'ajouter au dépôt et mettre à jour le README pour l'afficher.

## Git

Le dépôt a été initialisé et un commit initial a été fait. Pour pousser vers un remote, ajoutez un `remote` puis exécutez :

```bash
git remote add origin <URL>
git push -u origin main
```

## Suite

Dites-moi si vous souhaitez que je :
- pousse automatiquement vers un remote (fournissez l'URL et la méthode d'authentification),
- ajoute `archi.pdf` si vous le fournissez,
- ou améliore le notebook (ex: graphique de l'évolution de l'acc, CNN, non-IID).
