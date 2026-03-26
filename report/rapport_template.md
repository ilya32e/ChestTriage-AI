# Rapport Final - Systeme d'aide au tri radiologique

## 1. Probleme

- Contexte medical et metier
- Objectif du systeme d'aide au tri
- Problematique IA
- Hypotheses et perimetre

## 2. Donnees

- Dataset principal: ChestMNIST / ChestMNIST+
- Dataset multimodal texte: IU X-Ray / OpenI
- Variante complementaire: NIH Chest X-rays metadata/annotations
- Contraintes d'acces, limites et biais

## 3. Analyse exploratoire des donnees

- Distribution des classes
- Desequilibre des labels
- Exemples visuels
- Co-occurrences des pathologies
- Premiers constats sur le texte radiologique

## 4. Preparation des donnees

- Pretraitement image
- Normalisation
- Augmentation de donnees
- Encodage des labels
- Preparation et vectorisation du texte
- Gestion du desequilibre
- Strategie anti-fuite de donnees

## 5. Modelisation supervisee

- CNN simple entraine depuis zero
- CNN pre-entraine avec transfer learning
- Vision Transformer / TinyViT
- Justification des choix

## 6. Detection d'anomalies par AE / VAE

- Architecture retenue
- Fonction de perte
- Protocole d'entrainement
- Definition du score d'anomalie
- Analyse qualitative des cas atypiques

## 7. Modelisation multimodale image + compte-rendu

- Representation image
- Representation texte
- Strategie de fusion
- Comparaison image seule / texte seul / multimodal
- Robustesse quand une modalite manque

## 8. Evaluation

- ROC-AUC
- F1-score
- PR-AUC / Average Precision
- Analyse par classe
- Courbes et visualisations

## 9. Tracking MLflow et tracabilite

- Organisation des experiences
- Hyperparametres testes
- Runs compares
- Meilleur run retenu
- Correspondance entre run trace et modele deploye

## 10. Demonstrateur applicatif

- Architecture Streamlit
- Fonctionnalites
- Parcours utilisateur
- Limites du prototype

## 11. Analyse critique

- Resultats principaux
- Limites
- Erreurs frequentes
- Robustesse et generalisation
- Apport de la detection d'anomalies
- Apport de la multimodalite
- Compromis performance / cout de calcul

## 12. Conclusion et perspectives

- Modele recommande
- Interet de l'AE/VAE
- Interet de la multimodalite
- Limites experimentales
- Perspectives d'amelioration
