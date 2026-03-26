# Systeme d'aide au tri radiologique

Pipeline de deep learning pour radiographies thoraciques combinant classification supervisee multi-label, detection d'anomalies par autoencodeur, modelisation multimodale image + texte, suivi experimental avec MLflow et demonstrateur Streamlit.

## Sommaire

- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalites](#fonctionnalites)
- [Jeux de donnees](#jeux-de-donnees)
- [Architecture du depot](#architecture-du-depot)
- [Environnement](#environnement)
- [Installation](#installation)
- [Docker](#docker)
- [Execution](#execution)
- [Tracking et artefacts](#tracking-et-artefacts)
- [Application](#application)
- [Checkpoints attendus](#checkpoints-attendus)
- [Rapport](#rapport)
- [Reproductibilite](#reproductibilite)
- [Configuration materielle](#configuration-materielle)

## Vue d'ensemble

Le projet couvre les composantes suivantes :

- classification supervisee sur `ChestMNIST`
- detection d'anomalies par autoencodeur convolutionnel
- comparaison multimodale `image_only`, `text_only`, `fusion`
- suivi des runs et des artefacts avec `MLflow`
- application locale `Streamlit`

Le depot est organise autour d'un noyau Python modulaire dans [`src/radiology_triage`](src/radiology_triage), de configurations YAML dans [`configs`](configs), de scripts d'execution dans [`scripts`](scripts), d'une application dans [`app`](app) et d'un rapport LaTeX dans [`report`](report).

## Fonctionnalites

- `SimpleCNN` entraine depuis zero pour la baseline supervisee
- `ResNet18` en transfer learning
- `TinyViT` comme variante Transformer compacte
- `ConvAutoencoder` pour la detection d'images atypiques ou hors distribution
- pipeline multimodal avec encodeur image, encodeur texte `GRU` et fusion intermediaire
- sauvegarde automatique des meilleurs checkpoints
- export des metriques, figures et configurations dans `MLflow`
- application d'inference regroupant classification, anomalie et multimodalite

## Jeux de donnees

### Classification supervisee et anomalie

| Dataset | Usage | Source |
|---|---|---|
| `ChestMNIST` | classification supervisee, detection d'anomalies | Site officiel MedMNIST: <https://medmnist.com/> |

Reference scientifique :

- Yang et al., *MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification*, Scientific Data, 2023  
  <https://doi.org/10.1038/s41597-022-01721-8>

Le chargement est realise via le package `medmnist` dans [`src/radiology_triage/data/chestmnist.py`](src/radiology_triage/data/chestmnist.py).
Le projet est aligne sur le depot Git officiel MedMNIST : <https://github.com/MedMNIST/MedMNIST>.

### Multimodalite image + texte

| Dataset | Usage | Source |
|---|---|---|
| `IU X-Ray / OpenI` | branche multimodale principale `image + report_text` | Hugging Face dataset mirror: <https://huggingface.co/datasets/dz-osamu/IU-Xray> |

Reference scientifique :

- Demner-Fushman et al., *Preparing a collection of radiology examinations for distribution and retrieval*, JAMIA, 2016  
  <https://doi.org/10.1093/jamia/ocv080>

Le jeu prepare localement est materialise par :

- [`data/iu_xray/iu_xray_multimodal.csv`](data/iu_xray/iu_xray_multimodal.csv)
- [`data/iu_xray/import_summary.json`](data/iu_xray/import_summary.json)

Note de methode :

- les labels IU X-Ray utilises dans la preuve de concept sont des `weak labels` derives du texte

### Variante complementaire NIH

| Dataset | Usage | Source |
|---|---|---|
| `NIH Chest X-rays` | variante complementaire `image + metadata_text` | Kaggle dataset `nih-chest-xrays/data` via `kagglehub`, fiche dataset: <https://www.innovatiana.com/fr/datasets/nih-chest-x-rays> |

Reference scientifique :

- Wang et al., *ChestX-ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases*, CVPR, 2017  
  <https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html>

Le jeu prepare localement est materialise par :

- [`data/nih/nih_multimodal_metadata.csv`](data/nih/nih_multimodal_metadata.csv)
- [`data/nih/import_summary.json`](data/nih/import_summary.json)

Telechargement reproductible retenu dans le projet :

```python
import kagglehub

path = kagglehub.dataset_download("nih-chest-xrays/data")
print("Path to dataset files:", path)
```

Positionnement vis-a-vis du sujet :

- la contrainte stricte `image + compte-rendu` est satisfaite par `IU X-Ray / OpenI`
- la recommandation d'utiliser `NIH` est conservee via une branche complementaire `image + metadata_text`
- le depot garde donc a la fois une variante principale conforme a l'exigence texte libre et une variante complementaire conforme a la recommandation dataset

## Architecture du depot

```text
app/
  streamlit_app.py

configs/
  anomaly/
  multimodal/
  quickstart/
  supervised/

data/
  iu_xray/
  nih/
  medmnist/

report/
  main.tex
  main.pdf
  generated_results.tex

scripts/
  generate_eda.py
  import_iu_xray.py
  import_nih_kagglehub.py
  start_mlflow.ps1
  start_final_project.ps1
  train_anomaly.py
  train_multimodal.py
  train_supervised.py

src/
  radiology_triage/
    config.py
    data/
    models/
    training/
    utils/
```

## Environnement

Versions principales du projet :

- `Python 3.11.9`
- `PyTorch 2.11.0+cu128`
- `Torchvision 0.26.0+cu128`
- `MLflow 3.10.1`
- `Streamlit 1.53.1`

Dependances Python : [`requirements.txt`](requirements.txt)

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Installation explicite depuis le depot Git officiel MedMNIST :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_medmnist_from_git.ps1
```

Installation PyTorch CUDA sur machine NVIDIA :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_gpu_torch.ps1
```

## Docker

Une containerisation simple est fournie pour lancer la demo Streamlit et MLflow sans modifier l'architecture du depot.

Fichiers associes :

- [`Dockerfile`](Dockerfile)
- [`docker-compose.yml`](docker-compose.yml)
- [`.dockerignore`](.dockerignore)

Lancement des services :

```powershell
docker compose up --build streamlit mlflow
```

URI exposees :

- `http://127.0.0.1:8501` pour Streamlit
- `http://127.0.0.1:5000` pour MLflow

Arret des services :

```powershell
docker compose down
```

Notes d'usage :

- le `docker-compose.yml` monte le depot local dans `/workspace`
- les dossiers locaux [`data`](data), [`artifacts`](artifacts) et [`mlruns`](mlruns) restent donc reutilises tels quels
- l'image n'embarque pas les datasets ni les checkpoints par defaut, afin de garder un build compact
- cette configuration vise un usage local CPU et une execution reproductible de la demo

Exemples de commandes d'entrainement dans le conteneur :

```powershell
docker compose run --rm streamlit python scripts/train_supervised.py --config configs/supervised/resnet18_partial_finetune_224.yaml
docker compose run --rm streamlit python scripts/train_anomaly.py --config configs/anomaly/conv_autoencoder.yaml
docker compose run --rm streamlit python scripts/train_multimodal.py --config configs/multimodal/fusion.yaml
docker compose run --rm streamlit python scripts/calibrate_supervised_thresholds.py --checkpoint artifacts/supervised/resnet18_partial_finetune_224/best_model.pt
```

## Execution

### EDA

```powershell
python .\scripts\generate_eda.py --root data/medmnist --size 128 --output-dir artifacts/eda
```

### Telechargement officiel de ChestMNIST

```powershell
python -m pip install --upgrade git+https://github.com/MedMNIST/MedMNIST.git
python .\scripts\download_medmnist_official.py --root data/medmnist --sizes 64 128 224
```

Cette commande utilise l'API officielle du depot `MedMNIST/MedMNIST` et declenche le telechargement via `ChestMNIST(..., download=True, size=...)`. Un resume local est ecrit dans [`data/medmnist/download_summary.json`](data/medmnist/download_summary.json).

Equivalent officiel minimal :

```python
from medmnist import ChestMNIST

test_dataset = ChestMNIST(split="test", download=True, size=224)
```

Source Git retenue pour l'installation :

- <https://github.com/MedMNIST/MedMNIST.git>

Resultat local :

- [`data/medmnist`](data/medmnist)
- [`data/medmnist/download_summary.json`](data/medmnist/download_summary.json)

### Telechargement IU X-Ray / OpenI

```powershell
python .\scripts\import_iu_xray.py --output-dir data/iu_xray
```

Le script telecharge les fichiers du mirror configure dans le projet, extrait les images puis genere le CSV multimodal principal.

Source de telechargement utilisee par le projet :

- mirror Hugging Face : <https://huggingface.co/datasets/dz-osamu/IU-Xray>

Equivalent Python minimal :

```python
from huggingface_hub import hf_hub_download

repo_id = "dz-osamu/IU-Xray"

train_jsonl = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="train.jsonl")
val_jsonl = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="val.jsonl")
test_jsonl = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="test.jsonl")
image_zip = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="image.zip")
```

Le script [`import_iu_xray.py`](scripts/import_iu_xray.py) utilise ensuite ces fichiers pour extraire les images et construire le dataset local exploitable par le pipeline multimodal.

Resultat local :

- [`data/iu_xray/images`](data/iu_xray/images)
- [`data/iu_xray/iu_xray_multimodal.csv`](data/iu_xray/iu_xray_multimodal.csv)
- [`data/iu_xray/import_summary.json`](data/iu_xray/import_summary.json)

### Telechargement NIH Chest X-rays via KaggleHub

```powershell
python .\scripts\import_nih_kagglehub.py --output-dir data/nih
```

Equivalent minimal :

```python
import kagglehub

path = kagglehub.dataset_download("nih-chest-xrays/data")
print("Path to dataset files:", path)
```

Le script du projet telecharge le dataset, construit le CSV [`data/nih/nih_multimodal_metadata.csv`](data/nih/nih_multimodal_metadata.csv) et ecrit un resume dans [`data/nih/import_summary.json`](data/nih/import_summary.json).

### Donnees locales ignorees par Git

Le contenu telecharge ou genere dans les dossiers suivants est ignore par Git et doit etre regenere localement :

- [`data/medmnist`](data/medmnist)
- [`data/iu_xray`](data/iu_xray)
- [`data/nih`](data/nih)

### Classification supervisee

```powershell
python .\scripts\train_supervised.py --config .\configs\supervised\simple_cnn.yaml
python .\scripts\train_supervised.py --config .\configs\supervised\resnet18_transfer.yaml
python .\scripts\train_supervised.py --config .\configs\supervised\resnet18_partial_finetune.yaml
python .\scripts\train_supervised.py --config .\configs\supervised\resnet18_partial_finetune_224.yaml
python .\scripts\train_supervised.py --config .\configs\supervised\tiny_vit.yaml
```

### Calibration des seuils supervises

```powershell
python .\scripts\calibrate_supervised_thresholds.py
```

Artefact genere par defaut :

- [`artifacts/supervised/resnet18_partial_finetune_224/class_thresholds.json`](artifacts/supervised/resnet18_partial_finetune_224/class_thresholds.json)

Le script retrouve automatiquement le checkpoint supervise principal via [`artifacts/deployment_manifest.json`](artifacts/deployment_manifest.json) puis calibre des seuils par classe sur le split de validation.

### Detection d'anomalies

```powershell
python .\scripts\train_anomaly.py --config .\configs\anomaly\conv_autoencoder.yaml
```

### Multimodalite IU X-Ray

```powershell
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion.yaml
```

### Multimodalite NIH metadata

```powershell
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only_nih_metadata.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only_nih_metadata.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion_nih_metadata.yaml
```

### Pipeline final

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_final_project.ps1
```

Le pipeline final enchaine aussi le recalcul de [`class_thresholds.json`](artifacts/supervised/resnet18_partial_finetune_224/class_thresholds.json) avant le lancement de la demo.

## Tracking et artefacts

Demarrage de l'interface MLflow :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_mlflow.ps1
```

URI locale par defaut :

- `http://127.0.0.1:5000`

Les artefacts generes sont stockes dans :

- [`artifacts/`](artifacts)
- [`mlruns/`](mlruns)

Chaque run journalise notamment :

- hyperparametres
- metriques de validation et de test
- duree d'execution exploitable dans l'export du rapport
- courbes d'entrainement
- figures par classe et matrices de confusion adaptees
- reconstructions et scores d'anomalie
- meilleur checkpoint
- traceabilite du checkpoint deploye via [`artifacts/deployment_manifest.json`](artifacts/deployment_manifest.json)
- seuils calibres versionnes a cote du checkpoint supervise principal

## Application

Application locale :

```powershell
streamlit run .\app\streamlit_app.py
```

URI locale par defaut :

- `http://127.0.0.1:8501`

L'application regroupe :

- prediction supervisee multi-label image seule en sortie principale
- chargement automatique des seuils calibres si [`class_thresholds.json`](artifacts/supervised/resnet18_partial_finetune_224/class_thresholds.json) est present
- fallback explicite sur `0.5` par classe sinon
- score d'anomalie avec seuil et decision binaire interpretable
- comparaison multimodale `image_only`, `text_only`, `fusion`
- affichage de la traceabilite MLflow quand le manifest de deploiement est disponible
- export local des resultats d'inference dans [`artifacts/exports`](artifacts/exports)
- Grad-CAM optionnel pour `SimpleCNN` et `ResNet18`, sur une classe cible selectionnable dans l'interface
- details techniques et chemins charges regroupes dans des sections repliables

## Checkpoints attendus

Chemins de reference utilises par la demo :

- supervision : [`artifacts/supervised/resnet18_partial_finetune_224/best_model.pt`](artifacts/supervised/resnet18_partial_finetune_224/best_model.pt)
- seuils supervises : [`artifacts/supervised/resnet18_partial_finetune_224/class_thresholds.json`](artifacts/supervised/resnet18_partial_finetune_224/class_thresholds.json)
- anomalie : [`artifacts/anomaly/conv_autoencoder/best_autoencoder.pt`](artifacts/anomaly/conv_autoencoder/best_autoencoder.pt)
- multimodal `image_only` : [`artifacts/multimodal/iu_xray_image_only/best_multimodal_model.pt`](artifacts/multimodal/iu_xray_image_only/best_multimodal_model.pt)
- multimodal `text_only` : [`artifacts/multimodal/iu_xray_text_only/best_multimodal_model.pt`](artifacts/multimodal/iu_xray_text_only/best_multimodal_model.pt)
- multimodal `fusion` : [`artifacts/multimodal/iu_xray_fusion/best_multimodal_model.pt`](artifacts/multimodal/iu_xray_fusion/best_multimodal_model.pt)

## Rapport

Le rapport academique est disponible dans [`report`](report) :

- source LaTeX : [`report/main.tex`](report/main.tex)
- PDF compile : [`report/main.pdf`](report/main.pdf)
- tableaux de resultats MLflow : [`report/generated_results.tex`](report/generated_results.tex)
- manifest de deploiement : [`artifacts/deployment_manifest.json`](artifacts/deployment_manifest.json)

Regeneration des tableaux :

```powershell
python .\scripts\export_report_tables.py
```

## Reproductibilite

Elements integres au pipeline :

- seed fixe `42`
- separation explicite `train / val / test`
- split patient-level pour la variante NIH
- sauvegarde du meilleur modele
- configuration resolue exportee avec chaque run
- chargement des checkpoints traces dans l'application

Fichiers de configuration :

- [`configs/supervised`](configs/supervised)
- [`configs/anomaly`](configs/anomaly)
- [`configs/multimodal`](configs/multimodal)

## Configuration materielle

Configuration de reference utilisee pour le projet :

| Composant | Valeur |
|---|---|
| OS | `Windows 11 Famille` (`10.0.26200`) |
| CPU | `Intel Core i7-10750H` |
| RAM | `16 Go` |
| GPU | `NVIDIA GeForce RTX 2070 with Max-Q Design` |
| VRAM | `~8 Go` |
| CUDA detecte par PyTorch | `12.8` |
