# Systeme d'aide au tri radiologique

Projet de deep learning realise par Mouradi Iliasse et Mahi Karima.

L'idee du sujet etait de construire un petit systeme de tri radiologique autour de radios thoraciques, pas juste un classifieur. Du coup on a separe le projet en plusieurs blocs :

- une partie supervision sur images
- une partie detection d'anomalies avec autoencodeur
- une partie multimodale
- le suivi des experiences avec MLflow
- une demo Streamlit pour tester le tout

## Ce qu'on a garde dans la version finale

Pour la partie supervisee, on compare bien les 3 familles demandees dans le sujet :

- `SimpleCNN` entraine depuis zero
- `ResNet18` en transfer learning
- `TinyViT` pour la partie Transformer

Configs :

- [`configs/supervised/simple_cnn.yaml`](configs/supervised/simple_cnn.yaml)
- [`configs/supervised/resnet18_transfer.yaml`](configs/supervised/resnet18_transfer.yaml)
- [`configs/supervised/tiny_vit.yaml`](configs/supervised/tiny_vit.yaml)

Pour l'anomalie, on a garde un autoencodeur convolutionnel :

- [`configs/anomaly/conv_autoencoder.yaml`](configs/anomaly/conv_autoencoder.yaml)
- [`src/radiology_triage/models/autoencoder.py`](src/radiology_triage/models/autoencoder.py)

Pour le multimodal, dans la version finale du projet, on utilise NIH et pas IU X-Ray. On compare :

- `image_only`
- `metadata_only`
- `fusion`

Configs :

- [`configs/multimodal/image_only_nih_metadata.yaml`](configs/multimodal/image_only_nih_metadata.yaml)
- [`configs/multimodal/text_only_nih_metadata.yaml`](configs/multimodal/text_only_nih_metadata.yaml)
- [`configs/multimodal/fusion_nih_metadata.yaml`](configs/multimodal/fusion_nih_metadata.yaml)

## Donnees utilisees

### ChestMNIST

On l'utilise pour :

- la classification supervisee
- la detection d'anomalies

Sources :

- site officiel : [https://medmnist.com/](https://medmnist.com/)
- repo Git : [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST)
- article : Yang et al., 2023

Le chargement passe par :

- [`src/radiology_triage/data/chestmnist.py`](src/radiology_triage/data/chestmnist.py)

### NIH Chest X-rays

On l'utilise pour la partie multimodale.

Sources :

- fiche dataset : [https://www.innovatiana.com/fr/datasets/nih-chest-x-rays](https://www.innovatiana.com/fr/datasets/nih-chest-x-rays)
- article : Wang et al., 2017

Fichiers prepares localement :

- [`data/nih/nih_multimodal_metadata.csv`](data/nih/nih_multimodal_metadata.csv)
- [`data/nih/import_summary.json`](data/nih/import_summary.json)

Point important :

- la partie texte n'est pas un vrai compte-rendu radiologique libre
- dans notre pipeline, on utilise un champ `metadata_text` reconstruit a partir des metadonnees et annotations NIH

## Arborescence rapide

```text
app/
  streamlit_app.py

configs/
  anomaly/
  multimodal/
  supervised/

report/
  main.tex
  main.pdf

scripts/
  start_mlflow.ps1
  start_final_project.ps1
  train_supervised.py
  train_anomaly.py
  train_multimodal.py

src/
  radiology_triage/
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si besoin pour MedMNIST :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_medmnist_from_git.ps1
```

Si besoin pour PyTorch GPU :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_gpu_torch.ps1
```

## Recuperation des donnees

### ChestMNIST

```powershell
python .\scripts\download_medmnist_official.py --root data/medmnist --sizes 64 128 224
```

### NIH

```powershell
python .\scripts\import_nih_kagglehub.py --output-dir data/nih
```

## Lancer les trainings

### Supervise

```powershell
python .\scripts\train_supervised.py --config .\configs\supervised\simple_cnn.yaml
python .\scripts\train_supervised.py --config .\configs\supervised\resnet18_transfer.yaml
python .\scripts\train_supervised.py --config .\configs\supervised\tiny_vit.yaml
```

### Anomalie

```powershell
python .\scripts\train_anomaly.py --config .\configs\anomaly\conv_autoencoder.yaml
```

### Multimodal

```powershell
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only_nih_metadata.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only_nih_metadata.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion_nih_metadata.yaml
```

### Calibration des seuils

```powershell
python .\scripts\calibrate_supervised_thresholds.py
```

Le fichier genere est :

- [`artifacts/supervised/simple_cnn/class_thresholds.json`](artifacts/supervised/simple_cnn/class_thresholds.json)

## MLflow

Pour lancer MLflow :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_mlflow.ps1
```

Adresse locale :

- `http://127.0.0.1:5000`

## Demo

Pour lancer Streamlit :

```powershell
streamlit run .\app\streamlit_app.py
```

Adresse locale :

- `http://127.0.0.1:8501`

Dans la demo on peut :

- charger une radio thoracique
- voir les predictions supervisees
- voir le score d'anomalie
- tester la branche multimodale si on ajoute un texte

## Pipeline complet

Pour relancer le pipeline principal :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_final_project.ps1
```

Pour suivre l'avancement :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\monitor_final_project.ps1
```

## Checkpoints charges par la demo

- supervision : [`artifacts/supervised/simple_cnn/best_model.pt`](artifacts/supervised/simple_cnn/best_model.pt)
- seuils : [`artifacts/supervised/simple_cnn/class_thresholds.json`](artifacts/supervised/simple_cnn/class_thresholds.json)
- anomalie : [`artifacts/anomaly/conv_autoencoder/best_autoencoder.pt`](artifacts/anomaly/conv_autoencoder/best_autoencoder.pt)
- multimodal `image_only` : [`artifacts/multimodal/nih_image_only/best_multimodal_model.pt`](artifacts/multimodal/nih_image_only/best_multimodal_model.pt)
- multimodal `metadata_only` : [`artifacts/multimodal/nih_metadata_only/best_multimodal_model.pt`](artifacts/multimodal/nih_metadata_only/best_multimodal_model.pt)
- multimodal `fusion` : [`artifacts/multimodal/nih_fusion/best_multimodal_model.pt`](artifacts/multimodal/nih_fusion/best_multimodal_model.pt)

## Reproductibilite

On a essaye de garder quelque chose de propre :

- seed fixe `42`
- separation `train / val / test`
- sauvegarde du meilleur modele
- suivi des runs dans MLflow
- coherence entre les checkpoints traces et ceux charges dans la demo

## Config machine

Machine de reference :

- Windows 11
- Python 3.11.9
- PyTorch 2.11.0
- CUDA 12.8
- GPU NVIDIA RTX 2070 Max-Q
- 16 Go de RAM

## Temps observes

Temps observes a partir des runs finaux traces dans MLflow :

- `SimpleCNN` : 20 min 47 s
- `ResNet18 transfer` : 19 min 07 s
- `TinyViT` : 6 min 37 s
- `Conv Autoencoder` : 4 min 24 s
- `Image only (NIH)` : 44 min 16 s
- `Metadata only (NIH)` : 5 min 39 s
- `Fusion image + metadata (NIH)` : 1 h 17 min au total

A noter pour le `fusion` :

- il y a eu environ `50 min 30 s` de train initial
- puis `26 min 55 s` de finalisation/evaluation du checkpoint
- cette reprise etait necessaire a cause d'une limite Windows sur la memoire partagee pendant l'evaluation de robustesse

Si on additionne tout ce qui a servi a produire les artefacts finaux sur cette machine, on est autour de `2 h 58 min`, sans compter le telechargement des donnees, l'EDA et l'ouverture des interfaces.

En pratique, sans la reprise du `fusion`, on serait plutot autour de `2 h 31 min`.
