# Systeme d'aide au tri radiologique

Ce depot contient notre projet de deep learning sur des radiographies thoraciques.

Projet realise en collaboration par Mouradi Iliasse et Mahi Karima.

Le but est de combiner plusieurs briques dans un meme pipeline :

- une classification supervisee sur images
- une detection d'anomalies avec un autoencodeur
- une petite preuve de concept multimodale image + texte
- un suivi des experiences avec MLflow
- une demo Streamlit pour tester le systeme

## Ce qu'on a implemente

### Partie supervisee

On compare 3 architectures, comme demande dans le sujet :

- `SimpleCNN` entraine depuis zero
- `ResNet18` en transfer learning
- `TinyViT` pour la famille Transformer

Configs associees :

- [`configs/supervised/simple_cnn.yaml`](configs/supervised/simple_cnn.yaml)
- [`configs/supervised/resnet18_transfer.yaml`](configs/supervised/resnet18_transfer.yaml)
- [`configs/supervised/tiny_vit.yaml`](configs/supervised/tiny_vit.yaml)

### Detection d'anomalies

La partie anomalie repose sur un autoencodeur convolutionnel :

- [`configs/anomaly/conv_autoencoder.yaml`](configs/anomaly/conv_autoencoder.yaml)
- [`src/radiology_triage/models/autoencoder.py`](src/radiology_triage/models/autoencoder.py)

### Partie multimodale

Pour la preuve de concept multimodale, on compare :

- `image_only`
- `text_only`
- `fusion`

Configs associees :

- [`configs/multimodal/image_only.yaml`](configs/multimodal/image_only.yaml)
- [`configs/multimodal/text_only.yaml`](configs/multimodal/text_only.yaml)
- [`configs/multimodal/fusion.yaml`](configs/multimodal/fusion.yaml)

### Suivi et demo

- suivi experimental avec `MLflow`
- application locale avec `Streamlit`

## Jeux de donnees

### 1. ChestMNIST

Utilise pour :

- la classification supervisee
- la detection d'anomalies

Sources :

- site officiel : [https://medmnist.com/](https://medmnist.com/)
- depot Git : [https://github.com/MedMNIST/MedMNIST](https://github.com/MedMNIST/MedMNIST)
- article : Yang et al., 2023
  [https://doi.org/10.1038/s41597-022-01721-8](https://doi.org/10.1038/s41597-022-01721-8)

Le chargement passe par :

- [`src/radiology_triage/data/chestmnist.py`](src/radiology_triage/data/chestmnist.py)

### 2. IU X-Ray / OpenI

Utilise pour la partie `image + texte`.

Sources :

- mirror Hugging Face : [https://huggingface.co/datasets/dz-osamu/IU-Xray](https://huggingface.co/datasets/dz-osamu/IU-Xray)
- article : Demner-Fushman et al., 2016
  [https://doi.org/10.1093/jamia/ocv080](https://doi.org/10.1093/jamia/ocv080)

Fichiers prepares localement :

- [`data/iu_xray/iu_xray_multimodal.csv`](data/iu_xray/iu_xray_multimodal.csv)
- [`data/iu_xray/import_summary.json`](data/iu_xray/import_summary.json)

Important :

- dans cette branche, les labels sont des `weak labels` derives du texte

### 3. NIH Chest X-rays

Cette partie est gardee comme variante complementaire.

Sources :

- fiche dataset : [https://www.innovatiana.com/fr/datasets/nih-chest-x-rays](https://www.innovatiana.com/fr/datasets/nih-chest-x-rays)
- article : Wang et al., 2017
  [https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html](https://openaccess.thecvf.com/content_cvpr_2017/html/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.html)

Fichiers prepares localement :

- [`data/nih/nih_multimodal_metadata.csv`](data/nih/nih_multimodal_metadata.csv)
- [`data/nih/import_summary.json`](data/nih/import_summary.json)

## Structure du depot

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

Si besoin :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_medmnist_from_git.ps1
```

Pour PyTorch GPU :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_gpu_torch.ps1
```

## Telechargement des donnees

### ChestMNIST

```powershell
python .\scripts\download_medmnist_official.py --root data/medmnist --sizes 64 128 224
```

### IU X-Ray

```powershell
python .\scripts\import_iu_xray.py --output-dir data/iu_xray
```

### NIH

```powershell
python .\scripts\import_nih_kagglehub.py --output-dir data/nih
```

## Lancer les entrainements

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
python .\scripts\train_multimodal.py --config .\configs\multimodal\image_only.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\text_only.yaml
python .\scripts\train_multimodal.py --config .\configs\multimodal\fusion.yaml
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

## Demo Streamlit

Pour lancer la demo :

```powershell
streamlit run .\app\streamlit_app.py
```

Adresse locale :

- `http://127.0.0.1:8501`

L'application permet :

- de charger une radio thoracique
- d'obtenir les predictions supervisees
- d'afficher un score d'anomalie
- de tester la branche multimodale si on ajoute un texte

## Pipeline complet

Pour tout relancer d'un coup :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_final_project.ps1
```

Pour suivre l'avancement :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\monitor_final_project.ps1
```

## Checkpoints utilises par la demo

- supervision : [`artifacts/supervised/simple_cnn/best_model.pt`](artifacts/supervised/simple_cnn/best_model.pt)
- seuils : [`artifacts/supervised/simple_cnn/class_thresholds.json`](artifacts/supervised/simple_cnn/class_thresholds.json)
- anomalie : [`artifacts/anomaly/conv_autoencoder/best_autoencoder.pt`](artifacts/anomaly/conv_autoencoder/best_autoencoder.pt)
- multimodal `image_only` : [`artifacts/multimodal/iu_xray_image_only/best_multimodal_model.pt`](artifacts/multimodal/iu_xray_image_only/best_multimodal_model.pt)
- multimodal `text_only` : [`artifacts/multimodal/iu_xray_text_only/best_multimodal_model.pt`](artifacts/multimodal/iu_xray_text_only/best_multimodal_model.pt)
- multimodal `fusion` : [`artifacts/multimodal/iu_xray_fusion/best_multimodal_model.pt`](artifacts/multimodal/iu_xray_fusion/best_multimodal_model.pt)

## Reproductibilite

Quelques points importants :

- seed fixe `42`
- separation `train / val / test`
- sauvegarde du meilleur modele
- tracking MLflow
- coherence entre les checkpoints traces et ceux charges dans la demo

## Configuration utilisee

Machine de reference :

- Windows 11
- Python 3.11.9
- PyTorch 2.11.0
- CUDA 12.8
- GPU NVIDIA RTX 2070 Max-Q
- 16 Go de RAM

## Temps d'entrainement observes

Temps observes sur la machine de reference a partir des runs traces dans MLflow :

- `SimpleCNN` : 19 min 07 s
- `ResNet18 transfer` : 19 min 07 s
- `TinyViT` : 6 min 16 s
- `Conv Autoencoder` : 4 min 09 s
- `Image only` : 5 min 37 s
- `Text only` : 4 min 15 s
- `Fusion image + texte` : 6 min 36 s

Au total, le pipeline final d'entrainement represente environ `1 h 05 min`, hors telechargement des donnees, generation EDA et lancement des interfaces.
