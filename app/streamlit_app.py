from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from radiology_triage.data.chestmnist import build_transforms  # noqa: E402
from radiology_triage.data.multimodal import Vocabulary, preprocess_report_text  # noqa: E402
try:  # noqa: E402
    from radiology_triage.models.autoencoder import build_reconstruction_model  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - defensive fallback for stale environments
    from radiology_triage.models.autoencoder import ConvAutoencoder  # noqa: E402

    def build_reconstruction_model(
        model_name: str,
        in_channels: int = 1,
        latent_channels: int = 128,
    ) -> torch.nn.Module:
        normalized_name = model_name.lower()
        if normalized_name in {"conv_autoencoder", "autoencoder", "ae"}:
            return ConvAutoencoder(in_channels=in_channels, latent_channels=latent_channels)
        if normalized_name in {"conv_vae", "vae"}:
            raise NotImplementedError(
                "A convolutional VAE can be added on top of this builder without changing the Streamlit API."
            )
        raise ValueError(f"Unsupported reconstruction model: {model_name}")

from radiology_triage.models.multimodal import build_multimodal_model  # noqa: E402
from radiology_triage.models.supervised import build_supervised_model  # noqa: E402
from radiology_triage.utils.explainability import build_gradcam_package  # noqa: E402
from radiology_triage.utils.io import ensure_dir, load_checkpoint  # noqa: E402
from radiology_triage.utils.metrics import apply_multilabel_thresholds, top_k_predictions  # noqa: E402


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_default_checkpoint(candidates: list[str]) -> str:
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    return candidates[0]


def to_relative_display(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def shorten_identifier(identifier: str | None, keep: int = 8) -> str:
    if not identifier:
        return "indisponible"
    if len(identifier) <= keep:
        return identifier
    return identifier[:keep]


def format_supervised_model_name(model_name: str) -> str:
    mapping = {
        "simple_cnn": "SimpleCNN",
        "resnet18": "ResNet18",
        "tiny_vit": "TinyViT",
        "vit_b_16": "ViT-B/16",
        "densenet121": "DenseNet121",
        "efficientnet_b0": "EfficientNet-B0",
    }
    return mapping.get(model_name.lower(), model_name)


def format_anomaly_model_name(model_name: str) -> str:
    mapping = {
        "conv_autoencoder": "ConvAE",
        "autoencoder": "ConvAE",
        "ae": "ConvAE",
        "conv_vae": "ConvVAE",
        "vae": "ConvVAE",
    }
    return mapping.get(model_name.lower(), model_name)


def infer_multimodal_dataset_name(config: dict[str, Any]) -> str:
    csv_path = str(config.get("dataset", {}).get("csv_path", "")).lower()
    if "iu_xray" in csv_path or "openi" in csv_path:
        return "IU X-Ray"
    if "nih" in csv_path:
        return "NIH"
    return "Multimodal"


@st.cache_data(show_spinner=False)
def load_deployment_manifest(path: str) -> dict | None:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def lookup_traceability(checkpoint_path: str, checkpoint_metadata: dict | None, manifest: dict | None) -> dict | None:
    if checkpoint_metadata and checkpoint_metadata.get("run_id"):
        return checkpoint_metadata
    if manifest is None:
        return None
    resolved_checkpoint = str(Path(checkpoint_path).resolve())
    for entry in manifest.get("deployed_models", []):
        manifest_path = str((ROOT / entry["checkpoint_path"]).resolve())
        if manifest_path == resolved_checkpoint:
            return {
                "run_id": entry["run_id"],
                "experiment_name": entry["experiment_name"],
                "run_name": entry["run_name"],
                "duration": entry.get("duration"),
            }
    return None


@st.cache_resource(show_spinner=False)
def load_supervised_bundle(checkpoint_path: str) -> dict | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    checkpoint = load_checkpoint(path, map_location=DEVICE)
    config = checkpoint["config"]
    model = build_supervised_model(
        model_name=checkpoint["model_name"],
        num_labels=len(checkpoint["label_names"]),
        pretrained=False,
        dropout=checkpoint.get("dropout", config["model"].get("dropout", 0.2)),
        in_channels=3 if config["dataset"].get("as_rgb", True) else 1,
        image_size=checkpoint["image_size"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    transform = build_transforms(
        image_size=checkpoint["image_size"],
        augment=False,
        normalization=config["dataset"].get("normalization", "imagenet"),
    )
    return {
        "model": model,
        "model_name": checkpoint["model_name"],
        "display_name": format_supervised_model_name(checkpoint["model_name"]),
        "transform": transform,
        "label_names": checkpoint["label_names"],
        "config": config,
        "checkpoint_path": str(path.resolve()),
        "dataset_name": "ChestMNIST",
        "traceability": {
            "run_id": checkpoint.get("mlflow_run_id"),
            "experiment_name": checkpoint.get("mlflow_experiment_name"),
            "run_name": checkpoint.get("mlflow_run_name"),
        },
    }


@st.cache_resource(show_spinner=False)
def load_anomaly_bundle(checkpoint_path: str) -> dict | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    checkpoint = load_checkpoint(path, map_location=DEVICE)
    model = build_reconstruction_model(
        model_name=checkpoint["config"].get("model", {}).get("name", "conv_autoencoder"),
        in_channels=checkpoint.get("in_channels", 1),
        latent_channels=checkpoint["config"]["model"].get("latent_channels", 128),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    transform = build_transforms(
        image_size=checkpoint["image_size"],
        augment=False,
        normalization=checkpoint["config"]["dataset"].get("normalization", "none"),
    )
    return {
        "model": model,
        "model_name": checkpoint["config"].get("model", {}).get("name", "conv_autoencoder"),
        "display_name": format_anomaly_model_name(checkpoint["config"].get("model", {}).get("name", "conv_autoencoder")),
        "transform": transform,
        "threshold": checkpoint["threshold"],
        "config": checkpoint["config"],
        "checkpoint_path": str(path.resolve()),
        "dataset_name": "ChestMNIST",
        "traceability": {
            "run_id": checkpoint.get("mlflow_run_id"),
            "experiment_name": checkpoint.get("mlflow_experiment_name"),
            "run_name": checkpoint.get("mlflow_run_name"),
        },
    }


@st.cache_resource(show_spinner=False)
def load_multimodal_bundle(checkpoint_path: str) -> dict | None:
    path = Path(checkpoint_path)
    if not path.exists():
        return None
    checkpoint = load_checkpoint(path, map_location=DEVICE)
    config = checkpoint["config"]
    vocab = Vocabulary.from_state_dict(checkpoint["vocab_state"])
    model = build_multimodal_model(
        config,
        vocab_size=checkpoint["vocab_size"],
        num_labels=len(checkpoint["label_names"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE).eval()
    image_size = checkpoint["image_size"]
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {
        "model": model,
        "transform": transform,
        "label_names": checkpoint["label_names"],
        "vocab": vocab,
        "max_length": checkpoint["max_length"],
        "config": config,
        "mode": config["model"]["mode"],
        "checkpoint_path": str(path.resolve()),
        "dataset_name": infer_multimodal_dataset_name(config),
        "traceability": {
            "run_id": checkpoint.get("mlflow_run_id"),
            "experiment_name": checkpoint.get("mlflow_experiment_name"),
            "run_name": checkpoint.get("mlflow_run_name"),
        },
    }


def load_threshold_payload(checkpoint_path: str, label_names: list[str]) -> dict[str, Any]:
    threshold_path = Path(checkpoint_path).with_name("class_thresholds.json")
    default_thresholds = [0.5] * len(label_names)
    payload: dict[str, Any] = {
        "available": False,
        "ordered_thresholds": default_thresholds,
        "path": str(threshold_path.resolve()),
        "warning": "Seuils calibrés absents. Fallback explicite sur 0.50 par classe.",
    }
    if not threshold_path.exists():
        return payload

    try:
        raw_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        if "ordered_thresholds" in raw_payload:
            ordered_thresholds = [float(value) for value in raw_payload["ordered_thresholds"]]
        else:
            threshold_map = raw_payload.get("thresholds", {})
            ordered_thresholds = [float(threshold_map.get(label_name, 0.5)) for label_name in label_names]
        if len(ordered_thresholds) != len(label_names):
            raise ValueError("Threshold count does not match the label count.")

        payload.update(
            {
                "available": True,
                "ordered_thresholds": ordered_thresholds,
                "payload": raw_payload,
                "warning": None,
            }
        )
        return payload
    except (json.JSONDecodeError, OSError, ValueError) as error:
        payload["warning"] = (
            "Seuils calibrés illisibles. Fallback explicite sur 0.50 par classe. "
            f"Détails : {error}"
        )
        return payload


def predict_supervised(bundle: dict, image: Image.Image, threshold_payload: dict[str, Any]) -> dict[str, Any]:
    converted = image.convert("RGB" if bundle["config"]["dataset"].get("as_rgb", True) else "L")
    input_tensor = bundle["transform"](converted).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.sigmoid(bundle["model"](input_tensor))[0].cpu()

    decisions = apply_multilabel_thresholds(
        probabilities.unsqueeze(0),
        thresholds=threshold_payload["ordered_thresholds"],
    )[0]
    rows = []
    for index, label_name in enumerate(bundle["label_names"]):
        rows.append(
            {
                "Classe": label_name,
                "Probabilité": float(probabilities[index]),
                "Seuil": float(threshold_payload["ordered_thresholds"][index]),
                "Décision": "retenue" if int(decisions[index]) == 1 else "sous le seuil",
            }
        )

    predictions_df = pd.DataFrame(rows).sort_values("Probabilité", ascending=False).reset_index(drop=True)
    positives_df = predictions_df[predictions_df["Décision"] == "retenue"].reset_index(drop=True)
    positives_df["Statut"] = "retenue"
    positives_df = positives_df[["Classe", "Probabilité", "Seuil", "Statut"]]
    default_gradcam_label = positives_df.iloc[0]["Classe"] if not positives_df.empty else predictions_df.iloc[0]["Classe"]
    return {
        "input_tensor": input_tensor.detach(),
        "probabilities": probabilities,
        "all_predictions": predictions_df,
        "positive_predictions": positives_df,
        "top_predictions": predictions_df.head(min(8, len(predictions_df))).copy(),
        "default_gradcam_label": str(default_gradcam_label),
        "top_export_predictions": top_k_predictions(probabilities, bundle["label_names"], k=min(5, len(bundle["label_names"]))),
    }


def score_anomaly(bundle: dict, image: Image.Image) -> dict[str, Any]:
    converted = image.convert("L" if bundle["config"]["dataset"].get("as_rgb", False) is False else "RGB")
    tensor = bundle["transform"](converted).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        reconstruction = bundle["model"](tensor)
        score = torch.mean(torch.abs(reconstruction - tensor)).item()

    decision = "atypique à vérifier" if score >= bundle["threshold"] else "dans la distribution normale"
    if score >= bundle["threshold"]:
        interpretation = "Score de reconstruction élevé. Cas atypique à vérifier dans le flux de tri."
    else:
        interpretation = "Score compatible avec les cas vus à l'entraînement."

    return {
        "score": float(score),
        "threshold": float(bundle["threshold"]),
        "decision": decision,
        "interpretation": interpretation,
    }


def predict_multimodal(bundle: dict, image: Image.Image, report_text: str) -> dict[str, Any]:
    tensor = bundle["transform"](image.convert("RGB")).unsqueeze(0).to(DEVICE)
    model_text = preprocess_report_text(
        report_text,
        redact_label_mentions=bundle["config"]["dataset"].get("redact_label_mentions", False),
    )
    input_ids, attention_mask = bundle["vocab"].encode(model_text, bundle["max_length"])
    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    attention_mask = attention_mask.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mode = bundle["mode"].lower()
        if mode in {"image", "image_only"}:
            logits = bundle["model"](tensor)
        elif mode in {"text", "text_only"}:
            logits = bundle["model"](input_ids, attention_mask)
        else:
            logits = bundle["model"](tensor, input_ids, attention_mask)
        probabilities = torch.sigmoid(logits)[0].cpu()
    rows = top_k_predictions(probabilities, bundle["label_names"], k=min(5, len(bundle["label_names"])))
    return {
        "top_predictions": pd.DataFrame(rows, columns=["Classe", "Probabilité"]),
        "raw_predictions": rows,
    }


def export_inference_result(result_payload: dict[str, Any], export_format: str) -> Path:
    export_dir = ensure_dir(ROOT / "artifacts" / "exports")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = "json" if export_format == "json" else "txt"
    export_path = export_dir / f"inference_{timestamp}.{extension}"

    if export_format == "json":
        export_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return export_path

    lines = [
        f"timestamp: {result_payload['timestamp']}",
        f"supervised_model: {result_payload['supervised']['model_name']}",
        f"supervised_checkpoint: {result_payload['supervised']['checkpoint']}",
        "top_supervised_predictions:",
    ]
    for prediction in result_payload["supervised"]["top_predictions"]:
        lines.append(f"  - {prediction['label']}: {prediction['probability']:.4f}")
    lines.extend(
        [
            f"anomaly_score: {result_payload['anomaly']['score']:.4f}",
            f"anomaly_threshold: {result_payload['anomaly']['threshold']:.4f}",
            f"anomaly_decision: {result_payload['anomaly']['decision']}",
            f"text_input: {result_payload.get('text_input', '')}",
        ]
    )
    export_path.write_text("\n".join(lines), encoding="utf-8")
    return export_path


def build_export_payload(
    supervised_bundle: dict | None,
    supervised_predictions: dict | None,
    anomaly_bundle: dict | None,
    anomaly_result: dict | None,
    threshold_payload: dict[str, Any] | None,
    text_input: str,
) -> dict[str, Any]:
    top_predictions = []
    if supervised_predictions is not None:
        for label, probability in supervised_predictions["top_export_predictions"]:
            top_predictions.append({"label": label, "probability": float(probability)})

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "supervised": {
            "model_name": supervised_bundle["display_name"] if supervised_bundle else "unavailable",
            "checkpoint": supervised_bundle["checkpoint_path"] if supervised_bundle else "",
            "dataset": supervised_bundle["dataset_name"] if supervised_bundle else "",
            "thresholds_available": bool(threshold_payload and threshold_payload["available"]),
            "thresholds_path": threshold_payload["path"] if threshold_payload else "",
            "top_predictions": top_predictions,
        },
        "anomaly": {
            "model_name": anomaly_bundle["display_name"] if anomaly_bundle else "unavailable",
            "checkpoint": anomaly_bundle["checkpoint_path"] if anomaly_bundle else "",
            "score": anomaly_result["score"] if anomaly_result else float("nan"),
            "threshold": anomaly_result["threshold"] if anomaly_result else float("nan"),
            "decision": anomaly_result["decision"] if anomaly_result else "unavailable",
        },
        "text_input": text_input.strip(),
    }


def render_traceability(label: str, traceability: dict | None) -> None:
    if traceability is None:
        st.caption(f"{label}: traceabilité indisponible")
        return
    duration = f" / {traceability['duration']}" if traceability.get("duration") else ""
    st.caption(
        f"{label}: {traceability['experiment_name']} / {traceability['run_name']} / "
        f"{shorten_identifier(traceability['run_id'])}{duration}"
    )


def supports_gradcam(model_name: str) -> bool:
    return model_name.lower() in {"simple_cnn", "resnet18"}


st.set_page_config(page_title="Radiology Triage Demo", layout="wide")
st.title("Système d'aide au tri radiologique")
st.caption(
    "Sortie principale : classes retenues après calibration des seuils. "
    "Les modules anomalie et multimodal sont fournis comme compléments techniques."
)
deployment_manifest = load_deployment_manifest(str(ROOT / "artifacts" / "deployment_manifest.json"))

with st.sidebar:
    st.subheader("Options")
    show_gradcam = st.checkbox("Activer Grad-CAM", value=False)
with st.expander("Modifier les checkpoints", expanded=False):
        supervised_path = st.text_input(
            "Checkpoint supervisé",
            value=resolve_default_checkpoint(
                [
                    "artifacts/supervised/simple_cnn_224/best_model.pt",
                    "artifacts/supervised/simple_cnn/best_model.pt",
                    "artifacts/supervised/resnet18/best_model.pt",
                    "artifacts/supervised/tiny_vit/best_model.pt",
                ]
            ),
        )
        anomaly_path = st.text_input(
            "Checkpoint anomalie",
            value=resolve_default_checkpoint(["artifacts/anomaly/conv_autoencoder/best_autoencoder.pt"]),
        )
        multimodal_image_path = st.text_input(
            "Checkpoint image seule",
            value=resolve_default_checkpoint(
                [
                    "artifacts/multimodal/iu_xray_image_only_224/best_multimodal_model.pt",
                    "artifacts/multimodal/iu_xray_image_only/best_multimodal_model.pt",
                ]
            ),
        )
        multimodal_text_path = st.text_input(
            "Checkpoint texte seul",
            value=resolve_default_checkpoint(
                [
                    "artifacts/multimodal/iu_xray_text_only_224/best_multimodal_model.pt",
                    "artifacts/multimodal/iu_xray_text_only/best_multimodal_model.pt",
                ]
            ),
        )
        multimodal_fusion_path = st.text_input(
            "Checkpoint fusion",
            value=resolve_default_checkpoint(
                [
                    "artifacts/multimodal/iu_xray_fusion_224/best_multimodal_model.pt",
                    "artifacts/multimodal/iu_xray_fusion/best_multimodal_model.pt",
                    "artifacts/multimodal/fusion/best_multimodal_model.pt",
                ]
            ),
        )

supervised_bundle = load_supervised_bundle(supervised_path) if supervised_path else None
anomaly_bundle = load_anomaly_bundle(anomaly_path) if anomaly_path else None
multimodal_image_bundle = load_multimodal_bundle(multimodal_image_path) if multimodal_image_path else None
multimodal_text_bundle = load_multimodal_bundle(multimodal_text_path) if multimodal_text_path else None
multimodal_fusion_bundle = load_multimodal_bundle(multimodal_fusion_path) if multimodal_fusion_path else None
threshold_payload = (
    load_threshold_payload(supervised_bundle["checkpoint_path"], supervised_bundle["label_names"])
    if supervised_bundle
    else None
)
supervised_traceability = (
    lookup_traceability(supervised_bundle["checkpoint_path"], supervised_bundle["traceability"], deployment_manifest)
    if supervised_bundle
    else None
)
anomaly_traceability = (
    lookup_traceability(anomaly_bundle["checkpoint_path"], anomaly_bundle["traceability"], deployment_manifest)
    if anomaly_bundle
    else None
)
multimodal_traceability = (
    lookup_traceability(
        multimodal_fusion_bundle["checkpoint_path"],
        multimodal_fusion_bundle["traceability"],
        deployment_manifest,
    )
    if multimodal_fusion_bundle
    else None
)

with st.sidebar:
    st.subheader("Modèles chargés")
    st.markdown(f"**Supervisé** : {supervised_bundle['display_name'] if supervised_bundle else 'indisponible'}")
    st.markdown(f"**Anomalie** : {anomaly_bundle['display_name'] if anomaly_bundle else 'indisponible'}")
    st.markdown(
        f"**Image seule** : {multimodal_image_bundle['dataset_name'] if multimodal_image_bundle else 'indisponible'}"
    )
    st.markdown(
        f"**Texte seul** : {multimodal_text_bundle['dataset_name'] if multimodal_text_bundle else 'indisponible'}"
    )
    st.markdown(
        f"**Fusion** : {multimodal_fusion_bundle['dataset_name'] if multimodal_fusion_bundle else 'indisponible'}"
    )

    with st.expander("Chemins chargés", expanded=False):
        if supervised_bundle:
            st.code(to_relative_display(supervised_bundle["checkpoint_path"]))
        if anomaly_bundle:
            st.code(to_relative_display(anomaly_bundle["checkpoint_path"]))
        if multimodal_image_bundle:
            st.code(to_relative_display(multimodal_image_bundle["checkpoint_path"]))
        if multimodal_text_bundle:
            st.code(to_relative_display(multimodal_text_bundle["checkpoint_path"]))
        if multimodal_fusion_bundle:
            st.code(to_relative_display(multimodal_fusion_bundle["checkpoint_path"]))

    st.subheader("Traçabilité")
    with st.expander("Runs MLflow", expanded=False):
        render_traceability("Supervisé", supervised_traceability)
        render_traceability("Anomalie", anomaly_traceability)
        render_traceability("Fusion", multimodal_traceability)

uploaded_file = st.file_uploader("Charger une radiographie thoracique", type=["png", "jpg", "jpeg"])
report_text = st.text_area("Compte-rendu ou texte associé pour la preuve de concept multimodale", height=160)

if uploaded_file is None:
    st.info("Ajoute une image pour lancer l'inférence.")
else:
    image = Image.open(uploaded_file)
    col_image, col_results = st.columns([1, 1.6])

    with col_image:
        st.image(image, caption="Radiographie chargée", width="stretch")

    supervised_predictions: dict[str, Any] | None = None
    anomaly_result: dict[str, Any] | None = None

    with col_results:
        st.subheader("Classification supervisée")
        st.info("Sortie principale : classes retenues après calibration des seuils.")
        if supervised_bundle is None:
            st.warning("Checkpoint supervisé introuvable.")
        else:
            info_col_1, info_col_2, info_col_3 = st.columns(3)
            info_col_1.markdown(f"**Modèle**  \n{supervised_bundle['display_name']}")
            info_col_2.markdown(f"**Dataset**  \n{supervised_bundle['dataset_name']}")
            info_col_3.markdown(
                "**Seuils calibrés**  \n"
                + ("Oui" if threshold_payload and threshold_payload["available"] else "Non")
            )

            if threshold_payload and threshold_payload["warning"] is not None:
                st.warning(threshold_payload["warning"])

            supervised_predictions = predict_supervised(supervised_bundle, image, threshold_payload or {})
            if supervised_predictions["positive_predictions"].empty:
                st.info("Aucune classe ne dépasse son seuil de calibration pour cette image.")
            else:
                display_df = supervised_predictions["positive_predictions"].copy()
                display_df["Probabilité"] = display_df["Probabilité"].map(lambda value: f"{value:.4f}")
                display_df["Seuil"] = display_df["Seuil"].map(lambda value: f"{value:.4f}")
                st.dataframe(display_df, width="stretch", hide_index=True)

            with st.expander("Vue probabiliste complète", expanded=False):
                full_df = supervised_predictions["all_predictions"].copy()
                full_df["Probabilité"] = full_df["Probabilité"].map(lambda value: f"{value:.4f}")
                full_df["Seuil"] = full_df["Seuil"].map(lambda value: f"{value:.4f}")
                st.dataframe(full_df, width="stretch", hide_index=True)

            with st.expander("Détails techniques", expanded=False):
                st.write(f"Checkpoint chargé : `{to_relative_display(supervised_bundle['checkpoint_path'])}`")
                if threshold_payload:
                    st.write(f"Fichier de seuils : `{to_relative_display(threshold_payload['path'])}`")
                if supervised_traceability is not None:
                    st.write(f"Run name : `{supervised_traceability.get('run_name', 'indisponible')}`")
                    st.write(f"Run ID court : `{shorten_identifier(supervised_traceability.get('run_id'))}`")
                st.write("Prétraitement : redimensionnement et normalisation cohérents avec le checkpoint chargé.")

            if show_gradcam:
                st.markdown("**Grad-CAM**")
                if not supports_gradcam(supervised_bundle["model_name"]):
                    st.info(
                        "Grad-CAM n'est activé que pour SimpleCNN et ResNet18. "
                        "Les architectures Transformer compactes ne disposent pas ici d'une carte spatiale fiable et légère."
                    )
                else:
                    selectable_labels = supervised_predictions["all_predictions"]["Classe"].tolist()
                    default_label = supervised_predictions["default_gradcam_label"]
                    default_index = selectable_labels.index(default_label) if default_label in selectable_labels else 0
                    selected_label = st.selectbox(
                        "Classe cible Grad-CAM",
                        selectable_labels,
                        index=default_index,
                        help="Par défaut, la classe principale retenue est présélectionnée.",
                    )
                    selected_index = supervised_bundle["label_names"].index(selected_label)
                    gradcam_package = build_gradcam_package(
                        supervised_bundle["model"],
                        supervised_bundle["model_name"],
                        supervised_predictions["input_tensor"].clone().detach().to(DEVICE).requires_grad_(True),
                        image,
                        selected_index,
                        selected_label,
                    )
                    if gradcam_package is None:
                        st.info("Grad-CAM indisponible pour ce modèle.")
                    else:
                        st.caption(f"Carte calculée sur la classe cible : {gradcam_package['target_label']}")
                        st.image(gradcam_package["overlay"], width="stretch")

        st.divider()
        st.subheader("Détection d'anomalie")
        if anomaly_bundle is None:
            st.warning("Checkpoint anomalie introuvable.")
        else:
            anomaly_result = score_anomaly(anomaly_bundle, image)
            metric_col_1, metric_col_2 = st.columns(2)
            metric_col_1.metric("Score", f"{anomaly_result['score']:.4f}")
            metric_col_2.metric("Seuil", f"{anomaly_result['threshold']:.4f}")
            if anomaly_result["decision"] == "atypique à vérifier":
                st.warning("Décision : atypique à vérifier")
            else:
                st.success("Décision : dans la distribution normale")
            st.caption(anomaly_result["interpretation"])
            st.caption("Ce module sert d'indicateur technique de distribution et ne constitue jamais un diagnostic clinique.")

    st.divider()
    st.subheader("Preuve de concept multimodale")
    st.info(
        "Preuve de concept multimodale. Cette sortie est fournie à titre comparatif "
        "et peut être sensible au texte saisi."
    )

    multimodal_columns = st.columns(3)

    with multimodal_columns[0]:
        st.markdown("**Image seule**")
        if multimodal_image_bundle is None:
            st.warning("Checkpoint image seule introuvable.")
        else:
            image_only_predictions = predict_multimodal(multimodal_image_bundle, image, "")
            image_only_df = image_only_predictions["top_predictions"].copy()
            image_only_df["Probabilité"] = image_only_df["Probabilité"].map(lambda value: f"{value:.4f}")
            st.dataframe(image_only_df, width="stretch", hide_index=True)

    with multimodal_columns[1]:
        st.markdown("**Texte seul**")
        if multimodal_text_bundle is None:
            st.warning("Checkpoint texte seul introuvable.")
        elif not report_text.strip():
            st.info("Ajoute un texte pour activer la branche texte seule.")
        else:
            text_only_predictions = predict_multimodal(multimodal_text_bundle, image, report_text)
            text_only_df = text_only_predictions["top_predictions"].copy()
            text_only_df["Probabilité"] = text_only_df["Probabilité"].map(lambda value: f"{value:.4f}")
            st.dataframe(text_only_df, width="stretch", hide_index=True)

    with multimodal_columns[2]:
        st.markdown("**Fusion image + texte**")
        if multimodal_fusion_bundle is None:
            st.warning("Checkpoint fusion introuvable.")
        elif not report_text.strip():
            st.info("Ajoute un texte pour comparer image seule, texte seul et fusion.")
        else:
            fusion_predictions = predict_multimodal(multimodal_fusion_bundle, image, report_text)
            fusion_df = fusion_predictions["top_predictions"].copy()
            fusion_df["Probabilité"] = fusion_df["Probabilité"].map(lambda value: f"{value:.4f}")
            st.dataframe(fusion_df, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Exporter les résultats")
    export_payload = build_export_payload(
        supervised_bundle=supervised_bundle,
        supervised_predictions=supervised_predictions,
        anomaly_bundle=anomaly_bundle,
        anomaly_result=anomaly_result,
        threshold_payload=threshold_payload,
        text_input=report_text,
    )
    export_col_json, export_col_txt = st.columns(2)
    if export_col_json.button("Exporter en JSON"):
        export_path = export_inference_result(export_payload, "json")
        export_col_json.success(f"Export créé : {to_relative_display(export_path)}")
    if export_col_txt.button("Exporter en TXT"):
        export_path = export_inference_result(export_payload, "txt")
        export_col_txt.success(f"Export créé : {to_relative_display(export_path)}")
