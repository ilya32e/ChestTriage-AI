from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sys

from mlflow.tracking import MlflowClient


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
ARTIFACTS_DIR = ROOT / "artifacts"


@dataclass
class RunSnapshot:
    label: str
    status: str
    device: str
    epochs: str
    duration: str
    best_val: str
    test_roc_auc: str
    test_ap: str
    test_f1: str
    experiment_name: str
    run_name: str
    run_id: str


@dataclass
class DeploymentEntry:
    component: str
    checkpoint_label: str
    checkpoint_path: str
    experiment_name: str
    run_name: str
    run_id: str
    duration: str


def format_metric(value: float | None) -> str:
    if value is None:
        return "n.d."
    return f"{value:.4f}"


def format_duration(duration_ms: int | None) -> str:
    if duration_ms is None or duration_ms <= 0:
        return "n.d."
    total_seconds = int(round(duration_ms / 1000))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def get_experiment_id(client: MlflowClient, experiment_name: str) -> str | None:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    return experiment.experiment_id


def pick_run(
    client: MlflowClient,
    experiment_name: str,
    run_name: str,
    best_metric: str | None,
) -> dict | None:
    experiment_id = get_experiment_id(client, experiment_name)
    if experiment_id is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    finished_runs = [run for run in runs if run.info.status == "FINISHED"]
    if not finished_runs:
        return None

    if best_metric is not None:
        with_metric = [run for run in finished_runs if best_metric in run.data.metrics]
        if with_metric:
            chosen = max(with_metric, key=lambda run: run.data.metrics[best_metric])
            return {"run": chosen, "metric_key": best_metric}

    chosen = max(
        finished_runs,
        key=lambda run: run.info.end_time or run.info.start_time or 0,
    )
    return {"run": chosen, "metric_key": best_metric}


def build_snapshot(
    client: MlflowClient,
    *,
    label: str,
    experiment_name: str,
    run_name: str,
    best_metric: str,
    test_roc_auc_metric: str,
    test_ap_metric: str,
    test_f1_metric: str,
) -> RunSnapshot:
    selected = pick_run(client, experiment_name, run_name, best_metric)
    if selected is None:
        return RunSnapshot(
            label=label,
            status="non disponible",
            device="-",
            epochs="-",
            duration="n.d.",
            best_val="n.d.",
            test_roc_auc="n.d.",
            test_ap="n.d.",
            test_f1="n.d.",
            experiment_name=experiment_name,
            run_name=run_name,
            run_id="-",
        )

    run = selected["run"]
    params = run.data.params
    metrics = run.data.metrics
    duration_ms = None
    if run.info.start_time is not None and run.info.end_time is not None:
        duration_ms = run.info.end_time - run.info.start_time
    return RunSnapshot(
        label=label,
        status="termine",
        device=params.get("device", "-"),
        epochs=params.get("training.epochs", "-"),
        duration=format_duration(duration_ms),
        best_val=format_metric(metrics.get(best_metric)),
        test_roc_auc=format_metric(metrics.get(test_roc_auc_metric)),
        test_ap=format_metric(metrics.get(test_ap_metric)),
        test_f1=format_metric(metrics.get(test_f1_metric)),
        experiment_name=experiment_name,
        run_name=run_name,
        run_id=run.info.run_id,
    )


def build_robustness_row(client: MlflowClient) -> tuple[str, str, str]:
    selected = pick_run(
        client,
        experiment_name="radiology_multimodal",
        run_name="iu_xray_fusion_text",
        best_metric="test_macro_roc_auc",
    )
    if selected is None:
        return ("Fusion IU X-Ray", "n.d.", "n.d.")
    run = selected["run"]
    metrics = run.data.metrics
    return (
        "Fusion IU X-Ray",
        format_metric(metrics.get("test_text_missing_macro_roc_auc")),
        format_metric(metrics.get("test_image_missing_macro_roc_auc")),
    )


def supervised_rows(client: MlflowClient) -> list[RunSnapshot]:
    return [
        build_snapshot(
            client,
            label="SimpleCNN",
            experiment_name="chestmnist_supervised",
            run_name="simple_cnn_from_scratch",
            best_metric="best_val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
        build_snapshot(
            client,
            label="ResNet18 transfer",
            experiment_name="chestmnist_supervised",
            run_name="resnet18_transfer_learning",
            best_metric="best_val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
        build_snapshot(
            client,
            label="ResNet18 partial FT 224",
            experiment_name="chestmnist_supervised",
            run_name="resnet18_partial_finetune_224",
            best_metric="best_val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
        build_snapshot(
            client,
            label="TinyViT",
            experiment_name="chestmnist_supervised",
            run_name="tiny_vit",
            best_metric="best_val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
    ]


def anomaly_rows(client: MlflowClient) -> list[RunSnapshot]:
    return [
        build_snapshot(
            client,
            label="Conv Autoencoder",
            experiment_name="chestmnist_anomaly",
            run_name="conv_autoencoder",
            best_metric="best_val_roc_auc",
            test_roc_auc_metric="test_roc_auc",
            test_ap_metric="test_average_precision",
            test_f1_metric="test_f1",
        )
    ]


def multimodal_rows(client: MlflowClient) -> list[RunSnapshot]:
    return [
        build_snapshot(
            client,
            label="Image only",
            experiment_name="radiology_multimodal",
            run_name="iu_xray_image_only",
            best_metric="val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
        build_snapshot(
            client,
            label="Text only",
            experiment_name="radiology_multimodal",
            run_name="iu_xray_text_only",
            best_metric="val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
        build_snapshot(
            client,
            label="Fusion image + texte",
            experiment_name="radiology_multimodal",
            run_name="iu_xray_fusion_text",
            best_metric="val_macro_roc_auc",
            test_roc_auc_metric="test_macro_roc_auc",
            test_ap_metric="test_macro_average_precision",
            test_f1_metric="test_macro_f1",
        ),
    ]


def render_table(title: str, rows: list[RunSnapshot]) -> list[str]:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{title}}}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{3.0cm}p{1.9cm}c c c c c c}",
        r"\toprule",
        r"Modele & Statut & Device & Epoques & Duree & Val. & Test ROC-AUC & Test AP / F1 \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.label} & {row.status} & {row.device} & {row.epochs} & {row.duration} & {row.best_val} & {row.test_roc_auc} & {row.test_ap} / {row.test_f1} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return lines


def render_robustness_table(row: tuple[str, str, str]) -> list[str]:
    label, text_missing, image_missing = row
    return [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Robustesse du modele de fusion quand une modalite manque.}",
        r"\small",
        r"\begin{tabular}{p{4.5cm}c c}",
        r"\toprule",
        r"Modele & ROC-AUC sans texte & ROC-AUC sans image \\",
        r"\midrule",
        f"{label} & {text_missing} & {image_missing} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]


def build_deployment_entries(client: MlflowClient) -> list[DeploymentEntry]:
    selected_supervised = build_snapshot(
        client,
        label="ResNet18 partial FT 224",
        experiment_name="chestmnist_supervised",
        run_name="resnet18_partial_finetune_224",
        best_metric="best_val_macro_roc_auc",
        test_roc_auc_metric="test_macro_roc_auc",
        test_ap_metric="test_macro_average_precision",
        test_f1_metric="test_macro_f1",
    )
    selected_anomaly = build_snapshot(
        client,
        label="Conv Autoencoder",
        experiment_name="chestmnist_anomaly",
        run_name="conv_autoencoder",
        best_metric="best_val_roc_auc",
        test_roc_auc_metric="test_roc_auc",
        test_ap_metric="test_average_precision",
        test_f1_metric="test_f1",
    )
    selected_multimodal = build_snapshot(
        client,
        label="Fusion image + texte",
        experiment_name="radiology_multimodal",
        run_name="iu_xray_fusion_text",
        best_metric="val_macro_roc_auc",
        test_roc_auc_metric="test_macro_roc_auc",
        test_ap_metric="test_macro_average_precision",
        test_f1_metric="test_macro_f1",
    )
    return [
        DeploymentEntry(
            component="Supervision",
            checkpoint_label="supervised/resnet18_partial_finetune_224",
            checkpoint_path="artifacts/supervised/resnet18_partial_finetune_224/best_model.pt",
            experiment_name=selected_supervised.experiment_name,
            run_name=selected_supervised.run_name,
            run_id=selected_supervised.run_id,
            duration=selected_supervised.duration,
        ),
        DeploymentEntry(
            component="Anomalie",
            checkpoint_label="anomaly/conv_autoencoder",
            checkpoint_path="artifacts/anomaly/conv_autoencoder/best_autoencoder.pt",
            experiment_name=selected_anomaly.experiment_name,
            run_name=selected_anomaly.run_name,
            run_id=selected_anomaly.run_id,
            duration=selected_anomaly.duration,
        ),
        DeploymentEntry(
            component="Multimodal",
            checkpoint_label="multimodal/iu_xray_fusion",
            checkpoint_path="artifacts/multimodal/iu_xray_fusion/best_multimodal_model.pt",
            experiment_name=selected_multimodal.experiment_name,
            run_name=selected_multimodal.run_name,
            run_id=selected_multimodal.run_id,
            duration=selected_multimodal.duration,
        ),
    ]


def render_traceability_table(entries: list[DeploymentEntry]) -> list[str]:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Tra\c{c}abilite des checkpoints deployes dans le demonstrateur.}",
        r"\small",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{p{2.0cm}p{3.3cm}p{2.9cm}p{3.1cm}p{4.3cm}}",
        r"\toprule",
        r"Composante & Checkpoint deploye & Experience MLflow & Run name & Run ID \\",
        r"\midrule",
    ]
    for entry in entries:
        lines.append(
            f"{latex_escape(entry.component)} & \\texttt{{{latex_escape(entry.checkpoint_label)}}} & "
            f"\\texttt{{{latex_escape(entry.experiment_name)}}} & \\texttt{{{latex_escape(entry.run_name)}}} & "
            f"\\texttt{{{entry.run_id}}} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table}",
        ]
    )
    return lines


def write_deployment_manifest(entries: list[DeploymentEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "deployed_models": [
            {
                "component": entry.component,
                "checkpoint_label": entry.checkpoint_label,
                "checkpoint_path": entry.checkpoint_path,
                "experiment_name": entry.experiment_name,
                "run_name": entry.run_name,
                "run_id": entry.run_id,
                "duration": entry.duration,
            }
            for entry in entries
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=f"file:{(ROOT / 'mlruns').resolve()}")
    snapshot_time = datetime.now().strftime("%d/%m/%Y %H:%M")
    supervised = supervised_rows(client)
    anomaly = anomaly_rows(client)
    multimodal = multimodal_rows(client)
    robustness = build_robustness_row(client)
    deployment_entries = build_deployment_entries(client)
    all_rows = [*supervised, *anomaly, *multimodal]
    missing_results = any(
        row.best_val == "n.d." or row.test_roc_auc == "n.d." or row.test_ap == "n.d." or row.test_f1 == "n.d."
        for row in all_rows
    )
    snapshot_note = (
        rf"Les tableaux ci-dessous ont ete generes automatiquement depuis le suivi MLflow le {snapshot_time}. Tous les runs finaux attendus sont disponibles dans ce snapshot. Les quickstarts ne sont pas reportes ici afin de ne conserver que les experiences finales du projet."
        if not missing_results
        else rf"Les tableaux ci-dessous ont ete generes automatiquement depuis le suivi MLflow le {snapshot_time}. Les cellules \textit{{n.d.}} correspondent a des runs finaux absents ou non termines. Les quickstarts ne sont pas reportes ici afin de ne conserver que les experiences finales du projet."
    )

    lines: list[str] = [
        "% Auto-generated by scripts/export_report_tables.py",
        rf"% Snapshot: {snapshot_time}",
        r"\paragraph{Instantane MLflow.}",
        snapshot_note,
        "",
    ]
    lines.extend(render_table("Synthese des runs supervises finaux.", supervised))
    lines.append("")
    lines.extend(render_table("Synthese du run final de detection d'anomalies.", anomaly))
    lines.append("")
    lines.extend(render_table("Synthese des runs multimodaux finaux.", multimodal))
    lines.append("")
    lines.extend(render_robustness_table(robustness))
    lines.append("")
    lines.extend(render_traceability_table(deployment_entries))
    lines.append("")
    lines.append(
        r"Le manifest de deploiement associe au demonstrateur est exporte dans \texttt{../artifacts/deployment\_manifest.json}."
    )
    lines.append("")

    output_path = REPORT_DIR / "generated_results.tex"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    write_deployment_manifest(deployment_entries, ARTIFACTS_DIR / "deployment_manifest.json")
    print(f"Written {output_path}")


if __name__ == "__main__":
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    main()
