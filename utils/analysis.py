import os
import pandas as pd
import yaml
import json


def retrieve_hyperparameters(hparams_path: str) -> dict:
    """Legge gli iperparametri da un file YAML."""
    if not os.path.exists(hparams_path):
        print(f"Attenzione: file hparams non trovato in {hparams_path}")
        return {}
    try:
        with open(hparams_path, 'r') as f:
            hparams_raw = yaml.safe_load(f)
            # This structure seems to match the project context
            generator = hparams_raw.get('opt', {}).get('model', {}).get('generator', {}).get('name', 'N/A')
            loss = hparams_raw.get('opt', {}).get('training', {}).get('losses', {}).get('gan_mode', 'N/A')
            return {'generator': generator, 'loss': loss}
    except Exception as e:
        print(f"Errore nella lettura di {hparams_path}: {e}")
        return {}


def read_metrics_from_json(metrics_path: str) -> dict:
    """Legge le metriche da un file JSON."""
    if not os.path.exists(metrics_path):
        return {}
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Errore di decodifica JSON in {metrics_path}")
        return {}
    except Exception as e:
        print(f"Errore nella lettura di {metrics_path}: {e}")
        return {}


def analyze_experiment_results(
    main_path: str,
    testing_datasets: list[str],
    metrics_to_analyze: list[str] = ['MAE', 'PSNR', 'SSIM', 'MM-SSIM'],
    outlier_metric: str = 'SSIM'
) -> tuple[dict, pd.DataFrame]:
    """
    Analizza i risultati, gestisce gli outlier e produce DataFrame dettagliati e riassuntivi.
    Consolida le metriche con nomi scritti diversamente (es. mm-ssim, MM-SSIM) in un'unica colonna.

    Args:
        main_path (str): Percorso principale contenente le directory delle versioni.
        testing_datasets (list[str]): Lista dei nomi dei dataset di test.
        metrics_to_analyze (list[str], optional): Lista delle metriche da analizzare.
        outlier_metric (str, optional): Metrica su cui basare la rimozione degli outlier.

    Returns:
        tuple[dict, pd.DataFrame]:
            - Dizionario con i risultati dettagliati (grezzi) per ogni esperimento.
            - DataFrame riassuntivo con medie e deviazioni standard (senza outlier).
    """
    all_results = []

    # 1. Standardizza i parametri di input in MAIUSCOLO per coerenza
    metrics_to_analyze_upper = [m.upper() for m in metrics_to_analyze]
    outlier_metric_upper = outlier_metric.upper()

    if outlier_metric_upper not in metrics_to_analyze_upper:
        raise ValueError(f"La metrica per l'outlier '{outlier_metric}' deve essere in 'metrics_to_analyze'.")

    print(f"Avvio analisi in: {main_path}")
    for version_folder in sorted(os.listdir(main_path)):
        version_path = os.path.join(main_path, version_folder)
        if not os.path.isdir(version_path) or not version_folder.startswith('version_'):
            continue
        
        hparams_path = os.path.join(version_path, "hparams.yaml")
        h_params = retrieve_hyperparameters(hparams_path)
        if not h_params: continue

        test_results_base_path = os.path.join(version_path, "test_results")
        for test_dataset in testing_datasets:
            dataset_path = os.path.join(test_results_base_path, test_dataset)
            if not os.path.isdir(dataset_path): continue

            for subject in os.listdir(dataset_path):
                subject_path = os.path.join(dataset_path, subject)
                if not os.path.isdir(subject_path): continue
                
                metrics_path = os.path.join(subject_path, "metrics.json")
                metrics = read_metrics_from_json(metrics_path)

                if metrics:
                    standardized_metrics = {key.upper(): value for key, value in metrics.items()}

                    record = {
                        'VERSION': version_folder,
                        'GENERATOR': h_params.get('generator', 'N/A'),
                        'LOSS': h_params.get('loss', 'N/A'),
                        'TEST_DATASET': test_dataset,
                        'SUBJECT': subject,
                        **standardized_metrics
                    }
                    all_results.append(record)

    if not all_results:
        print("Nessun risultato trovato. Restituisco strutture vuote.")
        return {}, pd.DataFrame()
