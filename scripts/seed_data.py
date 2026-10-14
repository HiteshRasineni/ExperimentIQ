"""Seed ExperimentIQ with synthetic MLflow runs."""
import io, os, random, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ.setdefault("PYTHONUTF8", "1")
import mlflow
URI = "http://127.0.0.1:5000"
EXP = "Cloud-Native ML Training"
CONFIGS = ["config_v1", "config_v2", "config_v3", "config_v4"]
DATASETS = ["cifar10", "imagenet", "mnist"]
FLOWS = ["flow_a", "flow_b", "flow_c"]
def main(run_count: int = 20):
    mlflow.set_tracking_uri(URI)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(EXP)
    if exp is None:
        exp_id = client.create_experiment(EXP)
        print(f"Created experiment {EXP} id={exp_id}")
    else:
        exp_id = exp.experiment_id
        print(f"Using experiment {EXP} id={exp_id}")
    random.seed(42)
    for i in range(run_count):
        config = random.choice(CONFIGS)
        dataset = random.choice(DATASETS)
        flow = random.choice(FLOWS)
        base_nll = 0.35 if (config == "config_v3" and dataset == "cifar10") else random.uniform(0.6, 1.4)
        acc = round(random.uniform(0.78, 0.96), 4)
        lr = random.choice([1e-4, 3e-4, 1e-3, 3e-3])
        bs = random.choice([32, 64, 128])
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_param("flow_config", config)
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("flow_name", flow)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("batch_size", bs)
            for step in range(5):
                mlflow.log_metric("val_nll", round(base_nll + random.gauss(0, 0.03) - step * 0.02, 4), step=step)
                mlflow.log_metric("accuracy", round(min(0.99, acc + step * 0.005), 4), step=step)
            mlflow.set_tag("flow_name", flow)
        print(f"  run {i+1}/{run_count}: {config} {dataset} {flow} nll~{base_nll:.3f}")
    runs = client.search_runs([exp_id], max_results=100)
    print(f"SUCCESS: {len(runs)} runs in experiment")
if __name__ == "__main__":
    main()
