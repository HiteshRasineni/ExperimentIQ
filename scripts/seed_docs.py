"""Seed unstructured docs (configs + logs) into Qdrant/in-memory store."""
import sys
sys.path.insert(0, "c:/Users/rshit/Downloads/ExperimentIQ")
from app.unstructured.client import QdrantClientWrapper
DOCS = [
    {"id": "cfg-flow-a", "text": "flow_a config: distributed training, gradient_accumulation=2, mixed_precision=true, optimizer=adam, scheduler=step. Used for cifar10 config_v1 runs.", "type": "config", "title": "flow_a config"},
    {"id": "cfg-flow-b", "text": "flow_b config: standard training, gradient_accumulation=4, mixed_precision=true, optimizer=sgd momentum=0.9, scheduler=cosine. Used for config_v2.", "type": "config", "title": "flow_b config"},
    {"id": "log-err-01", "text": "ERROR epoch 3: training loss spiked to 2.4, gradient norm 12.5 exceeded threshold. Log trace shows NaN in batch 44 on flow_c. Reduced learning rate to recover.", "type": "log", "title": "training error log"},
    {"id": "log-info-01", "text": "INFO run completed: best val_nll 0.32 on config_v3 cifar10 flow_a. Training trace clean, no warnings after epoch 2.", "type": "log", "title": "training completion log"},
    {"id": "note-01", "text": "Experiment note: config_v3 consistently gives the best validation NLL on cifar10 across flow_a runs. Prefer it for final comparison.", "type": "note", "title": "experiment note"},
]
def main():
    c = QdrantClientWrapper()
    n = c.add_documents(DOCS)
    print(f"SUCCESS: added {n} docs (qdrant={c._use_qdrant}), count={c.count()}")
if __name__ == "__main__":
    main()
