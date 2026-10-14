"""Run ExperimentIQ benchmark against the live pipeline."""
import sys
sys.path.insert(0, "c:/Users/rshit/Downloads/ExperimentIQ")
from app.pipeline import RAGPipeline
from app.router import route_query
from app.eval.benchmark import BenchmarkQuestion, evaluate_question
from app.eval.metrics import calculate_metrics

QUESTIONS = [
    BenchmarkQuestion("q01", "which flow config gave the best validation NLL?", "structured", "config", "metric_search"),
    BenchmarkQuestion("q02", "which run has the highest accuracy?", "structured", "accuracy", "metric_search"),
    BenchmarkQuestion("q03", "show me runs with val_nll < 0.8", "structured", "val_nll", "metric_range"),
    BenchmarkQuestion("q04", "what do the logs say about the training error?", "unstructured", "ERROR", "log_retrieval"),
    BenchmarkQuestion("q05", "compare configs across all runs", "structured", "config", "comparison"),
    BenchmarkQuestion("q06", "explain the configuration for flow_a", "unstructured", "flow_a", "config_qa"),
    BenchmarkQuestion("q07", "which dataset gave the lowest val_nll?", "structured", "val_nll", "metric_search"),
    BenchmarkQuestion("q08", "what happened in epoch 3 according to logs?", "unstructured", "epoch 3", "log_retrieval"),
    BenchmarkQuestion("q09", "list runs with accuracy > 0.85", "structured", "accuracy", "metric_range"),
    BenchmarkQuestion("q10", "tell me about config_v3 on cifar10", "unstructured", "config_v3", "config_qa"),
]

def main():
    pipe = RAGPipeline()
    results = []
    for q in QUESTIONS:
        decision = route_query(q.question)
        try:
            out = pipe.answer(q.question, route="auto", top_k=3)
            ans, actual_route = out.answer, out.route
        except Exception as e:
            ans, actual_route = f"ERROR: {e}", decision.path
        r = evaluate_question(q, actual_route, ans)
        results.append(r)
        print(f"{q.id} exp_route={q.expected_route} got={actual_route} route_ok={r.route_correct} ans_ok={r.answer_correct}")
        print(f"   A: {ans[:160]}")
    m = calculate_metrics(results)
    print("\nMETRICS:", m)

if __name__ == "__main__":
    main()
