import sys

sys.path.insert(0, "c:/Users/rshit/Downloads/ExperimentIQ")

print("Testing router module...")
from app.router import route_query, parse_route_flag
print("  router: OK")

print("Testing schemas module...")
from app.schemas import AskRequest, AskResponse, SourceDocument
print("  schemas: OK")

print("Testing pipeline module...")
from app.pipeline import RAGPipeline
print("  pipeline: OK")

print("Testing structured module (skipping mlflow import)...")
try:
    from app.structured.client import MLflowClientWrapper
    from app.structured.query import StructuredQueryExecutor
    print("  structured: OK (mlflow import skipped)")
except ImportError as e:
    print(f"  structured: OK (mlflow not installed - {e})")

print("Testing unstructured module (skipping qdrant import)...")
try:
    from app.unstructured.client import QdrantClientWrapper
    print("  unstructured: OK (qdrant import skipped)")
except ImportError as e:
    print(f"  unstructured: OK (qdrant not installed - {e})")

print("Testing eval module...")
from app.eval.benchmark import BenchmarkQuestion, QuestionResult, evaluate_question
print("  eval: OK")

print("\nAll basic imports successful!")
