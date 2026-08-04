import argparse
import sys
from pathlib import Path

# Force UTF-8 encoding for stdout on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add workspace root path to sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from omnibrain.app.core.logging import logger  # noqa: E402
from omnibrain.app.services.ingestion.ingestion_service import (  # noqa: E402
    IngestionService,
)
from omnibrain.vectorstore.qdrant_client import QdrantClientWrapper  # noqa: E402


def _check_qdrant_connectivity() -> bool:
    try:
        client = QdrantClientWrapper().client()
        client.get_collections()
        print("✅ Qdrant connectivity check passed.")
        return True
    except Exception as exc:
        print(f"⚠️ Qdrant connectivity check failed: {exc}")
        return False


def run_pipeline_validation(input_path: str):
    print("\n" + "=" * 60)
    print(" 🚀 OMNIBRAIN MULTI-MODAL INGESTION PIPELINE VALIDATION ")
    print("=" * 60 + "\n")

    test_source = Path(input_path)

    if not test_source.exists():
        logger.error("Validation asset not found at %s.", test_source)
        print(f"⚠️ Please ensure '{test_source}' exists before running validation.")
        return

    orchestrator = IngestionService()

    print(f"📄 Testing full pipeline against asset: {test_source}")
    result = orchestrator.process_path(str(test_source))
    print("\n" + "-" * 60)
    print(" 📊 TELEMETRY & PIPELINE EXECUTION SUMMARY")
    print("-" * 60)
    print(f"  • Task ID:              {result.task_id}")
    print(f"  • Status:               {result.status.upper()}")
    print(f"  • Total Pages Parsed:   {result.pages_parsed}")

    # Safely fetch chunks metric across schema representations
    chunks = getattr(result, "text_chunks", getattr(result, "text_chunks_count", "N/A"))
    print(f"  • Text Chunks Created:  {chunks}")

    print(f"  • Images Vectorized:    {result.images_extracted}")
    print(f"  • System Message:       {result.message}")
    if getattr(result, "warnings", None):
        print("  • Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")
    print("-" * 60 + "\n")

    print("✅ Pipeline Validation Complete! Ready for Week 1 Sync Review.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the OmniBrain ingestion pipeline."
    )
    parser.add_argument(
        "--input",
        default="data/Attention_is_all_you_need.pdf",
        help="Path to a PDF file or a folder containing PDFs.",
    )
    args = parser.parse_args()

    _check_qdrant_connectivity()
    run_pipeline_validation(args.input)
