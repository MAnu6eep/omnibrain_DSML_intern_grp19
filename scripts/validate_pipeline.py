import sys
from pathlib import Path

from omnibrain.app.core.logging import logger
from omnibrain.app.services.ingestion.ingestion_service import IngestionService

# Add workspace root path to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_pipeline_validation():
    print("\n" + "=" * 60)
    print(" 🚀 OMNIBRAIN MULTI-MODAL INGESTION PIPELINE VALIDATION ")
    print("=" * 60 + "\n")

    test_pdf = "data/Attention_is_all_you_need.pdf"

    if not Path(test_pdf).exists():
        logger.error(f"Validation PDF asset not found at {test_pdf}.")
        print(f"⚠️ Please ensure '{test_pdf}' exists in your data/ directory.")
        return

    orchestrator = IngestionService()

    print(f"📄 Testing full pipeline against asset: {test_pdf}")
    result = orchestrator.process_pdf(test_pdf)
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
    print("-" * 60 + "\n")

    print("✅ Pipeline Validation Complete! Ready for Week 1 Sync Review.\n")


if __name__ == "__main__":
    run_pipeline_validation()
