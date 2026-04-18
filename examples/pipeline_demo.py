#!/usr/bin/env python3
import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import create_sample_request

from nuggetizer.core.metrics import calculate_nugget_scores
from nuggetizer.core.types import Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.utils.display import print_assigned_nuggets, print_nuggets


async def process_request(
    request: Request,
    model: str,
    use_azure_openai: bool,
    use_openrouter: bool,
    use_vllm: bool,
    vllm_port: int,
    log_level: int,
) -> None:
    """Process a request through the nuggetizer pipeline."""
    start_time = time.time()

    print("🚀 Initializing components...")
    # Initialize components - API keys and Azure config are loaded automatically

    # Option 1: Single model for all components
    nuggetizer1 = Nuggetizer(
        model=model,
        use_azure_openai=use_azure_openai,
        use_openrouter=use_openrouter,
        use_vllm=use_vllm,
        vllm_port=vllm_port,
        log_level=log_level,
    )

    # Option 2: Different models for each component
    # nuggetizer2 = Nuggetizer(
    #     creator_model="gpt-4o",
    #     scorer_model="gpt-3.5-turbo",
    #     assigner_model="gpt-4o",
    #     use_azure_openai=use_azure_openai,
    #     log_level=log_level
    # )

    # Use nuggetizer1 for this example
    nuggetizer = nuggetizer1

    # Extract and score nuggets
    print("\n📝 Extracting and scoring nuggets...")
    create_start = time.time()
    scored_nuggets = await nuggetizer.async_create(request)
    create_time = time.time() - create_start
    print(f"Found {len(scored_nuggets)} nuggets (took {create_time:.2f}s):")
    print_nuggets(scored_nuggets)

    # Assign nuggets to documents in parallel
    print("\n🎯 Assigning nuggets to documents...")
    assign_start = time.time()
    # Create tasks for parallel assignment
    assignment_tasks = []
    for doc in request.documents:
        print(f"\nDocument: {doc.docid}")
        print("Segment:", doc.segment)
        assignment_tasks.append(
            nuggetizer.async_assign(request.query.text, doc.segment, scored_nuggets)
        )

    # Run all assignments in parallel
    assigned_nuggets_list = await asyncio.gather(*assignment_tasks)
    assign_time = time.time() - assign_start
    print(f"\nAssignment completed in {assign_time:.2f}s")

    # Process results
    for doc, assigned_nuggets in zip(
        request.documents, assigned_nuggets_list, strict=True
    ):
        print_assigned_nuggets(doc, assigned_nuggets)

        # Calculate metrics for this document
        nugget_list = [
            {"text": n.text, "importance": n.importance, "assignment": n.assignment}
            for n in assigned_nuggets
        ]
        metrics = calculate_nugget_scores(request.query.qid, nugget_list)
        print("\nMetrics:")
        print(f"  Strict Vital Score: {metrics.strict_vital_score:.2f}")
        print(f"  Strict All Score: {metrics.strict_all_score:.2f}")
        print(f"  Vital Score: {metrics.vital_score:.2f}")
        print(f"  All Score: {metrics.all_score:.2f}")

    total_time = time.time() - start_time
    print("\n⏱️ Timing Summary:")
    print(f"  Creation time: {create_time:.2f}s")
    print(f"  Assignment time: {assign_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")


async def main() -> None:
    """Run the default pipeline demo."""
    parser = argparse.ArgumentParser(description="Run the default pipeline demo")
    parser.add_argument(
        "--use_azure_openai", action="store_true", help="Use Azure OpenAI"
    )
    parser.add_argument(
        "--use_openrouter", action="store_true", help="Use OpenRouter API"
    )
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM local server")
    parser.add_argument(
        "--vllm_port", type=int, default=8000, help="vLLM server port (default: 8000)"
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model to use")
    parser.add_argument("--log_level", type=int, default=0, help="Log level")
    args = parser.parse_args()

    print("🔧 Starting Nuggetizer pipeline demo...")
    print(f"Using model: {args.model}")

    request = create_sample_request()
    await process_request(
        request,
        args.model,
        args.use_azure_openai,
        args.use_openrouter,
        args.use_vllm,
        args.vllm_port,
        args.log_level,
    )
    print("\n✨ Example completed!")


if __name__ == "__main__":
    asyncio.run(main())
