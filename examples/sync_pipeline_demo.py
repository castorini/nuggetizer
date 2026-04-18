#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import create_sample_request

from nuggetizer.core.metrics import calculate_nugget_scores
from nuggetizer.core.types import Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.utils.display import print_assigned_nuggets, print_nuggets


def process_request(
    request: Request,
    model: str,
    use_azure_openai: bool,
    use_openrouter: bool,
    use_vllm: bool,
    vllm_port: int,
    log_level: int,
    print_reasoning: bool = False,
    print_trace: bool = False,
) -> None:
    """Process a request through the nuggetizer pipeline."""
    start_time = time.time()

    print("🚀 Initializing components...")

    nuggetizer1 = Nuggetizer(
        model=model,
        use_azure_openai=use_azure_openai,
        use_openrouter=use_openrouter,
        use_vllm=use_vllm,
        vllm_port=vllm_port,
        log_level=log_level,
        store_trace=print_trace,
        store_reasoning=print_reasoning,
    )

    nuggetizer = nuggetizer1

    print("\n📝 Extracting and scoring nuggets...")
    create_start = time.time()
    scored_nuggets = nuggetizer.create(request)
    create_time = time.time() - create_start
    print(f"Found {len(scored_nuggets)} nuggets (took {create_time:.2f}s):")

    if print_reasoning:
        creator_reasoning = nuggetizer.get_creator_reasoning()
        if creator_reasoning:
            print("\n🧠 Creator Reasoning:")
            print(f"   {creator_reasoning}\n")

        print("\n🧠 Reasoning for each nugget:")
        for i, nugget in enumerate(scored_nuggets, 1):
            if hasattr(nugget, "reasoning") and nugget.reasoning:
                print(f"{i}. {nugget.text}")
                print(f"   Reasoning: {nugget.reasoning}\n")

    if print_trace:
        print("\n🔍 Trace information for each nugget:")
        for i, nugget in enumerate(scored_nuggets, 1):
            if hasattr(nugget, "trace") and nugget.trace:
                t = nugget.trace
                print(f"{i}. {nugget.text}")
                print(f"   component={t.component} model={t.model} params={t.params}")
                if t.usage:
                    print(f"   usage={t.usage}")
                print()

    for i, nugget in enumerate(scored_nuggets, 1):
        importance_emoji = "⭐" if nugget.importance == "vital" else "✔️"
        print(
            f"{i}. {importance_emoji} {nugget.text} (Importance: {nugget.importance})"
        )
    print_nuggets(scored_nuggets)

    print("\n🎯 Assigning nuggets to documents...")
    assign_start = time.time()
    for doc in request.documents:
        assigned_nuggets = nuggetizer.assign(
            request.query.text, doc.segment, scored_nuggets
        )
        print_assigned_nuggets(doc, assigned_nuggets)

        if print_reasoning and hasattr(nugget, "reasoning") and nugget.reasoning:
            print(f"  Reasoning: {nugget.reasoning}")

        if print_trace and hasattr(nugget, "trace") and nugget.trace:
            t = nugget.trace
            print(f"  component={t.component} model={t.model} params={t.params}")
            if t.usage:
                print(f"  usage={t.usage}")

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

    assign_time = time.time() - assign_start
    total_time = time.time() - start_time
    print("\n⏱️ Timing Summary:")
    print(f"  Creation time: {create_time:.2f}s")
    print(f"  Assignment time: {assign_time:.2f}s")
    print(f"  Total time: {total_time:.2f}s")


def main() -> None:
    """Run the synchronous end-to-end example."""
    parser = argparse.ArgumentParser(description="Run the synchronous pipeline demo")
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
    parser.add_argument(
        "--print_reasoning", action="store_true", help="Print reasoning content"
    )
    parser.add_argument(
        "--print_trace", action="store_true", help="Print trace information"
    )
    args = parser.parse_args()

    print("🔧 Starting Synchronous Nuggetizer E2E Example...")
    print(f"Using model: {args.model}")
    request = create_sample_request()
    process_request(
        request,
        args.model,
        args.use_azure_openai,
        args.use_openrouter,
        args.use_vllm,
        args.vllm_port,
        args.log_level,
        args.print_reasoning,
        args.print_trace,
    )
    print("\n✨ Example completed!")


if __name__ == "__main__":
    main()
