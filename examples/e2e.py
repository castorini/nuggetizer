#!/usr/bin/env python3
import argparse

from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.core.metrics import calculate_nugget_scores


def create_sample_request() -> Request:
    """Create a sample request with a query and documents."""
    query = Query(
        qid="sample-1",
        text="What are the key benefits and features of Python programming language?"
    )
    
    documents = [
        Document(
            docid="doc1",
            segment="""Python is renowned for its simplicity and readability, making it an excellent choice for beginners. 
            Its extensive standard library provides built-in support for many programming tasks."""
        ),
        Document(
            docid="doc2",
            segment="""Python supports multiple programming paradigms including object-oriented, imperative, and functional 
            programming. It has a large and active community that contributes to thousands of third-party packages."""
        ),
        Document(
            docid="doc3",
            segment="""Python's dynamic typing and automatic memory management help developers focus on solving problems 
            rather than managing low-level details. It's widely used in web development, data science, and AI."""
        ),
        Document(
            docid="doc4",
            segment="""Python is open source. And some features included are:
            - Lambda functions are anonymous functions that can be used to create quick, one-time-use functions.
            - List comprehensions are a way to create lists in a concise manner.
            - Generators are a way to create iterators in a concise manner.
            - Decorators are a way to modify or enhance functions or methods.
            - Context managers are a way to manage resources in a concise manner.
            - Async/await are a way to write asynchronous code in a concise manner.
            - Type hints are a way to add type information to variables and function parameters.
            - etc.
            """
        )
    ]
    
    return Request(query=query, documents=documents)


def process_request(request: Request, model: str, use_azure_openai: bool) -> None:
    """Process a request through the nuggetizer pipeline."""
    print("🚀 Initializing components...")
    # Initialize components - API keys and Azure config are loaded automatically
    
    # Option 1: Single model for all components
    nuggetizer1 = Nuggetizer(model=model, use_azure_openai=use_azure_openai)
    
    # Option 2: Different models for each component
    nuggetizer2 = Nuggetizer(
        creator_model="gpt-4o",
        scorer_model="gpt-3.5-turbo",
        assigner_model="gpt-4o",
        use_azure_openai=use_azure_openai
    )
    
    # Use nuggetizer1 for this example
    nuggetizer = nuggetizer1
    
    # Extract and score nuggets
    print("\n📝 Extracting and scoring nuggets...")
    scored_nuggets = nuggetizer.create(request)
    print(f"Found {len(scored_nuggets)} nuggets:")
    for i, nugget in enumerate(scored_nuggets, 1):
        print(f"{i}. {nugget.text} (Importance: {nugget.importance})")
    
    # Assign nuggets to documents
    print("\n🎯 Assigning nuggets to documents...")
    for doc in request.documents:
        print(f"\nDocument: {doc.docid}")
        print("Segment:", doc.segment)
        assigned_nuggets = nuggetizer.assign(doc.segment, scored_nuggets)
        print("\nAssignments:")
        for nugget in assigned_nuggets:
            print(f"- {nugget.text}")
            print(f"  Importance: {nugget.importance}")
            print(f"  Assignment: {nugget.assignment}")
        
        # Calculate metrics for this document
        nugget_list = [
            {
                'text': n.text,
                'importance': n.importance,
                'assignment': n.assignment
            }
            for n in assigned_nuggets
        ]
        metrics = calculate_nugget_scores(request.query.qid, nugget_list)
        print("\nMetrics:")
        print(f"  Strict Vital Score: {metrics.strict_vital_score:.2f}")
        print(f"  Strict All Score: {metrics.strict_all_score:.2f}")
        print(f"  Vital Score: {metrics.vital_score:.2f}")
        print(f"  All Score: {metrics.all_score:.2f}")


def main():
    """Run the e2e example."""
    parser = argparse.ArgumentParser(description='Run the e2e example')
    parser.add_argument('--use_azure_openai', action='store_true', help='Use Azure OpenAI')
    parser.add_argument('--model', type=str, default="gpt-4o", help='Model to use')
    args = parser.parse_args()

    print("🔧 Starting E2E Nuggetizer Example...")
    print(f"Using model: {args.model}")
    request = create_sample_request()
    process_request(request, args.model, args.use_azure_openai)
    print("\n✨ Example completed!")


if __name__ == "__main__":
    main() 