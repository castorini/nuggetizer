#!/usr/bin/env python3
from nuggetizer.core.types import Query, Document, Request
from nuggetizer.models.nuggetizer import Nuggetizer
from nuggetizer.models.scorer import NuggetScorer
from nuggetizer.models.assigner import NuggetAssigner
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


def process_request(request: Request) -> None:
    """Process a request through the nuggetizer pipeline."""
    print("🚀 Initializing components...")
    # Initialize components - API keys and Azure config are loaded automatically
    nuggetizer = Nuggetizer(model="gpt-4o")
    scorer = NuggetScorer(model="gpt-4o")
    assigner = NuggetAssigner(model="gpt-4o")
    
    # Extract nuggets
    print("\n📝 Extracting nuggets...")
    nuggets, _ = nuggetizer.process(request)
    print(f"Found {len(nuggets)} nuggets:")
    for i, nugget in enumerate(nuggets, 1):
        print(f"{i}. {nugget.text}")
    
    # Score nuggets
    print("\n⭐ Scoring nuggets...")
    scored_nuggets = scorer.score(nuggets)
    print("Nugget importance scores:")
    for nugget in scored_nuggets:
        print(f"- {nugget.text}: {nugget.importance}")
    
    # Assign nuggets to documents
    print("\n🎯 Assigning nuggets to documents...")
    for doc in request.documents:
        print(f"\nDocument: {doc.docid}")
        print("Segment:", doc.segment)
        assigned_nuggets = assigner.assign(doc.segment, scored_nuggets)
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
    print("🔧 Starting E2E Nuggetizer Example...")
    request = create_sample_request()
    process_request(request)
    print("\n✨ Example completed!")


if __name__ == "__main__":
    main() 