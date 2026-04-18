from __future__ import annotations

from nuggetizer.core.types import Document, Query, Request


def create_sample_request() -> Request:
    query = Query(
        qid="sample-1",
        text="What are the key benefits and features of Python programming language?",
    )

    documents = [
        Document(
            docid="doc1",
            segment="""Python is renowned for its simplicity and readability, making it an excellent choice for beginners. 
            Its extensive standard library provides built-in support for many programming tasks.""",
        ),
        Document(
            docid="doc2",
            segment="""Python supports multiple programming paradigms including object-oriented, imperative, and functional 
            programming. It has a large and active community that contributes to thousands of third-party packages.""",
        ),
        Document(
            docid="doc3",
            segment="""Python's dynamic typing and automatic memory management help developers focus on solving problems 
            rather than managing low-level details. It's widely used in web development, data science, and AI.""",
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
            """,
        ),
        Document(
            docid="doc5",
            segment="""Benefits of Python Programming Language

1. Simplicity & Readability
	•	Intuitive, easy-to-read syntax reduces development time.
	•	Indentation-based structure enforces clean, readable code.

2. Versatility & General-Purpose Use
	•	Supports multiple paradigms: procedural, object-oriented, functional.
	•	Used in web development, data science, machine learning, automation, scripting, etc.

3. Extensive Libraries & Ecosystem
	•	Rich standard library (e.g., json, os, sys, datetime).
	•	Strong third-party ecosystem: NumPy, Pandas (data science), TensorFlow, PyTorch (ML), Flask, Django (web), Scrapy (web scraping).

4. Cross-Platform Compatibility
	•	Runs on Windows, macOS, Linux, and embedded systems.
	•	Code can be executed in various environments without modification.

5. Strong Community & Support
	•	Large open-source community with extensive documentation and forums (Stack Overflow, GitHub).
	•	Frequent updates and improvements from contributors worldwide.

6. Rapid Development & Prototyping
	•	Dynamic typing and interpreted nature allow fast iteration.
	•	Ideal for quick prototyping before optimizing in lower-level languages.

7. Integration & Interoperability
	•	Can interface with C/C++ (ctypes, Cython), Java (Jython), .NET (IronPython).
	•	Supports API calls via HTTP, WebSockets, and external services.

8. Scalability & Maintainability
	•	Modular design and structured code make large projects manageable.
	•	Popular in enterprise applications (Dropbox, Instagram, Reddit).

9. High Demand & Career Opportunities
	•	Widely used in industry, academia, and research.
	•	High job demand in data science, AI, software development, cybersecurity, etc.

10. Strong for AI & Data Science
	•	Dominates ML & AI fields with TensorFlow, PyTorch, and Scikit-learn.
	•	Integrated with big data tools (Apache Spark, Dask).

11. Automation & Scripting
	•	Ideal for automating repetitive tasks (web scraping, testing, system administration).
	•	Popular in DevOps workflows (Ansible, Fabric).

12. Open-Source & Free
	•	No licensing costs, freely available for any use case.
	•	Thrives on an active open-source ecosystem.

Conclusion

Python's simplicity, versatility, vast ecosystem, and strong community support make it one of the most powerful and widely used programming languages across industries.""",
        ),
    ]

    return Request(query=query, documents=documents)
