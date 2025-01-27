import ast
import logging
from typing import List, Dict, Optional, Tuple
from ..core.base import BaseNuggetizer
from ..core.llm import LLMHandler
from ..core.types import Request, Nugget, NuggetMode

class Nuggetizer(BaseNuggetizer):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_keys: Optional[str] = None,
        mode: NuggetMode = NuggetMode.ATOMIC,
        window_size: int = 10,
        stride: int = 10,
        max_nuggets: int = 30,
        log_level: int = 0,
        **llm_kwargs
    ):
        self.mode = mode
        self.window_size = window_size
        self.stride = stride
        self.max_nuggets = max_nuggets
        self.llm = LLMHandler(model, api_keys, **llm_kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if log_level > 0 else logging.WARNING)
        self.log_level = log_level

    def _create_prompt(self, request: Request, start: int, end: int, nuggets: List[str]) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
            },
            {
                "role": "user",
                "content": self._get_prompt_content(request, start, end, nuggets)
            }
        ]
        return messages

    def _get_prompt_content(self, request: Request, start: int, end: int, nuggets: List[str]) -> str:
        context = "\n".join([
            f"[{i+1}] {doc.segment}" 
            for i, doc in enumerate(request.documents[start:end])
        ])
        
        return f"""Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).

Search Query: {request.query.text}
Context:
{context}
Initial Nugget List: {nuggets}

Return only the final list of all nuggets in a Pythonic list format. Make sure there is no redundant information. Ensure the updated nugget list has at most {self.max_nuggets} nuggets, keeping only the most vital ones. Order them in decreasing order of importance.

Updated Nugget List:"""

    def process(self, request: Request) -> Tuple[List[Nugget], List[List[Nugget]]]:
        if self.log_level >= 1:
            self.logger.info("Starting nugget generation process")
            self.logger.info(f"Processing request with {len(request.documents)} documents")
        
        start = 0
        current_nuggets: List[str] = []
        nugget_trajectory: List[List[Nugget]] = [[]]
        
        while start < len(request.documents):
            end = min(start + self.window_size, len(request.documents))
            
            if self.log_level >= 1:
                self.logger.info(f"Processing window {start} to {end} of {len(request.documents)} documents")
            
            prompt = self._create_prompt(request, start, end, current_nuggets)
            if self.log_level >= 2:
                self.logger.info(f"Generated prompt:\n{prompt}")
            
            temperature = 0.0
            trial_count = 500
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting LLM call (trial {500-trial_count+1})")
                    response, _ = self.llm.run(prompt, temperature=temperature)
                    if self.log_level >= 2:
                        self.logger.info(f"Raw LLM response:\n{response}")
                    response = response.replace(
                        "```python", "").replace(
                            "```", "").strip()
                    nugget_texts = ast.literal_eval(response)
                    current_nuggets = nugget_texts[:self.max_nuggets]  # Ensure max nuggets
                    if self.log_level >= 1:
                        self.logger.info(f"Successfully processed window, current nugget count: {len(current_nuggets)}")
                    nugget_trajectory.append(current_nuggets)
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to parse response: {str(e)}")
                    temperature = 0.2
                    trial_count -= 1
                    if trial_count == 0:
                        self.logger.error("Failed to parse response after 500 attempts")
                        nugget_trajectory.append(current_nuggets)
                
            start += self.stride
            if self.log_level >= 1:
                self.logger.info(f"Moving window by stride {self.stride}, new start: {start}")
        
        if self.log_level >= 1:
            self.logger.info(f"Completed nugget generation with {len(current_nuggets)} nuggets")
        return [Nugget(text=text) for text in current_nuggets], [[Nugget(text=text) for text in nugget_list] for nugget_list in nugget_trajectory]

    def process_batch(self, requests: List[Request]) -> List[List[Nugget]]:
        return [self.process(request) for request in requests]
