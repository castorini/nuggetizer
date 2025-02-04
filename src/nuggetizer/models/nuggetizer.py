import ast
import logging
from typing import List, Dict, Optional
from ..core.base import BaseNuggetizer
from ..core.llm import LLMHandler
from ..core.types import (
    Request, Nugget, ScoredNugget, AssignedScoredNugget,
    NuggetMode, NuggetScoreMode, NuggetAssignMode
)

class Nuggetizer(BaseNuggetizer):
    def __init__(
        self,
        model: Optional[str] = None,
        creator_model: Optional[str] = "gpt-4o",
        scorer_model: Optional[str] = "gpt-4o",
        assigner_model: Optional[str] = "gpt-4o",
        api_keys: Optional[str] = None,
        creator_mode: NuggetMode = NuggetMode.ATOMIC,
        scorer_mode: NuggetScoreMode = NuggetScoreMode.VITAL_OKAY,
        assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
        window_size: Optional[int] = None,
        creator_window_size: Optional[int] = 10,
        scorer_window_size: Optional[int] = 10,
        assigner_window_size: Optional[int] = 10,
        max_nuggets: Optional[int] = None,
        creator_max_nuggets: Optional[int] = 30,
        scorer_max_nuggets: Optional[int] = 30,
        log_level: int = 0,
        **llm_kwargs
    ):
        self.creator_mode = creator_mode
        self.scorer_mode = scorer_mode
        self.assigner_mode = assigner_mode
        if window_size is not None:
            self.creator_window_size = window_size
            self.scorer_window_size = window_size
            self.assigner_window_size = window_size
        else:
            self.creator_window_size = creator_window_size
            self.scorer_window_size = scorer_window_size
            self.assigner_window_size = assigner_window_size
        
        # Initialize LLM handlers for each component
        if model is not None:
            creator_model = model
            scorer_model = model
            assigner_model = model

        self.creator_llm = LLMHandler(creator_model, api_keys, **llm_kwargs)
        self.scorer_llm = LLMHandler(scorer_model, api_keys, **llm_kwargs)
        self.assigner_llm = LLMHandler(assigner_model, api_keys, **llm_kwargs)
        
        if max_nuggets is not None:
            self.creator_max_nuggets = max_nuggets
            self.scorer_max_nuggets = max_nuggets
        else:
            self.creator_max_nuggets = creator_max_nuggets
            self.scorer_max_nuggets = scorer_max_nuggets
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if log_level > 0 else logging.WARNING)
        self.log_level = log_level
        if self.log_level >= 1:
            self.logger.info(f"Initialized Nuggetizer with models: {creator_model}, {scorer_model}, {assigner_model}")

    def _create_nugget_prompt(self, request: Request, start: int, end: int, nuggets: List[str]) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeLLM, an intelligent assistant that can update a list of atomic nuggets to best provide all the information required for the query."
            },
            {
                "role": "user",
                "content": self._get_nugget_prompt_content(request, start, end, nuggets)
            }
        ]
        return messages

    def _get_nugget_prompt_content(self, request: Request, start: int, end: int, nuggets: List[str]) -> str:
        context = "\n".join([
            f"[{i+1}] {doc.segment}" 
            for i, doc in enumerate(request.documents[start:end])
        ])
        
        return f"""Update the list of atomic nuggets of information (1-12 words), if needed, so they best provide the information required for the query. Leverage only the initial list of nuggets (if exists) and the provided context (this is an iterative process).  Return only the final list of all nuggets in a Pythonic list format (even if no updates). Make sure there is no redundant information. Ensure the updated nugget list has at most {self.creator_max_nuggets} nuggets (can be less), keeping only the most vital ones. Order them in decreasing order of importance. Prefer nuggets that provide more interesting information.

Search Query: {request.query.text}
Context:
{context}
Search Query: {request.query.text}
Initial Nugget List: {nuggets}
Initial Nugget List Length: {len(nuggets)}

Only update the list of atomic nuggets (if needed, else return as is). Do not explain. Always answer in short nuggets (not questions). List in the form ["a", "b", ...] and a and b are strings with no mention of ".
Updated Nugget List:"""

    def _create_score_prompt(self, query: str, nuggets: List[Nugget]) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeScoreLLM, an intelligent assistant that can label a list of atomic nuggets based on their importance for a given search query."
            },
            {
                "role": "user",
                "content": f"""Based on the query, label each of the {len(nuggets)} nuggets either a vital or okay based on the following criteria. Vital nuggets represent concepts that must be present in a “good” answer; on the other hand, okay nuggets contribute worthwhile information about the target but are not essential. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget.

Search Query: {query}
Nugget List: {[nugget.text for nugget in nuggets]}

Only return the list of labels (List[str]). Do not explain.
Labels:"""
            }
        ]
        return messages

    def _create_assign_prompt(self, query: str, context: str, nuggets: List[ScoredNugget]) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeAssignerLLM, an intelligent assistant that can label a list of atomic nuggets based on if they are captured by a given passage."
            },
            {
                "role": "user",
                "content": self._get_assign_prompt_content(query, context, nuggets)
            }
        ]
        return messages

    def _get_assign_prompt_content(self, query: str, context: str, nuggets: List[ScoredNugget]) -> str:
        nugget_texts = [nugget.text for nugget in nuggets]
        
        if self.assigner_mode == NuggetAssignMode.SUPPORT_GRADE_2:
            instruction = f"""Based on the query and passage, label each of the {len(nuggets)} nuggets either as support or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support; otherwise, label them as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget."""
        else:
            instruction = f"""Based on the query and passage, label each of the {len(nuggets)} nuggets either as support, partial_support, or not_support using the following criteria. A nugget that is fully captured in the passage should be labeled as support. A nugget that is partially captured in the passage should be labeled as partial_support. If the nugget is not captured at all, label it as not_support. Return the list of labels in a Pythonic list format (type: List[str]). The list should be in the same order as the input nuggets. Make sure to provide a label for each nugget."""
            
        return f"""{instruction}

Search Query: {query}
Passage: {context}
Nugget List: {nugget_texts}
Only return the list of labels (List[str]). Do not explain.
Labels:"""

    def create(self, request: Request) -> List[ScoredNugget]:
        if self.log_level >= 1:
            self.logger.info("Starting nugget creation process")
            self.logger.info(f"Processing request with {len(request.documents)} documents")
        
        start = 0
        current_nuggets: List[str] = []
        
        while start < len(request.documents):
            end = min(start + self.creator_window_size, len(request.documents))
            
            if self.log_level >= 1:
                self.logger.info(f"Processing window {start} to {end} of {len(request.documents)} documents")
            
            prompt = self._create_nugget_prompt(request, start, end, current_nuggets)
            if self.log_level >= 2:
                self.logger.info(f"Generated prompt:\n{prompt}")
            
            temperature = 0.0
            trial_count = 500
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting LLM call (trial {500-trial_count+1})")
                    response, _ = self.creator_llm.run(prompt, temperature=temperature)
                    if self.log_level >= 2:
                        self.logger.info(f"Raw LLM response:\n{response}")
                    response = response.replace("```python", "").replace("```", "").strip()
                    nugget_texts = ast.literal_eval(response)
                    current_nuggets = nugget_texts[:self.creator_max_nuggets]  # Ensure max nuggets
                    if self.log_level >= 1:
                        self.logger.info(f"Successfully processed window, current nugget count: {len(current_nuggets)}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to parse response: {str(e)}")
                    temperature = 0.2
                    trial_count -= 1
                    if trial_count == 0:
                        self.logger.error("Failed to parse response after 500 attempts")
            
            start += self.creator_window_size
            if self.log_level >= 1:
                self.logger.info(f"Moving window by stride {self.creator_window_size}, new start: {start}")
        
        # Score the nuggets
        nuggets = [Nugget(text=text) for text in current_nuggets]
        scored_nuggets = []
        start = 0
        
        while start < len(nuggets):
            end = min(start + self.scorer_window_size, len(nuggets))
            window_nuggets = nuggets[start:end]
            
            prompt = self._create_score_prompt(request.query.text, window_nuggets)
            trial_count = 500
            temperature = 0.0
            while trial_count > 0:
                try:
                    response, _ = self.scorer_llm.run(prompt, temperature=temperature)
                    response = response.replace("```python", "").replace("```", "").strip()
                    importance_labels = ast.literal_eval(response)
                
                    for nugget, importance in zip(window_nuggets, importance_labels):
                        scored_nuggets.append(
                            ScoredNugget(text=nugget.text, importance=importance.lower())
                        )
                    break
                except Exception as e:
                    trial_count -= 1
                    temperature = 0.2
                    if trial_count == 0:
                        scored_nuggets.extend([
                            ScoredNugget(text=nugget.text, importance="failed")
                            for nugget in window_nuggets
                        ])
            
            start += self.scorer_window_size
        # # First sort by importance then position and then take :self.scorer_max_nuggets
        scored_nuggets = sorted(scored_nuggets, 
                       key=lambda x: (0 if x.importance == 'vital' else 1, 
                                    scored_nuggets.index(x)))[:self.scorer_max_nuggets]
        
        if self.log_level >= 1:
            self.logger.info(f"Completed nugget creation with {len(scored_nuggets)} nuggets")
        return scored_nuggets

    def assign(self, query: str, context: str, nuggets: List[ScoredNugget]) -> List[AssignedScoredNugget]:
        if context.strip() == "":
            return [AssignedScoredNugget(text=nugget.text, importance=nugget.importance, assignment='not_support') for nugget in nuggets]
        
        if self.log_level >= 1:
            self.logger.info("Starting nugget assignment process")
            self.logger.info(f"Processing {len(nuggets)} nuggets")
        
        assigned_nuggets = []
        start = 0
        
        while start < len(nuggets):
            end = min(start + self.assigner_window_size, len(nuggets))
            window_nuggets = nuggets[start:end]
            
            if self.log_level >= 1:
                self.logger.info(f"Processing window {start} to {end} of {len(nuggets)} nuggets")
            
            prompt = self._create_assign_prompt(query, context, window_nuggets)
            if self.log_level >= 2:
                self.logger.info(f"Generated prompt:\n{prompt}")
            
            trial_count = 500
            temperature = 0.0
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting LLM call (trial {500-trial_count+1})")
                    response, _ = self.assigner_llm.run(prompt, temperature=temperature)
                    if self.log_level >= 2:
                        self.logger.info(f"Raw LLM response:\n{response}")
                    response = response.replace("```python", "").replace("```", "").strip()
                    assignments = ast.literal_eval(response)
                    for nugget, assignment in zip(window_nuggets, assignments):
                        assigned_nuggets.append(
                            AssignedScoredNugget(
                                text=nugget.text,
                                importance=nugget.importance,
                                assignment=assignment.lower()
                            )
                        )
                    if self.log_level >= 1:
                        self.logger.info(f"Successfully processed window with {len(window_nuggets)} nuggets")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to parse response: {str(e)}")
                    if trial_count > 0:
                        trial_count -= 1
                        temperature = 0.2
                    if trial_count == 0:
                        self.logger.error("Failed to parse response after 500 attempts")
                        assigned_nuggets.extend([
                            AssignedScoredNugget(text=nugget.text, importance=nugget.importance, assignment="failed")
                            for nugget in window_nuggets
                        ])
            
            start += self.assigner_window_size
        
        if self.log_level >= 1:
            self.logger.info(f"Completed assignment process with {len(assigned_nuggets)} nuggets")
        return assigned_nuggets

    def create_batch(self, requests: List[Request]) -> List[List[ScoredNugget]]:
        return [self.create(request) for request in requests]

    def assign_batch(
        self,
        queries: List[str],
        contexts: List[str],
        nuggets_list: List[List[ScoredNugget]]
    ) -> List[List[AssignedScoredNugget]]:
        return [
            self.assign(query, context, nuggets)
            for query, context, nuggets in zip(queries, contexts, nuggets_list)
        ]
