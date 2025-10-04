import ast
import logging
from datetime import datetime
from typing import List, Dict, Optional
from ..core.base import BaseNuggetizer
from ..core.llm import LLMHandler
from ..core.types import (
    Request, Nugget, ScoredNugget, AssignedScoredNugget, Trace,
    NuggetMode, NuggetScoreMode, NuggetAssignMode
)
from ..prompts import (
    create_nugget_prompt, get_nugget_prompt_content,
    create_score_prompt, create_assign_prompt, get_assign_prompt_content
)

class Nuggetizer(BaseNuggetizer):
    def __init__(
        self,
        model: Optional[str] = None,
        creator_model: Optional[str] = "gpt-4o",
        scorer_model: Optional[str] = "gpt-4o",
        assigner_model: Optional[str] = "gpt-4o",
        api_keys: Optional[str] = None,
        use_openrouter: bool = False,
        use_vllm: bool = False,
        openrouter_api_key: Optional[str] = None,
        vllm_port: int = 8000,
        creator_mode: NuggetMode = NuggetMode.ATOMIC,
        scorer_mode: NuggetScoreMode = NuggetScoreMode.VITAL_OKAY,
        assigner_mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
        window_size: Optional[int] = None,
        creator_window_size: int = 10,
        scorer_window_size: int = 10,
        assigner_window_size: int = 10,
        max_nuggets: Optional[int] = None,
        creator_max_nuggets: int = 30,
        scorer_max_nuggets: int = 30,
        log_level: int = 0,
        print_reasoning: bool = False,
        store_trace: bool = True,
        store_reasoning: bool = False,
        **llm_kwargs
    ):
        self.creator_mode = creator_mode
        self.scorer_mode = scorer_mode
        self.assigner_mode = assigner_mode
        self.store_trace = store_trace
        self.store_reasoning = store_reasoning
        
        # Initialize window sizes
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

        # Ensure models are not None before creating handlers
        if creator_model is None:
            creator_model = "gpt-4o"
        if scorer_model is None:
            scorer_model = "gpt-4o"
        if assigner_model is None:
            assigner_model = "gpt-4o"
            
        # Common LLM configuration shared across all handlers
        llm_config = {
            'api_keys': api_keys,
            'use_openrouter': use_openrouter,
            'use_vllm': use_vllm,
            'openrouter_api_key': openrouter_api_key,
            'vllm_port': vllm_port,
            'print_reasoning': print_reasoning,
            **llm_kwargs,
        }

        self.creator_llm = LLMHandler(creator_model, **llm_config)
        self.scorer_llm = LLMHandler(scorer_model, **llm_config)
        self.assigner_llm = LLMHandler(assigner_model, **llm_config)
        
        # Initialize max nuggets
        if max_nuggets is not None:
            self.creator_max_nuggets = max_nuggets
            self.scorer_max_nuggets = max_nuggets
        else:
            self.creator_max_nuggets = creator_max_nuggets
            self.scorer_max_nuggets = scorer_max_nuggets
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.log_level = log_level
        if log_level >= 1:
            self.logger.setLevel(logging.INFO)
        if log_level >= 2:
            self.logger.setLevel(logging.DEBUG)

    def _get_nugget_prompt_content(self, request: Request, start: int, end: int, nuggets: List[str]) -> str:
        """Get the prompt content for nugget creation."""
        return get_nugget_prompt_content(
            request, start, end, nuggets, 
            self.creator_max_nuggets, self.creator_mode
        )

    def _create_trace(self, component: str, model: str, params: Dict, messages: List[Dict[str, str]], 
                     usage: Optional[Dict] = None, raw_output: Optional[str] = None,
                     window_start: Optional[int] = None, window_end: Optional[int] = None) -> Trace:
        """Create a Trace object with the given parameters."""
        return Trace(
            component=component,
            model=model,
            params=params,
            messages=messages,
            usage=usage,
            raw_output=raw_output,
            window_start=window_start,
            window_end=window_end,
            timestamp_utc=datetime.utcnow().isoformat() + "Z"
        )

    def create(self, request: Request) -> List[ScoredNugget]:
        """Create and score nuggets from the request documents."""
        current_nuggets = []
        
        start = 0
        while start < len(request.documents):
            end = min(start + self.creator_window_size, len(request.documents))
            
            if self.log_level >= 1:
                self.logger.info(f"Processing window {start} to {end} of {len(request.documents)} documents")
            
            prompt = create_nugget_prompt(request, start, end, current_nuggets)
            if self.log_level >= 2:
                self.logger.info(f"Generated prompt:\n{prompt}")
            
            temperature = 0.0
            trial_count = 500
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting LLM call (trial {500-trial_count+1})")
                    
                    # Call LLM and get response with metadata
                    response, token_count, usage_metadata, reasoning_content = self.creator_llm.run(prompt, temperature=temperature)
                    
                    if self.log_level >= 2:
                        self.logger.info(f"Raw LLM response:\n{response}")
                except Exception as e:
                    self.logger.error(f"Failed to create nuggets: {str(e)}")
                    break
                try:
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

        # Score the nuggets
        scored_nuggets = []
        start = 0
        while start < len(current_nuggets):
            end = min(start + self.scorer_window_size, len(current_nuggets))
            
            if self.log_level >= 1:
                self.logger.info(f"Scoring window {start} to {end} of {len(current_nuggets)} nuggets")
            
            prompt = create_score_prompt(request, current_nuggets[start:end])
            if self.log_level >= 2:
                self.logger.info(f"Generated scoring prompt:\n{prompt}")
            
            temperature = 0.0
            trial_count = 500
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting scoring LLM call (trial {500-trial_count+1})")
                    
                    # Call LLM and get response with metadata
                    response, token_count, usage_metadata, reasoning_content = self.scorer_llm.run(prompt, temperature=temperature)
                    
                    if self.log_level >= 2:
                        self.logger.info(f"Raw scoring response:\n{response}")
                except Exception as e:
                    self.logger.error(f"Failed to score nuggets: {str(e)}")
                    break
                try:
                    response = response.replace("```python", "").replace("```", "").strip()
                    scores = ast.literal_eval(response)
                    
                    # Create ScoredNugget objects with trace information
                    for i, (nugget_text, score) in enumerate(zip(current_nuggets[start:end], scores)):
                        reasoning = reasoning_content if self.store_reasoning else None
                        trace = None
                        
                        if self.store_trace:
                            trace = self._create_trace(
                                component="scorer",
                                model=self.scorer_llm.model,
                                params={"temperature": temperature},
                                messages=prompt,
                                usage=usage_metadata,
                                raw_output=response,
                                window_start=start,
                                window_end=end
                            )
                        
                        scored_nuggets.append(ScoredNugget(
                            text=nugget_text,
                            importance=score.lower(),
                            reasoning=reasoning,
                            trace=trace
                        ))
                    
                    if self.log_level >= 1:
                        self.logger.info(f"Successfully scored window with {len(current_nuggets[start:end])} nuggets")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to parse scoring response: {str(e)}")
                    if trial_count > 0:
                        trial_count -= 1
                        temperature = 0.2
                    if trial_count == 0:
                        self.logger.error("Failed to parse scoring response after 500 attempts")
                        # Add nuggets with default scoring
                        for nugget_text in current_nuggets[start:end]:
                            scored_nuggets.append(ScoredNugget(
                                text=nugget_text,
                                importance="okay",
                                reasoning=reasoning_content if self.store_reasoning else None,
                                trace=self._create_trace(
                                    component="scorer",
                                    model=self.scorer_llm.model,
                                    params={"temperature": temperature},
                                    messages=prompt,
                                    usage=usage_metadata,
                                    raw_output=response,
                                    window_start=start,
                                    window_end=end
                                ) if self.store_trace else None
                            ))
            start = end

        return scored_nuggets

    def assign(self, query: str, context: str, nuggets: List[ScoredNugget]) -> List[AssignedScoredNugget]:
        """Assign scored nuggets to the given context."""
        assigned_nuggets = []
        
        start = 0
        while start < len(nuggets):
            end = min(start + self.assigner_window_size, len(nuggets))
            window_nuggets = nuggets[start:end]
            
            if self.log_level >= 1:
                self.logger.info(f"Assigning window {start} to {end} of {len(nuggets)} nuggets")
            
            prompt = create_assign_prompt(query, context, window_nuggets)
            if self.log_level >= 2:
                self.logger.info(f"Generated assignment prompt:\n{prompt}")
            
            temperature = 0.0
            trial_count = 500
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting assignment LLM call (trial {500-trial_count+1})")
                    
                    # Call LLM and get response with metadata
                    response, token_count, usage_metadata, reasoning_content = self.assigner_llm.run(prompt, temperature=temperature)
                    
                    if self.log_level >= 2:
                        self.logger.info(f"Raw assignment response:\n{response}")
                except Exception as e:
                    self.logger.error(f"Failed to assign nuggets: {str(e)}")
                    assigned_nuggets.extend([
                        AssignedScoredNugget(text=nugget.text, importance=nugget.importance, assignment="failed")
                        for nugget in window_nuggets
                    ])
                    break
                try:
                    response = response.replace("```python", "").replace("```", "").strip()
                    assignments = ast.literal_eval(response)
                    
                    # Create AssignedScoredNugget objects with trace information
                    for nugget, assignment in zip(window_nuggets, assignments):
                        reasoning = reasoning_content if self.store_reasoning else None
                        trace = None
                        
                        if self.store_trace:
                            trace = self._create_trace(
                                component="assigner",
                                model=self.assigner_llm.model,
                                params={"temperature": temperature},
                                messages=prompt,
                                usage=usage_metadata,
                                raw_output=response,
                                window_start=start,
                                window_end=end
                            )
                        
                        assigned_nuggets.append(
                            AssignedScoredNugget(
                                text=nugget.text,
                                importance=nugget.importance,
                                assignment=assignment.lower(),
                                reasoning=reasoning,
                                trace=trace
                            )
                        )
                    if self.log_level >= 1:
                        self.logger.info(f"Successfully processed window with {len(window_nuggets)} nuggets")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to parse assignment response: {str(e)}")
                    if trial_count > 0:
                        trial_count -= 1
                        temperature = 0.2
                    if trial_count == 0:
                        self.logger.error("Failed to parse assignment response after 500 attempts")
                        assigned_nuggets.extend([
                            AssignedScoredNugget(
                                text=nugget.text,
                                importance=nugget.importance,
                                assignment="failed",
                                reasoning=reasoning_content if self.store_reasoning else None,
                                trace=self._create_trace(
                                    component="assigner",
                                    model=self.assigner_llm.model,
                                    params={"temperature": temperature},
                                    messages=prompt,
                                    usage=usage_metadata,
                                    raw_output=response,
                                    window_start=start,
                                    window_end=end
                                ) if self.store_trace else None
                            )
                            for nugget in window_nuggets
                        ])
            start = end

        return assigned_nuggets

    def create_batch(self, requests: List[Request]) -> List[List[ScoredNugget]]:
        """Create nuggets for multiple requests."""
        return [self.create(request) for request in requests]

    def assign_batch(
        self,
        queries: List[str],
        contexts: List[str],
        nuggets_list: List[List[ScoredNugget]]
    ) -> List[List[AssignedScoredNugget]]:
        """Assign nuggets for multiple query-context pairs."""
        if len(queries) != len(contexts) or len(queries) != len(nuggets_list):
            raise ValueError("queries, contexts, and nuggets_list must have the same length")
        
        return [
            self.assign(query, context, nuggets)
            for query, context, nuggets in zip(queries, contexts, nuggets_list)
        ]