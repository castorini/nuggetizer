import ast
import logging
from typing import List, Dict, Union, Optional
from ..core.base import BaseNuggetAssigner
from ..core.llm import LLMHandler
from ..core.types import (
    Nugget, ScoredNugget, AssignedNugget, 
    AssignedScoredNugget, NuggetAssignMode
)

class NuggetAssigner(BaseNuggetAssigner):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_keys: Optional[str] = None,
        mode: NuggetAssignMode = NuggetAssignMode.SUPPORT_GRADE_3,
        window_size: int = 10,
        log_level: int = 0,
        **llm_kwargs
    ):
        self.mode = mode
        self.window_size = window_size
        self.llm = LLMHandler(model, api_keys, **llm_kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO if log_level > 0 else logging.WARNING)
        self.log_level = log_level

    def _create_prompt(
        self, 
        context: str, 
        nuggets: Union[List[Nugget], List[ScoredNugget]]
    ) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeAssignerLLM, an intelligent assistant that can determine how well a passage supports given nuggets of information."
            },
            {
                "role": "user",
                "content": self._get_prompt_content(context, nuggets)
            }
        ]
        return messages

    def _get_prompt_content(
        self, 
        context: str, 
        nuggets: Union[List[Nugget], List[ScoredNugget]]
    ) -> str:
        nugget_texts = [nugget.text for nugget in nuggets]
        
        if self.mode == NuggetAssignMode.SUPPORT_GRADE_2:
            instruction = """Label each nugget as either 'support' or 'not_support' based on whether it is fully supported by the passage."""
        else:
            instruction = """Label each nugget as 'support' (fully supported), 'partial_support' (partially supported), or 'not_support' (not supported) based on how well it is supported by the passage."""
            
        return f"""{instruction}

Passage:
{context}

Nuggets to assess:
{nugget_texts}

Return only a Python list of labels in the same order as the input nuggets.

Labels:"""

    def assign(
        self, 
        context: str, 
        nuggets: Union[List[Nugget], List[ScoredNugget]]
    ) -> Union[List[AssignedNugget], List[AssignedScoredNugget]]:
        if context.strip() == "":
            if isinstance(nuggets[0], ScoredNugget):
                return [AssignedScoredNugget(text=nugget.text, importance=nugget.importance, assignment='not_support') for nugget in nuggets]
            else:
                return [AssignedNugget(text=nugget.text, assignment='not_support') for nugget in nuggets]
        
        if self.log_level >= 1:
            self.logger.info("Starting nugget assignment process")
            self.logger.info(f"Processing {len(nuggets)} nuggets")
        
        assigned_nuggets = []
        start = 0
        
        while start < len(nuggets):
            end = min(start + self.window_size, len(nuggets))
            window_nuggets = nuggets[start:end]
            
            if self.log_level >= 1:
                self.logger.info(f"Processing window {start} to {end} of {len(nuggets)} nuggets")
            
            prompt = self._create_prompt(context, window_nuggets)
            if self.log_level >= 2:
                self.logger.info(f"Generated prompt:\n{prompt}")
            
            trial_count = 500
            temperature = 0.0
            while trial_count > 0:
                try:
                    if self.log_level >= 1:
                        self.logger.info(f"Attempting LLM call (trial {500-trial_count+1})")
                    response, _ = self.llm.run(prompt, temperature=temperature)
                    if self.log_level >= 2:
                        self.logger.info(f"Raw LLM response:\n{response}")
                    response = response.replace("```python", "").replace("```", "").strip()
                    assignments = ast.literal_eval(response)
                    for nugget, assignment in zip(window_nuggets, assignments):
                        if isinstance(nugget, ScoredNugget):
                            assigned_nuggets.append(
                                AssignedScoredNugget(
                                    text=nugget.text,
                                    importance=nugget.importance,
                                    assignment=assignment.lower()
                                )
                            )
                        else:
                            assigned_nuggets.append(
                                AssignedNugget(
                                    text=nugget.text,
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
                            AssignedNugget(text=nugget.text, assignment="failed")
                            for nugget in window_nuggets
                        ]) if not isinstance(nugget, ScoredNugget) else [
                            AssignedScoredNugget(text=nugget.text, importance=nugget.importance, assignment="failed")
                            for nugget in window_nuggets
                        ]
            
            start += self.window_size
        
        if self.log_level >= 1:
            self.logger.info(f"Completed assignment process with {len(assigned_nuggets)} nuggets")
        return assigned_nuggets

    def assign_batch(
        self,
        contexts: List[str],
        nuggets_list: Union[List[List[Nugget]], List[List[ScoredNugget]]]
    ) -> Union[List[List[AssignedNugget]], List[List[AssignedScoredNugget]]]:
        return [
            self.assign(context, nuggets)
            for context, nuggets in zip(contexts, nuggets_list)
        ]
