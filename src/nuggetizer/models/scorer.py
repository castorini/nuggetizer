import ast
import logging
from typing import List, Dict, Optional
from ..core.base import BaseNuggetScorer
from ..core.llm import LLMHandler
from ..core.types import Nugget, ScoredNugget, NuggetScoreMode

class NuggetScorer(BaseNuggetScorer):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_keys: Optional[str] = None,
        mode: NuggetScoreMode = NuggetScoreMode.VITAL_OKAY,
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

    def _create_prompt(self, nuggets: List[Nugget]) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": "You are NuggetizeScoreLLM, an intelligent assistant that can label atomic nuggets based on their importance."
            },
            {
                "role": "user",
                "content": f"""Label each nugget as either 'vital' or 'okay' based on its importance. Vital nuggets represent concepts that must be present in a "good" answer, while okay nuggets contribute worthwhile but non-essential information.

Nuggets to score:
{[nugget.text for nugget in nuggets]}

Return only a Python list of labels (["vital", "okay", ...]) in the same order as the input nuggets.

Labels:"""
            }
        ]
        return messages

    def score(self, nuggets: List[Nugget]) -> List[ScoredNugget]:
        if self.log_level >= 1:
            self.logger.info("Starting nugget scoring process")
        scored_nuggets = []
        start = 0
        
        while start < len(nuggets):
            end = min(start + self.window_size, len(nuggets))
            window_nuggets = nuggets[start:end]
            
            if self.log_level >= 1:
                self.logger.info(f"Processing window {start} to {end} of {len(nuggets)} nuggets")
            
            prompt = self._create_prompt(window_nuggets)
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
                    importance_labels = ast.literal_eval(response)
                
                    for nugget, importance in zip(window_nuggets, importance_labels):
                        scored_nuggets.append(
                            ScoredNugget(text=nugget.text, importance=importance.lower())
                        )
                    if self.log_level >= 1:
                        self.logger.info(f"Successfully processed window with {len(window_nuggets)} nuggets")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to parse response: {str(e)}")
                    trial_count -= 1
                    temperature = 0.2
                    if trial_count == 0:
                        self.logger.error("Failed to parse response after 500 attempts")
                        scored_nuggets.extend([
                            ScoredNugget(text=nugget.text, importance="failed")
                            for nugget in window_nuggets
                        ])
            
            start += self.window_size
        
        if self.log_level >= 1:
            self.logger.info(f"Completed scoring process with {len(scored_nuggets)} nuggets")
        return scored_nuggets

    def score_batch(self, nuggets_list: List[List[Nugget]]) -> List[List[ScoredNugget]]:
        return [self.score(nuggets) for nuggets in nuggets_list]
