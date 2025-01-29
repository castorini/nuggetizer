from abc import ABC, abstractmethod
from typing import List, Union
from .types import (
    Request, Nugget, ScoredNugget, 
    AssignedNugget, AssignedScoredNugget
)

class BaseNuggetizer(ABC):
    @abstractmethod
    def create(self, request: Request) -> List[ScoredNugget]:
        pass

    @abstractmethod
    def assign(
        self, 
        context: str, 
        nuggets: List[ScoredNugget]
    ) -> List[AssignedScoredNugget]:
        pass

    @abstractmethod
    def create_batch(self, requests: List[Request]) -> List[List[ScoredNugget]]:
        pass

    @abstractmethod
    def assign_batch(
        self,
        contexts: List[str],
        nuggets_list: List[List[ScoredNugget]]
    ) -> List[List[AssignedScoredNugget]]:
        pass

class BaseNuggetScorer(ABC):
    @abstractmethod
    def score(self, nuggets: List[Nugget]) -> List[ScoredNugget]:
        pass

    @abstractmethod
    def score_batch(self, nuggets_list: List[List[Nugget]]) -> List[List[ScoredNugget]]:
        pass

class BaseNuggetAssigner(ABC):
    @abstractmethod
    def assign(
        self, 
        context: str, 
        nuggets: Union[List[Nugget], List[ScoredNugget]]
    ) -> Union[List[AssignedNugget], List[AssignedScoredNugget]]:
        pass

    @abstractmethod
    def assign_batch(
        self,
        contexts: List[str],
        nuggets_list: Union[List[List[Nugget]], List[List[ScoredNugget]]]
    ) -> Union[List[List[AssignedNugget]], List[List[AssignedScoredNugget]]]:
        pass
