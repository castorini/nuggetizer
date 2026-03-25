from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from .types import AssignedNugget, AssignedScoredNugget, Nugget, Request, ScoredNugget


# Define a protocol for synchronous Nuggetizer
@runtime_checkable
class NuggetizerProtocol(Protocol):
    def create(self, request: Request) -> list[ScoredNugget]: ...

    def assign(
        self,
        query: str,  # Add query parameter to match implementations
        context: str,
        nuggets: list[ScoredNugget],
    ) -> list[AssignedScoredNugget]: ...

    def create_batch(self, requests: list[Request]) -> list[list[ScoredNugget]]: ...

    def assign_batch(
        self,
        queries: list[str],  # Add queries parameter to match implementations
        contexts: list[str],
        nuggets_list: list[list[ScoredNugget]],
    ) -> list[list[AssignedScoredNugget]]: ...


# Define a protocol for asynchronous Nuggetizer
@runtime_checkable
class AsyncNuggetizerProtocol(Protocol):
    def create(self, request: Request) -> Awaitable[list[ScoredNugget]]: ...

    def assign(
        self, query: str, context: str, nuggets: list[ScoredNugget]
    ) -> Awaitable[list[AssignedScoredNugget]]: ...

    def create_batch(
        self, requests: list[Request]
    ) -> Awaitable[list[list[ScoredNugget]]]: ...

    def assign_batch(
        self,
        queries: list[str],
        contexts: list[str],
        nuggets_list: list[list[ScoredNugget]],
    ) -> Awaitable[list[list[AssignedScoredNugget]]]: ...


# Keep the original ABC for backwards compatibility
class BaseNuggetizer(ABC):
    @abstractmethod
    def create(self, request: Request) -> list[ScoredNugget]:
        pass

    @abstractmethod
    def assign(
        self,
        query: str,  # Add query parameter to match implementations
        context: str,
        nuggets: list[ScoredNugget],
    ) -> list[AssignedScoredNugget]:
        pass

    @abstractmethod
    def create_batch(self, requests: list[Request]) -> list[list[ScoredNugget]]:
        pass

    @abstractmethod
    def assign_batch(
        self,
        queries: list[str],  # Add queries parameter to match implementations
        contexts: list[str],
        nuggets_list: list[list[ScoredNugget]],
    ) -> list[list[AssignedScoredNugget]]:
        pass


class BaseNuggetScorer(ABC):
    @abstractmethod
    def score(self, nuggets: list[Nugget]) -> list[ScoredNugget]:
        pass

    @abstractmethod
    def score_batch(self, nuggets_list: list[list[Nugget]]) -> list[list[ScoredNugget]]:
        pass


class BaseNuggetAssigner(ABC):
    @abstractmethod
    def assign(
        self, context: str, nuggets: list[Nugget] | list[ScoredNugget]
    ) -> list[AssignedNugget] | list[AssignedScoredNugget]:
        pass

    @abstractmethod
    def assign_batch(
        self,
        contexts: list[str],
        nuggets_list: list[list[Nugget]] | list[list[ScoredNugget]],
    ) -> list[list[AssignedNugget]] | list[list[AssignedScoredNugget]]:
        pass
