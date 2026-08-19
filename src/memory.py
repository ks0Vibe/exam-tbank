from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .environment import Experience
from .model import TrainedEncoder


class Memory(Protocol):
    def add(self, item: Experience) -> None: ...
    def predict(self, query: Experience) -> str | None: ...


@dataclass
class RecentMemory:
    budget: int
    items: list[Experience]

    def __init__(self, budget: int):
        self.budget, self.items = budget, []

    def add(self, item: Experience) -> None:
        self.items.append(item)
        self.items = self.items[-self.budget:]

    def predict(self, query: Experience) -> str | None:
        for item in reversed(self.items):
            if item.topic == query.topic:
                return item.action
        return self.items[-1].action if self.items else None


@dataclass
class SemanticMemory:
    budget: int
    items: list[Experience]

    def __init__(self, budget: int):
        self.budget, self.items = budget, []

    def add(self, item: Experience) -> None:
        self.items = [x for x in self.items if x.topic != item.topic]
        self.items.append(item)
        if len(self.items) > self.budget:
            self.items.pop(0)

    def predict(self, query: Experience) -> str | None:
        same = [x for x in self.items if x.topic == query.topic]
        return (same[-1] if same else self.items[-1]).action if self.items else None


@dataclass
class DecisionAwareMemory:
    budget: int
    encoder: TrainedEncoder
    items: dict[int, Experience]
    priority: dict[int, int]

    def __init__(self, budget: int, encoder: TrainedEncoder):
        self.budget, self.encoder = budget, encoder
        self.items, self.priority = {}, {}

    def add(self, item: Experience) -> None:
        key = self.encoder.decision_key(item.text)
        self.priority[key] = self.priority.get(key, 0) + 1
        self.items[key] = item
        if len(self.items) > self.budget:
            victim = min(self.items, key=lambda k: (self.priority[k], k))
            del self.items[victim]
            del self.priority[victim]

    def predict(self, query: Experience) -> str | None:
        key = self.encoder.decision_key(query.text)
        return self.items[key].action if key in self.items else None
