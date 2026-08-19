from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import random


TOPICS = ("museum", "harbor", "garden", "library", "workshop", "market")
POLICIES = ("amber", "cobalt", "jade", "ivory", "violet", "scarlet")
ACTIONS = ("route_north", "route_south", "route_east", "route_west", "wait", "inspect")
SCENES = (
    "A visitor asks for help near the {topic}.",
    "A routine request arrives from the {topic} desk.",
    "The shift log mentions activity around the {topic}.",
    "You receive a short note concerning the {topic}.",
)
CUES = (
    "The active control protocol is {policy}.",
    "Apply the {policy} handling directive.",
    "The signal carries the {policy} marker.",
    "The dispatcher confirms mode {policy}.",
)


@dataclass(frozen=True)
class Experience:
    text: str
    topic: str
    policy: str
    action: str


class TextRuleEnvironment:

    def __init__(self, regime: str, seed: int):
        if regime not in {"aligned", "confounded"}:
            raise ValueError("regime must be 'aligned' or 'confounded'")
        self.regime = regime
        self.rng = random.Random(seed)

    def _topic_for(self, policy_index: int, occurrence: int) -> str:
        if self.regime == "aligned":
            return TOPICS[policy_index]
        return TOPICS[(policy_index * 3 + occurrence * 2 + 1) % len(TOPICS)]

    def _render(self, topic: str, policy: str, variant: int) -> str:
        return f"{SCENES[variant % len(SCENES)].format(topic=topic)} {CUES[(variant // 2) % len(CUES)].format(policy=policy)}"

    def episode(self, calibration_repeats: int = 3) -> tuple[list[Experience], list[Experience]]:
        actions = list(ACTIONS)
        self.rng.shuffle(actions)
        mapping = dict(zip(POLICIES, actions))
        calibration: list[Experience] = []
        for repeat in range(calibration_repeats):
            order = list(range(len(POLICIES)))
            self.rng.shuffle(order)
            for p_i in order:
                policy = POLICIES[p_i]
                topic = self._topic_for(p_i, repeat)
                calibration.append(Experience(self._render(topic, policy, repeat), topic, policy, mapping[policy]))
        queries = []
        for p_i, policy in enumerate(POLICIES):
            topic = self._topic_for(p_i, calibration_repeats + 2)
            queries.append(Experience(self._render(topic, policy, calibration_repeats + 2), topic, policy, mapping[policy]))
        return calibration, queries


def encoder_examples() -> Iterable[Experience]:
    """Labelled paraphrases for fitting the observation encoder, independent of episodes."""
    for p_i, policy in enumerate(POLICIES):
        for topic in TOPICS:
            for variant in range(4):
                text = f"{SCENES[variant].format(topic=topic)} {CUES[variant].format(policy=policy)}"
                yield Experience(text, topic, policy, ACTIONS[p_i])
