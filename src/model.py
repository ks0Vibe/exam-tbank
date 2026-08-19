from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass

import torch
from torch import nn

from .environment import POLICIES, encoder_examples

TOKEN_RE = re.compile(r"[a-z_]+")


class Vocabulary:
    def __init__(self, texts: list[str]):
        tokens = Counter(token for text in texts for token in TOKEN_RE.findall(text.lower()))
        self.stoi = {"<pad>": 0, "<cls>": 1, "<unk>": 2, **{token: i + 3 for i, token in enumerate(sorted(tokens))}}

    def encode(self, text: str, max_len: int = 32) -> list[int]:
        ids = [1] + [self.stoi.get(token, 2) for token in TOKEN_RE.findall(text.lower())]
        return (ids[:max_len] + [0] * max(0, max_len - len(ids)))


class DecisionTransformer(nn.Module):
    def __init__(self, vocab_size: int, classes: int, width: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width, padding_idx=0)
        self.position = nn.Embedding(32, width)
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(width, classes)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        encoded = self.encoder(self.embedding(ids) + self.position(pos), src_key_padding_mask=ids.eq(0))
        representation = encoded[:, 0]
        return self.head(representation), representation


@dataclass
class TrainedEncoder:
    vocab: Vocabulary
    model: DecisionTransformer
    policy_to_id: dict[str, int]

    @torch.no_grad()
    def decision_key(self, text: str) -> int:
        ids = torch.tensor([self.vocab.encode(text)])
        logits, _ = self.model(ids)
        return int(logits.argmax(dim=-1).item())

    @torch.no_grad()
    def embedding(self, text: str) -> torch.Tensor:
        ids = torch.tensor([self.vocab.encode(text)])
        _, rep = self.model(ids)
        return nn.functional.normalize(rep[0], dim=0)


def train_decision_encoder(seed: int = 7, epochs: int = 50) -> tuple[TrainedEncoder, float]:
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    examples = list(encoder_examples())
    vocab = Vocabulary([x.text for x in examples])
    policy_to_id = {p: i for i, p in enumerate(POLICIES)}
    x = torch.tensor([vocab.encode(item.text) for item in examples])
    y = torch.tensor([policy_to_id[item.policy] for item in examples])
    model = DecisionTransformer(len(vocab.stoi), len(POLICIES))
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    for _ in range(epochs):
        logits, _ = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        accuracy = float((model(x)[0].argmax(-1) == y).float().mean())
    return TrainedEncoder(vocab, model, policy_to_id), accuracy
