# =============================================================================
#  layer_isomorphism_torch.py
#  All 15 isomorphism layers (L0–L14) as torch.nn.Module subclasses.
#
#  L0–L13 are the original token / zone / blend / cursor stack.
#  L14 (NEW) is a previous-state-dependent index dimension with monotonic
#  write-once semantics keyed by trigram prefix.  Its modulus indication
#  is offset by the count of "missing states" (observed-but-uncommitted
#  trigram keys), so unresolved context literally pushes the cursor
#  forward through the modulus.
#
#  Design principles
#  -----------------
#  • Every layer is a self-contained nn.Module with a custom __init__ that
#    registers its hyper-parameters as nn.Parameter (learnable) or as named
#    buffers (non-gradient scalars that still move with .to(device)).
#  • forward() accepts and returns plain Python / numpy inputs where the
#    upstream code expects them, but all heavy maths runs on torch tensors.
#  • IsomorphismPipeline wires L0..L13 together and mirrors the original
#    API.  LockedIsomorphismPipeline subclasses it and inserts L14 into
#    the final sampling stage.
#  • No external dependencies beyond torch, numpy, math, collections.
# =============================================================================

from __future__ import annotations

import math
from collections import Counter, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from datasets import load_dataset, Dataset, DatasetDict
import re
from dataclasses import dataclass, asdict

@dataclass
class HFSquadRecord:
    split: str
    original_index: int
    id: str
    title: str
    context: str
    question: str
    answer_text: str
    answer_start: int | None
    tokens: List[str]
    first_token: str
    last_token: str
    kept: bool
    drop_reason: str


class HFSquadSentenceDatasetPreprocessor:
    """
    Hugging Face SQuAD-backed analogue of SentenceDatasetPreprocessor.

    Each dataset entity becomes one token sequence. We then enforce the same
    quota-balanced boundary invariant already used by SentenceDatasetPreprocessor:
      - a first token may appear at most boundaryquota times
      - a last token may appear at most boundaryquota times
      - acceptance is greedy in dataset order
      - boundaryquota=1 gives strict globally-unique beginnings and endings

    Public attributes intentionally mirror SentenceDatasetPreprocessor enough
    for SentenceAwareGenerator / buildsentencepipeline-style code to use them.
    """

    def __init__(
        self,
        dataset_name: str = "squad",
        config_name: str | None = None,
        split_names: Sequence[str] = ("train", "validation"),
        lowercase: bool = True,
        minsentencelen: int = 3,
        uniquemiddlepool: bool = True,
        strict: bool = True,
        boundaryquota: int = 1,
        include_question: bool = True,
        include_context: bool = True,
        include_answer: bool = True,
        qca_mode: str = "question_context_answer",
        sep_qc: str | None = None,
        sep_ca: str | None = None,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split_names = tuple(split_names)
        self.lowercase = bool(lowercase)
        self.minsentencelen = max(2, int(minsentencelen))
        self.uniquemiddlepool = bool(uniquemiddlepool)
        self.strict = bool(strict)
        self.boundaryquota = max(1, int(boundaryquota))

        self.include_question = bool(include_question)
        self.include_context = bool(include_context)
        self.include_answer = bool(include_answer)
        self.qca_mode = str(qca_mode)
        self.sep_qc = sep_qc
        self.sep_ca = sep_ca

        self.sentences: List[List[str]] = []
        self.beginnings: List[str] = []
        self.endings: List[str] = []
        self.middlepool: List[str] = []
        self.tokens: List[str] = []

        self.records: List[HFSquadRecord] = []
        self.keptrecords: List[HFSquadRecord] = []
        self.droppedrecords: List[HFSquadRecord] = []

        self.dropped = 0
        self.skipped = 0
        self.begincounts: Counter = Counter()
        self.endcounts: Counter = Counter()

        self.beginningsset: Set[str] = set()
        self.endingsset: Set[str] = set()
        self.middleset: Set[str] = set()

        self.process()

    @staticmethod
    def _word_tokenize(text: str, lowercase: bool = True) -> List[str]:
        if not text:
            return []
        if lowercase:
            text = text.lower()
        return text.split()

    def _load_dataset(self) -> DatasetDict:
        if self.config_name is None:
            return load_dataset(self.dataset_name)
        return load_dataset(self.dataset_name, self.config_name)

    @staticmethod
    def _pick_answer(example: Dict) -> Tuple[str, int | None]:
        ans = example.get("answers", {}) or {}
        texts = ans.get("text", []) if isinstance(ans, dict) else []
        starts = ans.get("answer_start", []) if isinstance(ans, dict) else []
        text0 = texts[0] if texts else ""
        start0 = starts[0] if starts else None
        return text0, start0

    def _build_entity_tokens(self, example: Dict) -> List[str]:
        q = self._word_tokenize(example.get("question", ""), self.lowercase)
        c = self._word_tokenize(example.get("context", ""), self.lowercase)
        a_text, _ = self._pick_answer(example)
        a = self._word_tokenize(a_text, self.lowercase)

        mode = self.qca_mode.lower()

        if mode == "question_only":
            parts = [q] if self.include_question else []
        elif mode == "question_answer":
            parts = []
            if self.include_question:
                parts.append(q)
            if self.include_answer:
                if self.sep_qc:
                    parts.append([self.sep_qc])
                parts.append(a)
        elif mode == "question_context":
            parts = []
            if self.include_question:
                parts.append(q)
            if self.include_context:
                if self.sep_qc:
                    parts.append([self.sep_qc])
                parts.append(c)
        else:
            parts = []
            if self.include_question:
                parts.append(q)
            if self.include_context:
                if parts and self.sep_qc:
                    parts.append([self.sep_qc])
                parts.append(c)
            if self.include_answer:
                if parts and self.sep_ca:
                    parts.append([self.sep_ca])
                parts.append(a)

        out: List[str] = []
        for block in parts:
            out.extend(block)
        return out

    def _iter_entities(self):
        ds = self._load_dataset()
        for split in self.split_names:
            if split not in ds:
                continue
            split_ds: Dataset = ds[split]
            for idx, ex in enumerate(split_ds):
                yield split, idx, ex

    def _record_from_example(self, split: str, idx: int, ex: Dict) -> HFSquadRecord:
        answer_text, answer_start = self._pick_answer(ex)
        toks = self._build_entity_tokens(ex)

        if len(toks) < self.minsentencelen:
            self.skipped += 1
            return HFSquadRecord(
                split=split,
                original_index=idx,
                id=str(ex.get("id", f"{split}-{idx}")),
                title=str(ex.get("title", "")),
                context=str(ex.get("context", "")),
                question=str(ex.get("question", "")),
                answer_text=answer_text,
                answer_start=answer_start,
                tokens=toks,
                first_token=toks[0] if toks else "",
                last_token=toks[-1] if toks else "",
                kept=False,
                drop_reason=f"too_short_lt_{self.minsentencelen}",
            )

        first = toks[0]
        last = toks[-1]

        if self.strict:
            if self.begincounts[first] >= self.boundaryquota:
                self.dropped += 1
                return HFSquadRecord(
                    split=split,
                    original_index=idx,
                    id=str(ex.get("id", f"{split}-{idx}")),
                    title=str(ex.get("title", "")),
                    context=str(ex.get("context", "")),
                    question=str(ex.get("question", "")),
                    answer_text=answer_text,
                    answer_start=answer_start,
                    tokens=toks,
                    first_token=first,
                    last_token=last,
                    kept=False,
                    drop_reason=f"begin_quota_full:{first}",
                )
            if self.endcounts[last] >= self.boundaryquota:
                self.dropped += 1
                return HFSquadRecord(
                    split=split,
                    original_index=idx,
                    id=str(ex.get("id", f"{split}-{idx}")),
                    title=str(ex.get("title", "")),
                    context=str(ex.get("context", "")),
                    question=str(ex.get("question", "")),
                    answer_text=answer_text,
                    answer_start=answer_start,
                    tokens=toks,
                    first_token=first,
                    last_token=last,
                    kept=False,
                    drop_reason=f"end_quota_full:{last}",
                )

            self.begincounts[first] += 1
            self.endcounts[last] += 1

        return HFSquadRecord(
            split=split,
            original_index=idx,
            id=str(ex.get("id", f"{split}-{idx}")),
            title=str(ex.get("title", "")),
            context=str(ex.get("context", "")),
            question=str(ex.get("question", "")),
            answer_text=answer_text,
            answer_start=answer_start,
            tokens=toks,
            first_token=first,
            last_token=last,
            kept=True,
            drop_reason="",
        )

    def process(self) -> None:
        orderedpool: List[str] = []

        for split, idx, ex in self._iter_entities():
            rec = self._record_from_example(split, idx, ex)
            self.records.append(rec)

            if rec.kept:
                self.keptrecords.append(rec)
                s = rec.tokens
                self.sentences.append(s)
                self.beginnings.append(s[0])
                self.endings.append(s[-1])
                orderedpool.extend(s[1:-1])
                self.tokens.extend(s)
            else:
                self.droppedrecords.append(rec)

        if self.uniquemiddlepool:
            seen = set()
            self.middlepool = []
            for w in orderedpool:
                if w not in seen:
                    seen.add(w)
                    self.middlepool.append(w)
        else:
            self.middlepool = orderedpool

        self.beginningsset = set(self.beginnings)
        self.endingsset = set(self.endings)
        self.middleset = set(self.middlepool)

        if not self.sentences:
            raise ValueError(
                f"No SQuAD entities survived the quota-boundary invariant "
                f"(quota={self.boundaryquota}, dropped={self.dropped}, skipped={self.skipped})."
            )

    def tocorpus(self) -> str:
        return " ".join(self.tokens)

    def vocab(self) -> set:
        return set(self.tokens)

    def isbeginning(self, token: str) -> bool:
        return token in self.beginningsset

    def isnaturalending(self, token: str) -> bool:
        return token in self.endingsset

    def samplearbitrary(
        self,
        rngvalue: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        if not self.middlepool:
            return ""
        n = len(self.middlepool)
        if rngvalue is not None:
            i = int(rngvalue * n) % n
        else:
            rng = rng or random
            i = rng.randrange(n)
        return self.middlepool[i]

    def boundarybalancereport(self) -> str:
        def stats(c: Counter, label: str) -> str:
            if not c:
                return f"{label}: empty"
            vals = list(c.values())
            mn, mx = min(vals), max(vals)
            avg = sum(vals) / len(vals)
            perfectly = all(v == vals[0] for v in vals)
            return (
                f"{label}: {len(c)} words, "
                f"min={mn} max={mx} avg={avg:.2f} "
                f"{'perfectly balanced' if perfectly else 'imbalanced'}"
            )

        lines = [
            f"Boundary quota {self.boundaryquota}",
            stats(self.begincounts, "beginnings"),
            stats(self.endcounts, "endings"),
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        nsent = len(self.sentences)
        avg = sum(len(s) for s in self.sentences) / max(1, nsent)
        return "\n".join(
            [
                "HFSquadSentenceDatasetPreprocessor",
                f"dataset {self.dataset_name}",
                f"mode {self.qca_mode}",
                f"boundaryquota {self.boundaryquota}",
                f"entities kept {nsent}",
                f"dropped quota {self.dropped}",
                f"skipped too short {self.skipped}",
                f"total tokens {len(self.tokens)}",
                f"vocab size {len(self.vocab())}",
                f"unique beginnings {len(self.beginningsset)}",
                f"unique endings {len(self.endingsset)}",
                f"middle-pool size {len(self.middlepool)}",
                f"avg entity len {avg:.2f}",
                f"beginning example {self.beginnings[0] if self.beginnings else ''}",
                f"ending example {self.endings[0] if self.endings else ''}",
                f"middle example {self.middlepool[0] if self.middlepool else ''}",
                self.boundarybalancereport(),
            ]
        )

    def auditrows(self) -> List[Dict]:
        return [asdict(r) for r in self.records]

    def keptrows(self) -> List[Dict]:
        return [asdict(r) for r in self.keptrecords]

    def droppedrows(self) -> List[Dict]:
        return [asdict(r) for r in self.droppedrecords]


# ═══════════════════════════════════════════════════════════════════════════
#  SentenceAwareGenerator
# ═══════════════════════════════════════════════════════════════════════════

class SentenceAwareGenerator:
    """
    Wraps a pipeline so that whenever the live context's most-recent
    token is a NATURAL ENDING (one of the globally-unique sentence-final
    words), the next step:

        1. samples a word ARBITRARILY from the preprocessor's middle pool
        2. pushes it into the context as the new "beginning"
        3. the pipeline then PREDICTS UPON that word in the following step

    The arbitrary draw uses the same pi-stream-backed draw function the
    pipeline uses for L13, so the whole loop stays a single deterministic
    isomorphism over the input pi-stream.
    """

    def __init__(
        self,
        pipeline:    IsomorphismPipeline,
        preprocessor: Optional[SentenceDatasetPreprocessor] = None,
        *,
        emit_seed:   bool = True,
    ):
        if preprocessor is None:
            preprocessor = getattr(pipeline, "preprocessor", None)
        if preprocessor is None:
            raise ValueError(
                "SentenceAwareGenerator needs a preprocessor — either pass "
                "one in or use build_sentence_pipeline() which attaches it."
            )
        self.pipeline  = pipeline
        self.pre       = preprocessor
        self.emit_seed = bool(emit_seed)

    # ── helpers ──────────────────────────────────────────────────────

    def _seed_context(self, prompt: str) -> deque:
        tokens       = [w.lower() for w in prompt.split() if w.isalpha()]
        vocab_tokens = [w for w in tokens if w in self.pipeline.vocab]
        cw           = self.pipeline.context_window
        if len(vocab_tokens) >= cw:
            init = vocab_tokens[-cw:]
        else:
            init = [""] * (cw - len(vocab_tokens)) + vocab_tokens
        return deque(init, maxlen=cw)

    def _format(self, words: List[str], prompt: str, capitalise: bool) -> str:
        """
        Insert periods after natural endings, capitalise sentence starts.
        Beginnings and endings are themselves real corpus words, so we
        rely on the preprocessor's `is_natural_ending` to detect breaks.
        """
        prompt_words = prompt.strip().split() if prompt.strip() else []
        all_words    = prompt_words + words
        if not all_words:
            return ""

        out:      List[str] = []
        cap_next: bool      = True
        pre                 = self.pre
        for w in all_words:
            tok = w
            out.append(tok)
        return " ".join(out)

    # ── main API ─────────────────────────────────────────────────────

    def generate(
        self,
        prompt:            str,
        n_words:           int,
        *,
        stream:            Optional[List[int]] = None,
        digits_per_sample: int  = 3,
        seed:              Optional[int] = None,
    ) -> List[str]:
        """Return the raw list of emitted tokens (including arbitrary seeds)."""
        pipe = self.pipeline
        pre  = self.pre

        # FIX 2 + 3: reset all mutable per-run state before building
        # draw_fn so that cursor, history, step counter, and L14 lock
        # table all start from zero on every call.

        draw_fn = pipe._make_draw_fn(stream, digits_per_sample, seed)

        prompt_tokens = [w.lower() for w in prompt.split() if w.isalpha()]
        ctx           = self._seed_context(prompt)

        words:  List[str]        = []
        frames: List[LayerFrame] = []
        produced                 = 0

        # safety cap: don't loop forever if every prediction is an ending
        max_iters = n_words * 80
        iters     = 0

        while produced < n_words and iters < max_iters:
            iters += 1
            last  = ctx[-1] if (len(ctx) and ctx[-1]) else ""

          
            # Normal pipeline step
            frame = pipe.step(ctx, prompt_tokens, draw_fn())
            if frame is None:
                # dead context — arbitrary restart from the middle pool
                ctx.clear()
                ctx.extend([""] * pipe.context_window)
                seed_word = pre.sample_arbitrary()
                if not seed_word:
                    break
                ctx.append(seed_word)
                if self.emit_seed:
                    words.append(seed_word)
                    produced += 1
                continue

            frames.append(frame)
            ctx.append(frame.chosen)
            words.append(frame.chosen)
            produced += 1

        pipe.frames = frames
        return words

    def generate_text(
        self,
        prompt:            str,
        n_words:           int,
        *,
        stream:            Optional[List[int]] = None,
        digits_per_sample: int  = 3,
        seed:              Optional[int] = None,
        capitalise:        bool = True,
    ) -> str:
        words = self.generate(
            prompt            = prompt,
            n_words           = n_words,
            stream            = stream,
            digits_per_sample = digits_per_sample,
            seed              = seed,
        )
        return self._format(words, prompt, capitalise)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _to_tensor(probs, dtype=torch.float64) -> torch.Tensor:
    if isinstance(probs, torch.Tensor):
        return probs.to(dtype)
    if isinstance(probs, np.ndarray):
        return torch.from_numpy(probs.astype(np.float64)).to(dtype)
    return torch.tensor(probs, dtype=dtype)


def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Safe L1 normalisation."""
    s = t.sum()
    return t / s.clamp(min=1e-30)


def _char_trigrams(word: str):
    return {word[i:i + 3] for i in range(len(word) - 2)} if len(word) >= 3 else {word}


# ---------------------------------------------------------------------------
# L0 – Raw distribution with repetition penalty
# ---------------------------------------------------------------------------

class L0_RawDist(nn.Module):
    """
    Extracts the CPD posterior for a context, applies per-token repetition
    penalty, and returns a normalised probability distribution.
    """

    def __init__(self, rep_penalty: float = 1.13):
        super().__init__()
        self.rep_penalty = nn.Parameter(torch.tensor(rep_penalty, dtype=torch.float64))

    def forward(self, dist, history: Counter) -> Tuple[List[Tuple[str, float]], Dict]:
        pen = self.rep_penalty.clamp(min=1.0)

        raw: List[Tuple[str, float]] = []
        for s in dist.samples():
            if not s:
                continue
            p   = max(1e-12, float(dist.prob(s)))
            cnt = history[s]
            if cnt > 0:
                p /= pen.detach().item() ** cnt
            raw.append((s, p))

        if not raw:
            return raw, {}

        raw.sort(key=lambda x: x[1], reverse=True)

        probs_t = _to_tensor([p for _, p in raw])
        probs_t = _normalise(probs_t)

        words   = [w for w, _ in raw]
        pairs   = list(zip(words, probs_t.tolist()))

        layer = {
            "name":   "L0_RAW_DIST",
            "words":  words,
            "probs":  probs_t.detach().numpy(),
            "source": f"CPD posterior + rep_penalty={pen.detach().item():.4f}",
        }
        return pairs, layer


# ---------------------------------------------------------------------------
# L1 – Temperature scaling
# ---------------------------------------------------------------------------

class L1_TempScaled(nn.Module):
    """Applies temperature scaling: p_i ∝ p_i^(1/T)."""

    def __init__(self, temperature: float = 4.3):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float64))

    def forward(self, pairs: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Dict]:
        T = self.temperature.clamp(min=1e-3)

        probs_t  = _to_tensor([p for _, p in pairs])
        scaled_t = probs_t.pow(1.0 / T)
        scaled_t = _normalise(scaled_t)

        words = [w for w, _ in pairs]
        out   = list(zip(words, scaled_t.tolist()))

        layer = {
            "name":   "L1_TEMP_SCALED",
            "words":  words,
            "probs":  scaled_t.detach().numpy(),
            "source": f"temperature={T.detach().item():.4f}",
        }
        return out, layer


# ---------------------------------------------------------------------------
# L2 – Insight penalty
# ---------------------------------------------------------------------------

class L2_InsightPenalty(nn.Module):
    """Penalises tokens whose probability exceeds the mean."""

    def __init__(self, insight_penalty: float = 3.95):
        super().__init__()
        self.insight_penalty = nn.Parameter(
            torch.tensor(insight_penalty, dtype=torch.float64)
        )

    def forward(self, pairs: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Dict]:
        strength = self.insight_penalty.clamp(min=0.0)

        probs_t   = _to_tensor([p for _, p in pairs])
        mean_p    = probs_t.mean().clamp(min=1e-30)
        excess    = (probs_t - mean_p).clamp(min=0.0)
        penalised = probs_t / (1.0 + strength * excess / mean_p)
        penalised = _normalise(penalised.clamp(min=1e-12))

        words = [w for w, _ in pairs]
        out   = list(zip(words, penalised.tolist()))

        layer = {
            "name":   "L2_INSIGHT",
            "words":  words,
            "probs":  penalised.detach().numpy(),
            "source": f"insight_penalty={strength.detach().item():.4f}",
        }
        return out, layer


# ---------------------------------------------------------------------------
# L3 – Top-K / Top-P truncation
# ---------------------------------------------------------------------------

class L3_TopKTopP(nn.Module):
    """Truncates the candidate set to top-K then applies nucleus (top-P)."""

    def __init__(self, top_k: int = 100, top_p: float = 1.0):
        super().__init__()
        self.register_buffer("top_k_buf", torch.tensor(top_k, dtype=torch.int64))
        self.top_p = nn.Parameter(torch.tensor(top_p, dtype=torch.float64))

    def forward(self, pairs: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], Dict]:
        k    = int(self.top_k_buf.item())
        p_th = float(self.top_p.clamp(1e-3, 1.0).item())

        truncated = pairs[:k]
        kept, cumulative = [], 0.0
        for w, p in truncated:
            kept.append((w, p))
            cumulative += p
            if cumulative >= p_th:
                break

        probs_t = _to_tensor([p for _, p in kept])
        probs_t = _normalise(probs_t)

        words = [w for w, _ in kept]
        out   = list(zip(words, probs_t.tolist()))

        layer = {
            "name":   "L3_TOPK_TOPP",
            "words":  words,
            "probs":  probs_t.detach().numpy(),
            "source": f"top_k={k} top_p={p_th:.4f}",
        }
        return out, layer


# ---------------------------------------------------------------------------
# Shared Gaussian-gradient zone layer base (L4–L9, L13)
# ---------------------------------------------------------------------------

class _ZoneGradientBase(nn.Module):
    """
    Base class for zone layers L4–L9.  Each applies a Gaussian gradient
    over the ranked candidate list, centring the peak at the mean rank
    of tokens in the active zone.
    """

    def __init__(self, name: str, sigma: float, floor: float, source_hint: str = ""):
        super().__init__()
        self._layer_name  = name
        self._source_hint = source_hint
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def _gradient(self, zone_set: set, candidates: List[Tuple[str, float]]) -> torch.Tensor:
        n = len(candidates)
        if n == 0:
            return torch.zeros(0, dtype=torch.float64)

        sigma = self.sigma.clamp(min=1e-6)
        floor = self.floor.clamp(min=0.0, max=1.0 - 1e-6)

        indices  = torch.arange(n, dtype=torch.float64)
        norm_idx = indices / max(1, n - 1)

        zone_ranks = [
            i / max(1, n - 1)
            for i, (w, _) in enumerate(candidates)
            if w in zone_set
        ]
        centre = float(torch.tensor(zone_ranks).mean()) if zone_ranks else 0.0

        gauss   = torch.exp(-0.5 * ((norm_idx - centre) / sigma) ** 2)
        weights = floor + (1.0 - floor) * gauss
        return _normalise(weights)

    def _make_layer(self, weights: torch.Tensor, candidates, source_extra: str = "") -> Dict:
        return {
            "name":   self._layer_name,
            "words":  [w for w, _ in candidates],
            "probs":  weights.detach().numpy(),
            "source": f"{self._source_hint} {source_extra}".strip(),
        }


# ---------------------------------------------------------------------------
# L4 – Frequency zone gradient
# ---------------------------------------------------------------------------

class L4_ZoneFreq(_ZoneGradientBase):
    def __init__(self, sigma: float = 0.50, floor: float = 0.05,
                 freq_high_thresh: int = 10, freq_mid_thresh: int = 3):
        super().__init__("L4_ZONE_FREQ", sigma, floor, "freq_zone")
        self.register_buffer("freq_high_thresh",
                             torch.tensor(freq_high_thresh, dtype=torch.int64))
        self.register_buffer("freq_mid_thresh",
                             torch.tensor(freq_mid_thresh, dtype=torch.int64))

    def forward(self, candidates, prompt_words, freq_zones, token_freq):
        mi_th = int(self.freq_mid_thresh.item())

        high_set = set(freq_zones.get("high", []))
        if any(w in high_set for w in prompt_words):
            key = "high"
        elif all(token_freq.get(w, 0) < mi_th for w in prompt_words):
            key = "low"
        else:
            key = "mid"

        zone_set = set(freq_zones.get(key, []))
        weights  = self._gradient(zone_set, candidates)
        return self._make_layer(weights, candidates, f"key={key}")


# ---------------------------------------------------------------------------
# L5 – Alpha-zone gradient
# ---------------------------------------------------------------------------

class L5_ZoneAlpha(_ZoneGradientBase):
    def __init__(self, sigma: float = 0.40, floor: float = 0.05):
        super().__init__("L5_ZONE_ALPHA", sigma, floor, "alpha_zone")

    def forward(self, candidates, prompt_words, alpha_zones):
        alpha_words: set = set()
        for w in prompt_words:
            if w and w[0].isalpha():
                alpha_words.update(alpha_zones.get(w[0], []))
        weights = self._gradient(alpha_words, candidates)
        keys    = [w[0] for w in prompt_words if w]
        return self._make_layer(weights, candidates, f"keys={keys}")


# ---------------------------------------------------------------------------
# L6 – Bigram n-gram zone gradient
# ---------------------------------------------------------------------------

class L6_ZoneBigram(_ZoneGradientBase):
    def __init__(self, sigma: float = 0.25, floor: float = 0.04):
        super().__init__("L6_ZONE_BIGRAM", sigma, floor, "ngram bigram context")

    def forward(self, candidates, prompt_words, ngram_zones):
        bigram_words: set = set()
        for i in range(len(prompt_words) - 1):
            bigram_words.update(
                ngram_zones.get((prompt_words[i], prompt_words[i + 1]), [])
            )
        weights = self._gradient(bigram_words, candidates)
        return self._make_layer(weights, candidates)


# ---------------------------------------------------------------------------
# L7 – Live trigram context zone gradient
# ---------------------------------------------------------------------------

class L7_ZoneTrigram(_ZoneGradientBase):
    def __init__(self, sigma: float = 0.20, floor: float = 0.03):
        super().__init__("L7_ZONE_TRIGRAM", sigma, floor, "live_ctx")

    def forward(self, candidates, context_deque, ngram_zones):
        ctx_list = list(context_deque)
        if len(ctx_list) >= 2:
            key      = tuple(ctx_list[-2:])
            zone_set = set(ngram_zones.get(key, []))
            src      = f"live_ctx={key}"
        elif len(ctx_list) == 1:
            key      = (ctx_list[-1],)
            zone_set = set(ngram_zones.get(key, []))
            src      = f"live_ctx=({ctx_list[-1]},)"
        else:
            zone_set = set()
            src      = "no live context"

        weights = self._gradient(zone_set, candidates)
        return self._make_layer(weights, candidates, src)


# ---------------------------------------------------------------------------
# L8 – Character-trigram neighbour gradient
# ---------------------------------------------------------------------------

class L8_ZoneCharTrig(_ZoneGradientBase):
    def __init__(self, sigma: float = 0.35, floor: float = 0.04):
        super().__init__("L8_ZONE_CHAR_TRIG", sigma, floor, "char-trigram neighbours")

    def forward(self, candidates, prompt_words, char_trig_idx):
        prompt_tgs: set = set()
        for w in prompt_words:
            prompt_tgs |= _char_trigrams(w)

        char_neighbours: set = set()
        for tg in prompt_tgs:
            char_neighbours |= char_trig_idx.get(tg, set())

        weights = self._gradient(char_neighbours, candidates)
        return self._make_layer(weights, candidates)


# ---------------------------------------------------------------------------
# L9 – Latent BOS quartile gradient
# ---------------------------------------------------------------------------

class L9_ZoneLatent(_ZoneGradientBase):
    def __init__(self, sigma: float = 0.30, floor: float = 0.04):
        super().__init__("L9_ZONE_LATENT", sigma, floor, "latent_bos_quartile")

    def forward(self, candidates, prompt_words, latent_sorted_keys, latent_bos_data):
        n_keys = len(latent_sorted_keys)
        q_key  = "q0"
        for ctx in latent_sorted_keys:
            if any(w in ctx for w in prompt_words):
                rank  = latent_sorted_keys.index(ctx)
                q_key = f"q{min(3, rank * 4 // max(1, n_keys))}"
                break

        zone_set = set(latent_bos_data.get(q_key, []))
        weights  = self._gradient(zone_set, candidates)
        return self._make_layer(weights, candidates, f"quartile={q_key}")


# ---------------------------------------------------------------------------
# L10 – History repetition column
# ---------------------------------------------------------------------------

class L10_History(nn.Module):
    """1/(smoothing+count) column. Default smoothing=1.0 reproduces the original."""

    def __init__(self, smoothing: float = 1.0):
        super().__init__()
        self.smoothing = nn.Parameter(torch.tensor(smoothing, dtype=torch.float64))

    def forward(self, candidates, history):
        smooth   = self.smoothing.clamp(min=0.0)
        counts   = torch.tensor([history[w] for w, _ in candidates], dtype=torch.float64)
        hist_vec = _normalise(1.0 / (smooth + counts))

        words = [w for w, _ in candidates]
        return {
            "name":   "L10_HISTORY",
            "words":  words,
            "probs":  hist_vec.detach().numpy(),
            "source": f"repetition history smoothing={smooth.detach().item():.4f}",
        }


# ---------------------------------------------------------------------------
# L11 – Tensor blend of zone layers (softmax row weights)
# ---------------------------------------------------------------------------

class L11_TensorBlend(nn.Module):
    N_ROWS = 7  # L4..L10

    def __init__(self, init_weights: Optional[List[float]] = None):
        super().__init__()
        if init_weights is None:
            init_weights = [1.0] * self.N_ROWS
        assert len(init_weights) == self.N_ROWS
        self.zone_weights = nn.Parameter(torch.tensor(init_weights, dtype=torch.float64))
        self.register_buffer("eps", torch.tensor(1e-12, dtype=torch.float64))

    def forward(self, zone_layers, candidates):
        n    = len(candidates)
        rows = []
        for layer in zone_layers:
            p = _to_tensor(layer["probs"])
            if p.shape[0] != n:
                p = F.pad(p, (0, n - p.shape[0]))[:n]
            rows.append(p.unsqueeze(0))

        zone_matrix = torch.cat(rows, dim=0)
        row_weights = F.softmax(self.zone_weights, dim=0)

        blended = (zone_matrix * row_weights.unsqueeze(1)).sum(dim=0)
        blended = _normalise(blended.clamp(min=float(self.eps)))

        words = [w for w, _ in candidates]
        return {
            "name":   "L11_TENSOR_BLEND",
            "words":  words,
            "probs":  blended.detach().numpy(),
            "source": (
                f"row-weighted blend of L4..L10 ({len(zone_layers)} rows) "
                f"softmax_weights={row_weights.tolist()}"
            ),
        }


# ---------------------------------------------------------------------------
# L12 – Final distribution (geometric mean of L3 and L11)
# ---------------------------------------------------------------------------

class L12_Final(nn.Module):
    def __init__(self, blend_alpha: float = 0.5):
        super().__init__()
        self.blend_alpha = nn.Parameter(torch.tensor(blend_alpha, dtype=torch.float64))

    def forward(self, L3_pairs, L11):
        alpha = self.blend_alpha.clamp(min=1e-6, max=1.0 - 1e-6)
        beta  = 1.0 - alpha

        p3  = _to_tensor([p for _, p in L3_pairs]).clamp(min=1e-24)
        p11 = _to_tensor(L11["probs"]).clamp(min=1e-24)

        blended = p3.pow(alpha) * p11.pow(beta)
        blended = _normalise(blended)

        sorted_idx = torch.argsort(blended, descending=True)
        words_arr  = [L3_pairs[i][0] for i in sorted_idx.tolist()]
        probs_arr  = blended[sorted_idx]

        out = list(zip(words_arr, probs_arr.tolist()))
        layer = {
            "name":   "L12_FINAL",
            "words":  words_arr,
            "probs":  probs_arr.detach().numpy(),
            "source": f"geo_mean(L3^{alpha.detach().item():.3f}, L11^{beta.detach().item():.3f})",
        }
        return out, layer


# ---------------------------------------------------------------------------
# L13 – Contextual requestor position (Gaussian over pi-cursor)
# ---------------------------------------------------------------------------

class L13_CtxReqPos(nn.Module):
    def __init__(self, sigma: float = 0.30, floor: float = 0.04):
        super().__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float64))
        self.floor = nn.Parameter(torch.tensor(floor, dtype=torch.float64))

    def forward(self, candidates, draw_pos, stream_len):
        n        = len(candidates)
        sigma    = self.sigma.clamp(min=1e-6)
        floor    = self.floor.clamp(min=0.0, max=1.0 - 1e-6)
        norm_pos = (draw_pos % max(1, stream_len)) / max(1, stream_len - 1)

        indices = torch.arange(n, dtype=torch.float64) / max(1, n - 1)
        gauss   = torch.exp(-0.5 * ((indices - norm_pos) / sigma) ** 2)
        weights = _normalise(floor + (1.0 - floor) * gauss)

        words = [w for w, _ in candidates]
        return {
            "name":   "L13_CTX_REQ_POS",
            "words":  words,
            "probs":  weights.detach().numpy(),
            "source": (
                f"ctx_req_pos={draw_pos} norm={norm_pos:.4f} stream_len={stream_len}"
            ),
        }


# ---------------------------------------------------------------------------
# L14 – Locked state index (NEW)
# ---------------------------------------------------------------------------

class L14_LockedStateIndex(nn.Module):
    """
    Previous-state-dependent index dimension with monotonic write-once
    semantics — the "forbid altering" rule.

    Keyed by the trigram prefix (last two non-empty tokens of the live
    context, matching L7's key space).  Maintains:

        _locked   : Dict[key -> first committed token]
        _observed : Set[keys seen but not yet committed]

    Forward branches
    ----------------
    LOCKED   — key already in _locked.  Returns a near one-hot
               distribution on the locked token (weighted by
               `lock_strength`), with the floor used as the residual
               mass on everything else.  Higher transient indexes
               therefore CANNOT alter the previously locked state.

    UNLOCKED — key not yet locked.  Records it in _observed and emits
               a Gaussian gradient.  The Gaussian centre is the live
               cursor position offset by the number of "missing states":

                   offset_pos = (draw_pos + n_missing) mod stream_len

               so unresolved context literally pushes the cursor
               forward through the modulus.

    Custom init
    -----------
    sigma         : Gaussian width (clamped > 0)
    floor         : minimum weight  (clamped to [0, 1))
    lock_strength : hardness of the lock peak (clamped to [0, 1])
                    1.0 -> hard lock (~one-hot)
                    0.0 -> recovers a flat floor (no locking effect)
    """

    LAYER_NAME = "L14_LOCKED_STATE_INDEX"

    def __init__(
        self,
        sigma:         float = 0.25,
        floor:         float = 0.03,
        lock_strength: float = 1.0,
    ):
        super().__init__()
        self.sigma         = nn.Parameter(torch.tensor(sigma,         dtype=torch.float64))
        self.floor         = nn.Parameter(torch.tensor(floor,         dtype=torch.float64))
        self.lock_strength = nn.Parameter(torch.tensor(lock_strength, dtype=torch.float64))

        # Non-parameter state (not in state_dict; reset between runs).
        self._locked:   Dict[Tuple[str, ...], str] = {}
        self._observed: Set[Tuple[str, ...]]       = set()

    # ── state control ────────────────────────────────────────────────

    def reset_state(self) -> None:
        """Forget every lock and observation.  Called at run start."""
        self._locked.clear()
        self._observed.clear()

    def commit(self, key: Tuple[str, ...], token: str) -> bool:
        """
        Lock ``token`` under ``key`` iff key is not already locked.
        Returns True on a successful new lock, False otherwise.  This
        enforces the write-once / monotonic rule: higher-transient
        indexes cannot overwrite a previous commitment.
        """
        if not key or not token:
            return False
        if key in self._locked:
            return False
        self._locked[key] = token
        self._observed.discard(key)
        return True

    @property
    def n_locked(self) -> int:
        return len(self._locked)

    @property
    def n_missing(self) -> int:
        return len(self._observed)

    # ── trigram key extraction ───────────────────────────────────────

    @staticmethod
    def key_from_ctx(context_deque: deque) -> Tuple[str, ...]:
        """Trigram prefix: last two non-empty tokens of the context."""
        ctx_list = [w for w in context_deque if w]
        if len(ctx_list) >= 2:
            return tuple(ctx_list[-2:])
        if len(ctx_list) == 1:
            return (ctx_list[-1],)
        return ()

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        candidates:    List[Tuple[str, float]],
        context_deque: deque,
        draw_pos:      int,
        stream_len:    int,
    ) -> Dict:
        n     = len(candidates)
        words = [w for w, _ in candidates]

        if n == 0:
            return {
                "name":   self.LAYER_NAME,
                "words":  [],
                "probs":  np.zeros(0, dtype=np.float64),
                "source": "empty candidates",
                "key":    (),
                "locked": False,
            }

        sigma = self.sigma.clamp(min=1e-6)
        floor = self.floor.clamp(min=0.0, max=1.0 - 1e-6)
        lockw = self.lock_strength.clamp(min=0.0, max=1.0)

        key = self.key_from_ctx(context_deque)

        # ─── LOCKED branch ───────────────────────────────────────────
        if key and key in self._locked:
            locked_token = self._locked[key]
            f = float(floor)
            w = float(lockw)

            weights = torch.full((n,), f, dtype=torch.float64)
            if locked_token in words:
                idx          = words.index(locked_token)
                weights[idx] = f + w * (1.0 - f)
            # else: locked token isn't in the candidate set; we degrade
            # gracefully to a uniform floor — the parent's L12·L13
            # contribution then dominates the blend for this step.
            weights = _normalise(weights)

            source = (
                f"LOCKED key={key} -> '{locked_token}' "
                f"(lock_strength={w:.3f}, n_locked={self.n_locked})"
            )
            locked_flag = True

        # ─── UNLOCKED branch ─────────────────────────────────────────
        else:
            if key:
                self._observed.add(key)

            missing    = self.n_missing
            sl         = max(1, int(stream_len))
            offset_pos = (int(draw_pos) + missing) % sl
            norm_pos   = offset_pos / max(1, sl - 1)

            indices = torch.arange(n, dtype=torch.float64) / max(1, n - 1)
            gauss   = torch.exp(-0.5 * ((indices - norm_pos) / sigma) ** 2)
            weights = _normalise(floor + (1.0 - floor) * gauss)

            source = (
                f"UNLOCKED key={key or '∅'} "
                f"draw_pos={draw_pos} missing={missing} "
                f"offset={offset_pos} norm={norm_pos:.4f}"
            )
            locked_flag = False

        return {
            "name":   self.LAYER_NAME,
            "words":  words,
            "probs":  weights.detach().numpy(),
            "source": source,
            "key":    key,
            "locked": locked_flag,
        }


# ---------------------------------------------------------------------------
# LayerFrame (unchanged data class)
# ---------------------------------------------------------------------------

class LayerFrame:
    """Container for one generation step's full layer stack."""
    __slots__ = (
        "step", "layers", "chosen", "context_window",
        "zone_name", "draw_pos", "next_draw_pos",
    )

    def __init__(
        self,
        step:           int,
        layers:         List[Dict],
        chosen:         str         = "",
        context_window: Tuple       = (),
        zone_name:      str         = "",
        draw_pos:       int         = 0,
        next_draw_pos:  int         = 0,
    ):
        self.step           = step
        self.layers         = layers
        self.chosen         = chosen
        self.context_window = context_window
        self.zone_name      = zone_name
        self.draw_pos       = draw_pos
        self.next_draw_pos  = next_draw_pos

    def get(self, name: str) -> Optional[Dict]:
        for layer in self.layers:
            if layer["name"] == name:
                return layer
        return None

    def tensor(self) -> torch.Tensor:
        rows = [_to_tensor(l["probs"]) for l in self.layers]
        if not rows:
            return torch.zeros(0, dtype=torch.float64)
        max_len = max(r.shape[0] for r in rows)
        padded  = [F.pad(r, (0, max_len - r.shape[0])) for r in rows]
        return torch.stack(padded)


# ---------------------------------------------------------------------------
# IsomorphismPipeline  – drop-in replacement for IsomorphismGenerator
# ---------------------------------------------------------------------------

class IsomorphismPipeline(nn.Module):
    """
    Full 14-layer isomorphic probability pipeline (L0..L13) as a single
    nn.Module.  See LockedIsomorphismPipeline below for the 15-layer
    variant that adds L14.
    """

    SAVE_FORMAT_VERSION = 2

    LAYER_NAMES = [
        "L0_RAW_DIST",      "L1_TEMP_SCALED",   "L2_INSIGHT",
        "L3_TOPK_TOPP",     "L4_ZONE_FREQ",     "L5_ZONE_ALPHA",
        "L6_ZONE_BIGRAM",   "L7_ZONE_TRIGRAM",  "L8_ZONE_CHAR_TRIG",
        "L9_ZONE_LATENT",   "L10_HISTORY",      "L11_TENSOR_BLEND",
        "L12_FINAL",        "L13_CTX_REQ_POS",
    ]

    def __init__(
        self,
        cpd,
        context_index,
        vocab,
        ngram_n:         int   = 2,
        temperature:     float = 4.3,
        top_k:           int   = 100,
        top_p:           float = 1.0,
        rep_penalty:     float = 1.13,
        insight_penalty: float = 3.95,
        history:         Optional[Counter] = None,
        # ── per-layer custom init overrides ──────────────────────────
        l4_sigma: float = 0.50,   l4_floor: float = 0.05,
        l5_sigma: float = 0.40,   l5_floor: float = 0.05,
        l6_sigma: float = 0.25,   l6_floor: float = 0.04,
        l7_sigma: float = 0.20,   l7_floor: float = 0.03,
        l8_sigma: float = 0.35,   l8_floor: float = 0.04,
        l9_sigma: float = 0.30,   l9_floor: float = 0.04,
        l10_smoothing:    float = 1.0,
        l11_init_weights: Optional[List[float]] = None,
        l12_blend_alpha:  float = 0.5,
        l13_sigma: float = 0.30,  l13_floor: float = 0.04,
    ):
        super().__init__()

        self.cpd            = cpd
        self.ctx_idx        = context_index
        self.vocab          = set(vocab)
        self.ngram_n        = max(2, int(ngram_n))
        self.context_window = self.ngram_n - 1
        self.history        = Counter(history) if history else Counter()
        self._step          = 0
        self._pos           = 0
        self._stream: List[int] = []
        self._char_trig_index: Dict[str, set] = (
            getattr(context_index, "_trig_index", {}) if context_index else {}
        )

        self._init_hparams: Dict = dict(
            ngram_n         = self.ngram_n,
            temperature     = float(temperature),
            top_k           = int(top_k),
            top_p           = float(top_p),
            rep_penalty     = float(rep_penalty),
            insight_penalty = float(insight_penalty),
            l4_sigma=float(l4_sigma), l4_floor=float(l4_floor),
            l5_sigma=float(l5_sigma), l5_floor=float(l5_floor),
            l6_sigma=float(l6_sigma), l6_floor=float(l6_floor),
            l7_sigma=float(l7_sigma), l7_floor=float(l7_floor),
            l8_sigma=float(l8_sigma), l8_floor=float(l8_floor),
            l9_sigma=float(l9_sigma), l9_floor=float(l9_floor),
            l10_smoothing   = float(l10_smoothing),
            l11_init_weights= list(l11_init_weights) if l11_init_weights is not None else None,
            l12_blend_alpha = float(l12_blend_alpha),
            l13_sigma=float(l13_sigma), l13_floor=float(l13_floor),
        )

        self.l0  = L0_RawDist(rep_penalty=rep_penalty)
        self.l1  = L1_TempScaled(temperature=temperature)
        self.l2  = L2_InsightPenalty(insight_penalty=insight_penalty)
        self.l3  = L3_TopKTopP(top_k=top_k, top_p=top_p)
        self.l4  = L4_ZoneFreq(sigma=l4_sigma, floor=l4_floor)
        self.l5  = L5_ZoneAlpha(sigma=l5_sigma, floor=l5_floor)
        self.l6  = L6_ZoneBigram(sigma=l6_sigma, floor=l6_floor)
        self.l7  = L7_ZoneTrigram(sigma=l7_sigma, floor=l7_floor)
        self.l8  = L8_ZoneCharTrig(sigma=l8_sigma, floor=l8_floor)
        self.l9  = L9_ZoneLatent(sigma=l9_sigma, floor=l9_floor)
        self.l10 = L10_History(smoothing=l10_smoothing)
        self.l11 = L11_TensorBlend(init_weights=l11_init_weights)
        self.l12 = L12_Final(blend_alpha=l12_blend_alpha)
        self.l13 = L13_CtxReqPos(sigma=l13_sigma, floor=l13_floor)

    # ── internal helpers ──────────────────────────────────────────────

    def _dist_for_ctx(self, ctx_tuple):
        for cut in range(len(ctx_tuple), 0, -1):
            trial = ("",) * (self.context_window - cut) + ctx_tuple[-cut:]
            try:
                d = self.cpd[trial]
                if list(d.samples()):
                    return d
            except Exception:
                continue
        try:
            d = self.cpd[("",) * self.context_window]
            if list(d.samples()):
                return d
        except Exception:
            pass
        return None

    # ── public API ────────────────────────────────────────────────────

    def seed_stream(self, stream: list):
        """Attach the raw pi-stream so L13 can read its length & cursor."""
        self._stream = list(stream)
        self._pos    = 0

    def step(
        self,
        context_deque: deque,
        prompt_words:  List[str],
        draw:          float,
        zone_name:     str = "",
    ) -> Optional[LayerFrame]:
        dist = self._dist_for_ctx(tuple(context_deque))
        if dist is None:
            return None

        L0_pairs, L0 = self.l0(dist, self.history)
        if not L0_pairs:
            return None

        L1_pairs, L1 = self.l1(L0_pairs)
        L2_pairs, L2 = self.l2(L1_pairs)
        L3_pairs, L3 = self.l3(L2_pairs)
        if not L3_pairs:
            return None

        ci = self.ctx_idx

        if ci is None:
            flat = _normalise(torch.ones(len(L3_pairs), dtype=torch.float64))
            flat_np = flat.detach().numpy()
            words = [w for w, _ in L3_pairs]
            zone_layers = [
                {"name": n, "words": words, "probs": flat_np.copy(),
                 "source": "no context_index"}
                for n in ["L4_ZONE_FREQ", "L5_ZONE_ALPHA", "L6_ZONE_BIGRAM",
                           "L7_ZONE_TRIGRAM", "L8_ZONE_CHAR_TRIG", "L9_ZONE_LATENT"]
            ]
        else:
            L4 = self.l4(L3_pairs, prompt_words, ci.freq_zones, ci.token_freq)
            L5 = self.l5(L3_pairs, prompt_words, ci.alpha_zones)
            L6 = self.l6(L3_pairs, prompt_words, ci.ngram_zones)
            L7 = self.l7(L3_pairs, context_deque, ci.ngram_zones)
            L8 = self.l8(L3_pairs, prompt_words, self._char_trig_index)
            L9 = self.l9(
                L3_pairs, prompt_words,
                ci.latent_sorted_keys, ci.latent_bos_data,
            )
            zone_layers = [L4, L5, L6, L7, L8, L9]

        L10 = self.l10(L3_pairs, self.history)
        L11 = self.l11(zone_layers + [L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)

        draw_pos   = self._pos
        stream_len = max(1, len(self._stream))
        L13 = self.l13(L3_pairs, draw_pos, stream_len)

        # Geometric blend of L12 and L13
        l12_map = dict(L12_pairs)
        l13_map = dict(zip(L13["words"], L13["probs"].tolist()))
        all_words = list(l12_map.keys())
        floor_val = 1e-12

        blended = [
            (w, math.sqrt(
                max(floor_val, l12_map.get(w, floor_val)) *
                max(floor_val, l13_map.get(w, floor_val))
            ))
            for w in all_words
        ]
        bt      = sum(p for _, p in blended)
        blended = [(w, p / bt) for w, p in blended] if bt > 0 else blended

        unseen = [(w, p) for w, p in blended if self.history[w] == 0]
        pool   = unseen if unseen else blended
        t      = sum(p for _, p in pool)
        pool   = [(w, p / t) for w, p in pool] if t > 0 else pool

        chosen, cumulative = pool[-1][0], 0.0
        for w, p in pool:
            cumulative += p
            if draw < cumulative:
                chosen = w
                break

        self.history[chosen] += 1

        next_draw_pos = (draw_pos + (draw_pos % max(1, stream_len))) % stream_len
        self._pos     = next_draw_pos
        self._step   += 1

        return LayerFrame(
            step           = self._step - 1,
            layers         = [L0, L1, L2, L3] + zone_layers + [L10, L11, L12, L13],
            chosen         = chosen,
            context_window = tuple(context_deque),
            zone_name      = zone_name,
            draw_pos       = draw_pos,
            next_draw_pos  = next_draw_pos,
        )

    def generate(self, prompt: str, n_words: int, draw_fn, zone_fn=None):
        tokens       = [w.lower() for w in prompt.split() if w.isalpha()]
        vocab_tokens = [w for w in tokens if w in self.vocab]

        if len(vocab_tokens) >= self.context_window:
            init = vocab_tokens[-self.context_window:]
        else:
            init = [""] * (self.context_window - len(vocab_tokens)) + vocab_tokens

        ctx = deque(init, maxlen=self.context_window)

        for _ in range(n_words):
            zone_name = zone_fn(draw_fn()) if zone_fn is not None else ""
            draw      = draw_fn()
            frame     = self.step(ctx, tokens, draw, zone_name=zone_name)
            if frame is None:
                ctx.clear()
                ctx.extend([""] * self.context_window)
                continue
            ctx.append(frame.chosen)
            yield frame

    # ── PyTorch convenience ───────────────────────────────────────────

    def param_summary(self) -> str:
        lines = [f"{'Parameter':<45} {'Value':>14}"]
        lines.append("-" * 61)
        for name, param in self.named_parameters():
            v = param.data
            if v.numel() == 1:
                lines.append(f"  {name:<43} {v.item():>14.6f}")
            else:
                lines.append(
                    f"  {name:<43} shape={list(v.shape)}  "
                    f"mean={v.mean().item():.4f}"
                )
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────

    def _extract_cfd_counts(self) -> Dict[Tuple[str, ...], Dict[str, int]]:
        counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
        for ctx in self.cpd.conditions():
            dist = self.cpd[ctx]
            fd   = getattr(dist, "freqdist", lambda: None)()
            if fd is None:
                continue
            ctx_counts = {str(w): int(c) for w, c in fd.items()}
            if ctx_counts:
                counts[tuple(ctx)] = ctx_counts
        return counts

    def _build_save_payload(
        self,
        kind:           str,
        corpus_text:    Optional[str] = None,
        lidstone_gamma: Optional[float] = None,
        tokens:         Optional[List[str]] = None,
        include_history: bool = True,
    ) -> Dict:
        payload: Dict = {
            "format_version": self.SAVE_FORMAT_VERSION,
            "kind":           kind,
            "state_dict":     self.state_dict(),
            "hparams":        dict(self._init_hparams),
            "class_name":     type(self).__name__,
        }
        if include_history:
            payload["history"] = dict(self.history)

        if kind == "full":
            if corpus_text is None:
                raise ValueError("full save requires corpus_text")
            payload["corpus_text"]    = corpus_text
            payload["lidstone_gamma"] = float(lidstone_gamma) if lidstone_gamma is not None else 0.1
            payload["tokens"]         = list(tokens) if tokens else []
            payload["vocab"]          = sorted(self.vocab)

        elif kind == "midstate":
            payload["lidstone_gamma"] = (
                float(lidstone_gamma) if lidstone_gamma is not None else 0.1
            )
            payload["vocab"]          = sorted(self.vocab)
            payload["tokens"]         = list(tokens) if tokens else []
            payload["cfd_counts"]     = self._extract_cfd_counts()
        return payload

    def save(
        self,
        path:           str,
        *,
        kind:           str             = "full",
        corpus_text:    Optional[str]   = None,
        lidstone_gamma: Optional[float] = None,
        tokens:         Optional[List[str]] = None,
        include_history: bool           = True,
    ) -> str:
        if kind not in ("full", "weights", "midstate"):
            raise ValueError(
                f"kind must be 'full', 'midstate' or 'weights', got {kind!r}"
            )

        payload = self._build_save_payload(
            kind            = kind,
            corpus_text     = corpus_text,
            lidstone_gamma  = lidstone_gamma,
            tokens          = tokens,
            include_history = include_history,
        )
        torch.save(payload, path)
        return path

    @classmethod
    def _construct_from_payload(cls, payload, cpd, ctx_idx, vocab, ngram_n):
        """Build a pipeline of the appropriate class from a save payload."""
        hparams = dict(payload.get("hparams", {}))
        init_kwargs = dict(hparams)
        init_kwargs.pop("ngram_n", None)

        # If the saved class is LockedIsomorphismPipeline, dispatch to it.
        saved_class = payload.get("class_name", cls.__name__)
        target_cls = cls
        if saved_class == "LockedIsomorphismPipeline" and cls is IsomorphismPipeline:
            target_cls = LockedIsomorphismPipeline

        return target_cls(
            cpd           = cpd,
            context_index = ctx_idx,
            vocab         = vocab,
            ngram_n       = ngram_n,
            **init_kwargs,
        )

    @classmethod
    def load(
        cls,
        path:           str,
        *,
        rebuild_context_index: bool = True,
    ) -> "IsomorphismPipeline":
        payload = torch.load(path, map_location="cpu", weights_only=False)

        fmt = payload.get("format_version", 0)
        if fmt > cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is newer than supported "
                f"({cls.SAVE_FORMAT_VERSION}). Upgrade the code."
            )

        if payload.get("kind") != "full":
            raise ValueError(
                "load() requires a 'full' snapshot.  For weights-only files "
                "use IsomorphismPipeline.load_into(existing_pipeline, path)."
            )

        corpus_text = payload["corpus_text"]
        gamma       = float(payload.get("lidstone_gamma", 0.1))
        ngram_n     = int(payload.get("hparams", {}).get("ngram_n", 2))

        cpd, vocab, tokens = build_real_cpd(corpus_text, ngram_n, gamma)
        ctx_idx = build_real_context_index(vocab, cpd, tokens) if rebuild_context_index else None

        pipeline = cls._construct_from_payload(payload, cpd, ctx_idx, vocab, ngram_n)

        missing, unexpected = pipeline.load_state_dict(payload["state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [load] missing keys: {list(missing)}")
            print(f"  [load] unexpected keys: {list(unexpected)}")

        if "history" in payload and isinstance(payload["history"], dict):
            pipeline.history = Counter(payload["history"])

        return pipeline

    @classmethod
    def load_midstate(
        cls,
        path:                  str,
        *,
        rebuild_context_index: bool = True,
    ) -> "IsomorphismPipeline":
        payload = torch.load(path, map_location="cpu", weights_only=False)

        fmt = payload.get("format_version", 0)
        if fmt > cls.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is newer than supported "
                f"({cls.SAVE_FORMAT_VERSION}). Upgrade the code."
            )
        if payload.get("kind") != "midstate":
            raise ValueError(
                f"load_midstate() requires a 'midstate' snapshot; got "
                f"kind={payload.get('kind')!r}."
            )

        cfd_counts = payload.get("cfd_counts")
        if cfd_counts is None:
            raise ValueError(
                "Midstate file is missing 'cfd_counts' — was it written "
                "by an older version of the code?"
            )

        ngram_n = int(payload.get("hparams", {}).get("ngram_n", 2))
        gamma   = float(payload.get("lidstone_gamma", 0.1))
        vocab   = set(payload.get("vocab") or [])
        tokens  = list(payload.get("tokens") or [])

        cpd = _cpd_from_counts(cfd_counts, vocab, gamma)

        ctx_idx = (
            build_real_context_index(vocab, cpd, tokens)
            if rebuild_context_index and tokens
            else None
        )

        pipeline = cls._construct_from_payload(payload, cpd, ctx_idx, vocab, ngram_n)

        missing, unexpected = pipeline.load_state_dict(payload["state_dict"], strict=False)
        if missing or unexpected:
            print(f"  [load_midstate] missing keys: {list(missing)}")
            print(f"  [load_midstate] unexpected keys: {list(unexpected)}")

        if "history" in payload and isinstance(payload["history"], dict):
            pipeline.history = Counter(payload["history"])

        return pipeline

    def load_into(self, path: str, *, strict: bool = False) -> Dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)

        fmt = payload.get("format_version", 0)
        if fmt > self.SAVE_FORMAT_VERSION:
            raise ValueError(
                f"Snapshot format version {fmt} is newer than supported "
                f"({self.SAVE_FORMAT_VERSION})."
            )

        sd = payload.get("state_dict", payload)
        result = self.load_state_dict(sd, strict=strict)

        missing    = list(getattr(result, "missing_keys",    []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])

        if "history" in payload and isinstance(payload["history"], dict):
            self.history = Counter(payload["history"])

        return {
            "missing":        missing,
            "unexpected":     unexpected,
            "kind":           payload.get("kind", "weights"),
            "format_version": fmt,
        }

    # ── last-run frame store ──────────────────────────────────────────
    frames: List[LayerFrame] = []

    @staticmethod
    def _frames_to_text(
        frames:        List[LayerFrame],
        prompt:        str  = "",
        capitalise:    bool = True,
        include_prompt: bool = True,
    ) -> str:
        gen_words = [f.chosen for f in frames if f.chosen]

        prompt_words: List[str] = (
            prompt.strip().split()
            if include_prompt and prompt.strip()
            else []
        )
        all_words = prompt_words + gen_words

        if not all_words:
            return ""

        if not capitalise:
            return " ".join(all_words)

        result: List[str] = []
        cap_next = True
        for w in all_words:
            result.append(w.capitalize() if cap_next else w)
            cap_next = bool(w.rstrip("\"'")[-1:] in {".", "!", "?"})
        return " ".join(result)

    def _make_draw_fn(self, stream, digits_per_sample, seed):
        if stream is not None:
            self.seed_stream(stream)

        active_stream = getattr(self, "_stream", [])

        if active_stream:
            pos   = [self._pos]
            dps   = max(1, int(digits_per_sample))
            s_len = len(active_stream)

            def _draw_pi() -> float:
                val  = 0
                base = 26 ** dps
                for _ in range(dps):
                    val    = val * 26 + active_stream[pos[0] % s_len]
                    pos[0] = (pos[0] + 1) % s_len
                self._pos = pos[0]
                return val / base

            return _draw_pi

        import random as _random
        rng = _random.Random(seed)
        return rng.random

    def generate_text(
        self,
        prompt:     str,
        n_words:    int,
        stream:     Optional[List[int]] = None,
        *,
        digits_per_sample: int          = 3,
        seed:              Optional[int] = None,
        capitalise:        bool          = True,
        include_prompt:    bool          = True,
        zone_fn                          = None,
    ) -> str:
        draw_fn = self._make_draw_fn(stream, digits_per_sample, seed)

        self.frames = list(
            self.generate(
                prompt  = prompt,
                n_words = n_words,
                draw_fn = draw_fn,
                zone_fn = zone_fn,
            )
        )

        return self._frames_to_text(
            self.frames,
            prompt         = prompt,
            capitalise     = capitalise,
            include_prompt = include_prompt,
        )


# ---------------------------------------------------------------------------
# LockedIsomorphismPipeline – 15-layer variant including L14
# ---------------------------------------------------------------------------

class LockedIsomorphismPipeline(IsomorphismPipeline):
    """
    IsomorphismPipeline + L14_LockedStateIndex.

    Adds a previous-state-dependent index dimension keyed by trigram
    prefix.  Once a key is committed during generation it is locked —
    higher transient indexes (later steps) cannot alter the state.
    The unlocked branch uses a Gaussian whose centre is offset by the
    count of observed-but-uncommitted keys ("missing states").

    Adds the following hyper-parameters
    -----------------------------------
        l14_sigma          : Gaussian width (unlocked branch)
        l14_floor          : floor weight
        l14_lock_strength  : peak hardness on the locked token   [0..1]
        l14_blend_alpha    : exponent of L14 in the final blend  (0..1)
                             0 = ignore L14; 1 = pure L14
    """

    LAYER_NAMES = IsomorphismPipeline.LAYER_NAMES + ["L14_LOCKED_STATE_INDEX"]

    def __init__(
        self,
        *args,
        l14_sigma:         float = 0.25,
        l14_floor:         float = 0.03,
        l14_lock_strength: float = 1.0,
        l14_blend_alpha:   float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.l14 = L14_LockedStateIndex(
            sigma         = l14_sigma,
            floor         = l14_floor,
            lock_strength = l14_lock_strength,
        )
        self.l14_blend_alpha = nn.Parameter(
            torch.tensor(l14_blend_alpha, dtype=torch.float64)
        )

        self._init_hparams.update(
            l14_sigma         = float(l14_sigma),
            l14_floor         = float(l14_floor),
            l14_lock_strength = float(l14_lock_strength),
            l14_blend_alpha   = float(l14_blend_alpha),
        )

    # ── lifecycle ────────────────────────────────────────────────────

    def seed_stream(self, stream):
        super().seed_stream(stream)
        self.l14.reset_state()

    def reset_locked_state(self) -> None:
        """Wipe L14's lock table without disturbing the stream."""
        self.l14.reset_state()

    def lock_table_summary(self, limit: int = 50) -> str:
        items = list(self.l14._locked.items())
        if not items:
            return "(lock table empty)"
        lines = [
            f"Locked: {len(items)}   Missing/observed: {self.l14.n_missing}",
            "-" * 60,
        ]
        for k, v in items[:limit]:
            lines.append(f"  {str(k):<40} -> {v}")
        if len(items) > limit:
            lines.append(f"  ... and {len(items) - limit} more")
        return "\n".join(lines)

    # ── step (overrides parent) ──────────────────────────────────────

    def step(
        self,
        context_deque: deque,
        prompt_words:  List[str],
        draw:          float,
        zone_name:     str = "",
    ) -> Optional[LayerFrame]:

        dist = self._dist_for_ctx(tuple(context_deque))
        if dist is None:
            return None

        L0_pairs, L0 = self.l0(dist, self.history)
        if not L0_pairs:
            return None

        L1_pairs, L1 = self.l1(L0_pairs)
        L2_pairs, L2 = self.l2(L1_pairs)
        L3_pairs, L3 = self.l3(L2_pairs)
        if not L3_pairs:
            return None

        ci = self.ctx_idx
        if ci is None:
            flat    = _normalise(torch.ones(len(L3_pairs), dtype=torch.float64))
            flat_np = flat.detach().numpy()
            words   = [w for w, _ in L3_pairs]
            zone_layers = [
                {"name": n, "words": words, "probs": flat_np.copy(),
                 "source": "no context_index"}
                for n in (
                    "L4_ZONE_FREQ", "L5_ZONE_ALPHA", "L6_ZONE_BIGRAM",
                    "L7_ZONE_TRIGRAM", "L8_ZONE_CHAR_TRIG", "L9_ZONE_LATENT",
                )
            ]
        else:
            L4 = self.l4(L3_pairs, prompt_words, ci.freq_zones, ci.token_freq)
            L5 = self.l5(L3_pairs, prompt_words, ci.alpha_zones)
            L6 = self.l6(L3_pairs, prompt_words, ci.ngram_zones)
            L7 = self.l7(L3_pairs, context_deque, ci.ngram_zones)
            L8 = self.l8(L3_pairs, prompt_words, self._char_trig_index)
            L9 = self.l9(
                L3_pairs, prompt_words,
                ci.latent_sorted_keys, ci.latent_bos_data,
            )
            zone_layers = [L4, L5, L6, L7, L8, L9]

        L10            = self.l10(L3_pairs, self.history)
        L11            = self.l11(zone_layers + [L10], L3_pairs)
        L12_pairs, L12 = self.l12(L3_pairs, L11)

        draw_pos   = self._pos
        stream_len = max(1, len(self._stream))
        L13        = self.l13(L3_pairs, draw_pos, stream_len)

        # ── NEW: L14 ─────────────────────────────────────────────────
        L14 = self.l14(L3_pairs, context_deque, draw_pos, stream_len)

        # Geometric blend L12 · L13 · L14
        alpha14   = float(self.l14_blend_alpha.clamp(min=1e-6, max=1.0 - 1e-6))
        floor_val = 1e-12

        l12_map = dict(L12_pairs)
        l13_map = dict(zip(L13["words"], L13["probs"].tolist()))
        l14_map = dict(zip(L14["words"], L14["probs"].tolist()))

        blended = []
        for w in l12_map.keys():
            p12 = min(floor_val, l12_map.get(w, floor_val))
            p13 = torch.argmax(torch.tensor(L13["probs"]))
            p14 = torch.argmax(torch.tensor(L14["probs"]))
            base   = math.sqrt(p12 * p13)
            merged = (base ** (1.0 - alpha14)) * (p14 ** alpha14)
            blended.append((w, merged))

        total = sum(p for _, p in blended)
        if total > 0:
            blended = [(w, p / total) for w, p in blended]

        unseen = [(w, p) for w, p in blended if self.history[w] == 0]
        pool   = unseen if unseen else blended
        t      = sum(p for _, p in pool)
        pool   = [(w, p / t) for w, p in pool] if t > 0 else pool

        chosen, cumulative = pool[-1][0] if pool else "", 0.0
        for w, p in pool:
            cumulative += p
            if draw < cumulative:
                chosen = w
                break

        # COMMIT — the write-once rule in action
        key = L14_LockedStateIndex.key_from_ctx(context_deque)
        if key and chosen:
            self.l14.commit(key, chosen)

        self.history[chosen] += 1

        next_draw_pos = (draw_pos + (draw_pos % max(1, stream_len))) % stream_len
        self._pos     = next_draw_pos
        self._step   += 1

        return LayerFrame(
            step           = self._step - 1,
            layers         = [L0, L1, L2, L3] + zone_layers
                              + [L10, L11, L12, L13, L14],
            chosen         = chosen,
            context_window = tuple(context_deque),
            zone_name      = zone_name,
            draw_pos       = draw_pos,
            next_draw_pos  = next_draw_pos,
        )


# ---------------------------------------------------------------------------
# Real corpus builder
# ---------------------------------------------------------------------------

def _cpd_from_counts(
    cfd_counts:     Dict[Tuple[str, ...], Dict[str, int]],
    vocab:          set,
    lidstone_gamma: float = 0.1,
):
    from nltk.probability import (
        ConditionalFreqDist, ConditionalProbDist, LidstoneProbDist, FreqDist,
    )

    cfd = ConditionalFreqDist()
    for ctx, counts in cfd_counts.items():
        ctx_key = tuple(ctx)
        fd = FreqDist()
        for word, c in counts.items():
            fd[word] = int(c)
        cfd[ctx_key] = fd

    class _LidFactory:
        def __init__(self, gamma, bins):
            self.gamma = gamma; self.bins = bins
        def __call__(self, fd):
            return LidstoneProbDist(fd, gamma=self.gamma, bins=self.bins)

    return ConditionalProbDist(
        cfd,
        _LidFactory(gamma=float(lidstone_gamma), bins=max(1, len(vocab))),
    )


def build_real_cpd(corpus: str, ngram_n: int = 2, lidstone_gamma: float = 0.1):
    import os
    import nltk
    from nltk.util import ngrams as nltk_ngrams

    NLTK_DATA_DIR = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    if NLTK_DATA_DIR not in nltk.data.path:
        nltk.data.path.insert(0, NLTK_DATA_DIR)
    for pkg, path in [("punkt", "tokenizers/punkt"),
                      ("punkt_tab", "tokenizers/punkt_tab")]:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)
            except Exception:
                pass

    tokens = corpus.lower().split()
    if not tokens:
        raise ValueError("Corpus produced zero tokens.")

    ngram_n = max(2, int(ngram_n))
    padded  = [""] * (ngram_n - 1) + tokens + [""]
    all_ng  = list(nltk_ngrams(padded, ngram_n))

    cfd_counts: Dict[Tuple[str, ...], Dict[str, int]] = {}
    for ng in all_ng:
        ctx, word = tuple(ng[:-1]), ng[-1]
        cfd_counts.setdefault(ctx, {})
        cfd_counts[ctx][word] = cfd_counts[ctx].get(word, 0) + 1

    vocab = set(tokens) | {""}
    cpd   = _cpd_from_counts(cfd_counts, vocab, lidstone_gamma)
    return cpd, vocab, tokens


def build_real_context_index(vocab, cpd, tokens):
    """Build a ContextZoneIndex from real corpus tokens (if app.py is available)."""
    try:
        import sys, os
        _here = os.path.dirname(os.path.abspath(__file__))
        if _here not in sys.path:
            sys.path.insert(0, _here)
        from app import ContextZoneIndex
        return ContextZoneIndex(vocab, cpd, Counter(tokens))
    except Exception as e:
        print(f"  [context_index] not available ({e}); zone layers will use uniform weights.")
        return None



# =============================================================================
#  Gradio helpers
# =============================================================================

def _make_heatmap(frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not frames:
        return None

    layer_names = [layer["name"] for layer in frames[0].layers]
    n_layers = len(layer_names)
    n_steps  = len(frames)

    mat = np.zeros((n_layers, n_steps))
    for s, frame in enumerate(frames):
        for r, layer in enumerate(frame.layers):
            words = layer.get("words", [])
            probs = layer.get("probs", np.array([]))
            if frame.chosen in words and len(probs):
                idx = words.index(frame.chosen)
                if idx < len(probs):
                    mat[r, s] = float(probs[idx])

    fig, ax = plt.subplots(figsize=(max(8, n_steps * 0.30), max(4, n_layers * 0.45)))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                   cmap="viridis", origin="upper")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_names, fontsize=7)
    ax.set_xlabel("Generation step")
    ax.set_ylabel("Layer")
    ax.set_title("P(chosen token) — per layer x step", fontsize=9)
    fig.colorbar(im, ax=ax, label="probability", shrink=0.8)
    fig.tight_layout()
    return fig


def _make_step_log(frames, limit=40):
    if not frames:
        return ""
    header = "{:>4}  {:<18} {:>9}  {:>9}".format("Step", "Chosen", "draw_pos", "next_pos")
    lines  = [header, "-" * len(header)]
    for i, f in enumerate(frames[:limit]):
        lines.append("{:>4}  {:<18} {:>9}  {:>9}".format(
            i, f.chosen, f.draw_pos, f.next_draw_pos))
    if len(frames) > limit:
        lines.append("... ({} more steps not shown)".format(len(frames) - limit))
    return "\\n".join(lines)


def run_generation(
    corpus_file, prompt, n_words, use_locked, seed,
    ngram_n, lidstone_gamma,
    temperature, top_k, top_p,
    rep_penalty, insight_penalty, l12_blend_alpha,
    l13_sigma, l13_floor,
    l14_sigma, l14_floor, l14_lock_strength, l14_blend_alpha,
    progress=gr.Progress(track_tqdm=True),
):
    # read corpus
    if corpus_file is None:
        return "Please upload a corpus .txt file.", "", None, ""
    try:
        path = corpus_file.name if hasattr(corpus_file, "name") else str(corpus_file)
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            corpus_text = fh.read()
    except Exception as exc:
        return "Could not read corpus file:\\n{}".format(exc), "", None, ""

    if not corpus_text.strip():
        return "The uploaded corpus file is empty.", "", None, ""
    if not str(prompt).strip():
        return "Please enter a prompt.", "", None, ""

    try:
        progress(0.10, desc="Building CPD & context index ...")
        cpd, vocab, tokens = build_real_cpd(
            corpus_text, ngram_n=int(ngram_n), lidstone_gamma=float(lidstone_gamma)
        )
        ctx_idx = build_real_context_index(vocab, cpd, tokens)

        progress(0.40, desc="Constructing pipeline ...")
        cls = LockedIsomorphismPipeline if bool(use_locked) else IsomorphismPipeline
        kwargs = dict(
            cpd=cpd, context_index=ctx_idx, vocab=vocab,
            ngram_n=int(ngram_n),
            temperature=float(temperature),
            top_k=int(top_k), top_p=float(top_p),
            rep_penalty=float(rep_penalty),
            insight_penalty=float(insight_penalty),
            l12_blend_alpha=float(l12_blend_alpha),
            l13_sigma=float(l13_sigma), l13_floor=float(l13_floor),
        )
        if bool(use_locked):
            kwargs.update(
                l14_sigma=float(l14_sigma), l14_floor=float(l14_floor),
                l14_lock_strength=float(l14_lock_strength),
                l14_blend_alpha=float(l14_blend_alpha),
            )
        pipe = cls(**kwargs)

        progress(0.60, desc="Generating text ...")
        seed_val = int(seed) if seed is not None else None
        text = pipe.generate_text(
            prompt=str(prompt).strip(),
            n_words=int(n_words),
            seed=seed_val,
        )

        progress(0.85, desc="Building heatmap ...")
        frames   = getattr(pipe, "frames", [])
        heatmap  = _make_heatmap(frames)
        step_log = _make_step_log(frames)

        param_summary = pipe.param_summary()
        if bool(use_locked) and hasattr(pipe, "lock_table_summary"):
            param_summary += "\\n\\n" + pipe.lock_table_summary()

        progress(1.0, desc="Done!")
        return text, param_summary, heatmap, step_log

    except Exception:
        return "Error:\\n\\n{}".format(traceback.format_exc()), "", None, ""
def buildhfsquadpipeline(
    dataset_name: str = "squad",
    config_name: str | None = None,
    split_names: Sequence[str] = ("train", "validation"),
    *,
    locked: bool = True,
    ngram_n: int = 3,
    lidstone_gamma: float = 0.1,
    minsentencelen: int = 3,
    pipelinekwargs: Optional[Dict] = None,
    preprocessorkwargs: Optional[Dict] = None,
) -> Tuple[IsomorphismPipeline, HFSquadSentenceDatasetPreprocessor]:
    """
    Hugging Face SQuAD -> HFSquadSentenceDatasetPreprocessor -> CPD/context index -> pipeline

    Mirrors buildsentencepipeline() but sources corpus entities from a HF SQuAD dataset
    instead of raw text sentence splitting.
    """
    pre = HFSquadSentenceDatasetPreprocessor(
        dataset_name=dataset_name,
        config_name=config_name,
        split_names=split_names,
        minsentencelen=minsentencelen,
        **(preprocessorkwargs or {}),
    )
    corpustext = pre.tocorpus()
    cpd, vocab, tokens = build_real_cpd(
        corpustext,
        ngram_n=int(ngram_n),
        lidstone_gamma=float(lidstone_gamma),
    )
    ctxidx = build_real_context_index(vocab, cpd, tokens)

    cls = LockedIsomorphismPipeline if locked else IsomorphismPipeline
    kwargs = dict(
        cpd=cpd,
        context_index=ctxidx,
        vocab=vocab,
        ngram_n=int(ngram_n),
    )
    if pipelinekwargs:
        kwargs.update(pipelinekwargs)

    pipe = cls(**kwargs)
    pipe.preprocessor = pre
    return pipe, pre
# =============================================================================
#  hf_dataset_preprocessor.py
#
#  Generic Hugging Face dataset preprocessor for the isomorphism pipeline.
#
#  Replaces (and supersedes) HFSquadSentenceDatasetPreprocessor with a
#  schema-agnostic version that works on any dataset on the Hub.
#
#  Public API surface mirrors SentenceDatasetPreprocessor exactly enough
#  that SentenceAwareGenerator, build_real_cpd, build_real_context_index,
#  and the rest of layer_isomorphism_torch.py continue to work unchanged.
#
#  Drop this file next to layer_isomorphism_torch.py and import from it.
# =============================================================================


import random
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union,
)

from datasets import load_dataset


# -----------------------------------------------------------------------------
# Field spec
# -----------------------------------------------------------------------------
#
# A field spec is either:
#   * a dotted-path string:  "context"  /  "answers.text"  /  "translation.en"
#       - supports list indexing: "answers.text[0]"
#       - resolves to: a string, a list of strings, a number, or None
#
#   * a callable:  fn(example) -> str | list[str] | None
#       - use this for anything the dotted-path syntax can't express
#         (e.g. join a list of turns, format a label, conditional logic)
#
FieldSpec = Union[str, Callable[[Dict[str, Any]], Union[str, List[str], None]]]


_INDEXED_PART_RE = re.compile(r"^([^\[]+)\[(\d+)\]$")


def _resolve_path(obj: Any, path: str) -> Any:
    """Walk a dotted path through nested dicts/lists. Returns None on miss."""
    if obj is None:
        return None
    cur = obj
    for part in path.split("."):
        m = _INDEXED_PART_RE.match(part)
        if m:
            key, idx = m.group(1), int(m.group(2))
            if isinstance(cur, dict):
                cur = cur.get(key)
            else:
                return None
            if not isinstance(cur, (list, tuple)) or idx >= len(cur):
                return None
            cur = cur[idx]
        else:
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                return None
        if cur is None:
            return None
    return cur


def _extract_field(example: Dict, spec: FieldSpec) -> str:
    """Resolve a FieldSpec into a single string."""
    if callable(spec):
        val = spec(example)
    else:
        val = _resolve_path(example, spec)

    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, (list, tuple)):
        parts: List[str] = []
        for v in val:
            if isinstance(v, str) and v:
                parts.append(v)
            elif isinstance(v, (int, float, bool)):
                parts.append(str(v))
            elif isinstance(v, dict):
                parts.append(" ".join(
                    str(x) for x in v.values() if isinstance(x, (str, int, float))
                ))
        return " ".join(parts)
    if isinstance(val, (int, float, bool)):
        return str(val)
    if isinstance(val, dict):
        return " ".join(str(v) for v in val.values() if isinstance(v, (str, int, float)))
    return str(val)


# -----------------------------------------------------------------------------
# Audit record
# -----------------------------------------------------------------------------

@dataclass
class HFGenericRecord:
    split: str
    original_index: int
    id: str
    tokens: List[str]
    first_token: str
    last_token: str
    kept: bool
    drop_reason: str
    raw_text: str


# -----------------------------------------------------------------------------
# Preset registry — common HF datasets in one line.
# -----------------------------------------------------------------------------
#
# Each value is a dict of constructor kwargs.  Override anything you want
# at call time, e.g.:
#   HFSentenceDatasetPreprocessor.from_preset("cnn_dailymail", boundaryquota=2)
#
HF_DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    # ── Question answering ───────────────────────────────────────────
    "squad":             {"text_fields": ["question", "context", "answers.text"]},
    "squad_v2":          {"text_fields": ["question", "context", "answers.text"]},
    "trivia_qa":         {"config_name": "rc",
                          "text_fields": ["question", "answer.value"]},
    "hotpot_qa":         {"config_name": "distractor",
                          "text_fields": ["question", "answer"]},

    # ── Text classification (single-text) ────────────────────────────
    "imdb":              {"text_fields": ["text"]},
    "ag_news":           {"text_fields": ["text"]},
    "yelp_review_full":  {"text_fields": ["text"]},
    "yelp_polarity":     {"text_fields": ["text"]},
    "amazon_polarity":   {"text_fields": ["title", "content"]},
    "dbpedia_14":        {"text_fields": ["title", "content"]},
    "tweet_eval":        {"config_name": "emotion", "text_fields": ["text"]},

    # ── GLUE family (override config_name to switch task) ────────────
    "glue_sst2":         {"dataset_name": "glue", "config_name": "sst2",
                          "text_fields": ["sentence"]},
    "glue_cola":         {"dataset_name": "glue", "config_name": "cola",
                          "text_fields": ["sentence"]},
    "glue_mrpc":         {"dataset_name": "glue", "config_name": "mrpc",
                          "text_fields": ["sentence1", "sentence2"]},
    "glue_qqp":          {"dataset_name": "glue", "config_name": "qqp",
                          "text_fields": ["question1", "question2"]},
    "glue_mnli":         {"dataset_name": "glue", "config_name": "mnli",
                          "text_fields": ["premise", "hypothesis"]},

    # ── Summarisation ────────────────────────────────────────────────
    "cnn_dailymail":     {"config_name": "3.0.0",
                          "text_fields": ["article", "highlights"]},
    "xsum":              {"text_fields": ["document", "summary"]},
    "billsum":           {"text_fields": ["text", "summary"]},
    "samsum":            {"text_fields": ["dialogue", "summary"]},
    "multi_news":        {"text_fields": ["document", "summary"]},
    "reddit_tifu":       {"config_name": "long",
                          "text_fields": ["title", "documents", "tldr"]},

    # ── Language modelling ───────────────────────────────────────────
    "wikitext":          {"config_name": "wikitext-2-raw-v1",
                          "text_fields": ["text"]},
    "wikitext_103":      {"dataset_name": "wikitext",
                          "config_name": "wikitext-103-raw-v1",
                          "text_fields": ["text"]},
    "bookcorpus":        {"text_fields": ["text"]},
    "openwebtext":       {"text_fields": ["text"]},
    "c4":                {"config_name": "en", "streaming": True,
                          "text_fields": ["text"]},

    # ── Dialogue ─────────────────────────────────────────────────────
    "daily_dialog":      {"text_fields": [
                              lambda ex: " ".join(ex.get("dialog", []) or [])
                          ]},
    "blended_skill_talk":{"text_fields": [
                              lambda ex: " ".join(ex.get("free_messages", []) or []),
                              lambda ex: " ".join(ex.get("guided_messages", []) or []),
                          ]},
    "empathetic_dialogues":{"text_fields": ["prompt", "utterance"]},

    # ── Translation (English side; flip the lambda for other side) ───
    "wmt14_de_en":       {"dataset_name": "wmt14", "config_name": "de-en",
                          "text_fields": ["translation.en"]},
    "wmt16_de_en":       {"dataset_name": "wmt16", "config_name": "de-en",
                          "text_fields": ["translation.en"]},
    "opus_books":        {"config_name": "en-fr",
                          "text_fields": ["translation.en"]},

    # ── NER / token classification (flatten the token list) ──────────
    "conll2003":         {"text_fields": [
                              lambda ex: " ".join(ex.get("tokens", []) or [])
                          ]},
    "wnut_17":           {"text_fields": [
                              lambda ex: " ".join(ex.get("tokens", []) or [])
                          ]},

    # ── Common-sense / multiple choice ───────────────────────────────
    "commonsense_qa":    {"text_fields": ["question", "choices.text"]},
    "openbookqa":        {"config_name": "main",
                          "text_fields": ["question_stem", "choices.text"]},
}


# -----------------------------------------------------------------------------
# The generic preprocessor
# -----------------------------------------------------------------------------

class HFSentenceDatasetPreprocessor:
    """
    Generic HF-dataset preprocessor that enforces the same quota-balanced
    boundary invariant as SentenceDatasetPreprocessor.

    Each surviving dataset entity becomes one token sequence. Acceptance
    is greedy in dataset order. With boundaryquota=1 you get globally
    unique sequence beginnings AND endings, which is what the
    L7/L14/SentenceAwareGenerator stack relies on.

    Quick start
    -----------
        # Use a preset (one-liner)
        pre = HFSentenceDatasetPreprocessor.from_preset("imdb")

        # Custom — pick fields yourself
        pre = HFSentenceDatasetPreprocessor(
            "squad",
            text_fields=["question", "context", "answers.text"],
            boundaryquota=1,
        )

        # Auto-detect text fields (good for simple single/dual-text datasets)
        pre = HFSentenceDatasetPreprocessor("ag_news")

        # Streaming for huge datasets — cap with max_examples_per_split
        pre = HFSentenceDatasetPreprocessor(
            "c4", config_name="en",
            text_fields=["text"],
            streaming=True,
            max_examples_per_split=10_000,
        )

    Parameters
    ----------
    dataset_name : str
        Dataset id on the HF Hub (e.g. "squad", "imdb", "cnn_dailymail").
    config_name : str | None
        Required for multi-config datasets (e.g. "wikitext-2-raw-v1").
    text_fields : Sequence[FieldSpec] | None
        Ordered list of dotted paths or callables. None = auto-detect.
    split_names : Sequence[str] | None
        Which splits to pull from. None = every split the dataset exposes.
    field_sep : str | None
        Optional separator token inserted between fields when joining.
    lowercase : bool
        Lowercase before tokenisation.
    minsentencelen : int
        Drop entities shorter than this (in tokens).
    max_examples_per_split : int | None
        Hard cap for streaming or just to keep things small.
    uniquemiddlepool : bool
        Deduplicate middle-pool tokens (matches SentenceDatasetPreprocessor).
    strict : bool
        Enforce the boundary quota.  Disable to keep every entity.
    boundaryquota : int
        Per-token cap on appearances as first/last token. 1 = globally unique.
    streaming : bool
        Pass through to load_dataset; auto-detect of fields is skipped.
    trust_remote_code : bool
        Some HF datasets require this in recent versions.
    id_field : FieldSpec | None
        Where to read a per-example id. Falls back to common keys, then split-idx.
    token_pattern : str | None
        Regex tokeniser; None = whitespace split.
    keep_alpha_only : bool
        Filter to alpha-only tokens after tokenisation.
    exclude_auto_fields : Sequence[str]
        When auto-detecting, skip columns with these names (ids, urls, ...).
    """

    DEFAULT_EXCLUDED_AUTO = (
        "id", "uid", "_id", "example_id", "qid",
        "url", "title", "source", "filename", "doc_id", "subset",
    )

    # ─────────────────────────────────────────────────────────────────
    # construction
    # ─────────────────────────────────────────────────────────────────

    def __init__(
        self,
        dataset_name: str,
        config_name: Optional[str] = None,
        text_fields: Optional[Sequence[FieldSpec]] = None,
        *,
        split_names: Optional[Sequence[str]] = None,
        field_sep: Optional[str] = None,
        lowercase: bool = True,
        minsentencelen: int = 3,
        max_examples_per_split: Optional[int] = None,
        uniquemiddlepool: bool = True,
        strict: bool = True,
        boundaryquota: int = 1,
        streaming: bool = False,
        trust_remote_code: bool = False,
        id_field: Optional[FieldSpec] = None,
        token_pattern: Optional[str] = None,
        keep_alpha_only: bool = False,
        exclude_auto_fields: Sequence[str] = DEFAULT_EXCLUDED_AUTO,
    ):
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.text_fields: Optional[List[FieldSpec]] = (
            list(text_fields) if text_fields else None
        )
        self.split_names = tuple(split_names) if split_names else None
        self.field_sep = field_sep
        self.lowercase = bool(lowercase)
        self.minsentencelen = max(2, int(minsentencelen))
        self.max_examples_per_split = max_examples_per_split
        self.uniquemiddlepool = bool(uniquemiddlepool)
        self.strict = bool(strict)
        self.boundaryquota = max(1, int(boundaryquota))
        self.streaming = bool(streaming)
        self.trust_remote_code = bool(trust_remote_code)
        self.id_field = id_field
        self.token_pattern = token_pattern
        self.keep_alpha_only = bool(keep_alpha_only)
        self.exclude_auto_fields = set(exclude_auto_fields)

        # outputs (match SentenceDatasetPreprocessor)
        self.sentences: List[List[str]] = []
        self.beginnings: List[str] = []
        self.endings: List[str] = []
        self.middlepool: List[str] = []
        self.tokens: List[str] = []

        self.records: List[HFGenericRecord] = []
        self.keptrecords: List[HFGenericRecord] = []
        self.droppedrecords: List[HFGenericRecord] = []

        self.dropped = 0
        self.skipped = 0
        self.begincounts: Counter = Counter()
        self.endcounts: Counter = Counter()

        self.beginningsset: Set[str] = set()
        self.endingsset: Set[str] = set()
        self.middleset: Set[str] = set()

        # bookkeeping
        self._auto_detected_fields: List[str] = []
        self._detected_splits: List[str] = []

        self.process()

    # ─────────────────────────────────────────────────────────────────
    # preset constructor
    # ─────────────────────────────────────────────────────────────────

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "HFSentenceDatasetPreprocessor":
        """Build from a registered preset; any kwargs override the preset."""
        if preset_name not in HF_DATASET_PRESETS:
            raise KeyError(
                f"Unknown preset {preset_name!r}. "
                f"Available: {sorted(HF_DATASET_PRESETS)}"
            )
        spec = dict(HF_DATASET_PRESETS[preset_name])
        spec.setdefault("dataset_name", preset_name)
        spec.update(overrides)
        return cls(**spec)

    # ─────────────────────────────────────────────────────────────────
    # tokenisation
    # ─────────────────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        if self.lowercase:
            text = text.lower()
        if self.token_pattern:
            toks = re.findall(self.token_pattern, text)
        else:
            toks = text.split()
        if self.keep_alpha_only:
            toks = [t for t in toks if t.isalpha()]
        return toks

    # ─────────────────────────────────────────────────────────────────
    # dataset loading
    # ─────────────────────────────────────────────────────────────────

    def _load(self):
        kwargs: Dict[str, Any] = {"streaming": self.streaming}
        if self.trust_remote_code:
            kwargs["trust_remote_code"] = True
        if self.config_name is None:
            return load_dataset(self.dataset_name, **kwargs)
        return load_dataset(self.dataset_name, self.config_name, **kwargs)

    # ─────────────────────────────────────────────────────────────────
    # auto-detect string-valued fields
    # ─────────────────────────────────────────────────────────────────

    def _autodetect_fields(self, dsd) -> List[str]:
        """Pick every Value('string') / Sequence(Value('string')) feature."""
        try:
            from datasets import Value
            from datasets import Sequence as HFSequence
        except Exception:  # pragma: no cover - very old datasets versions
            return []

        # find the first split that exposes .features
        feats = None
        for name in dsd:
            split_obj = dsd[name]
            if hasattr(split_obj, "features") and split_obj.features is not None:
                feats = split_obj.features
                break
        if feats is None:
            return []

        detected: List[str] = []
        for name, feat in feats.items():
            if name in self.exclude_auto_fields:
                continue
            try:
                if isinstance(feat, Value) and feat.dtype == "string":
                    detected.append(name)
                elif isinstance(feat, HFSequence):
                    inner = feat.feature
                    if isinstance(inner, Value) and inner.dtype == "string":
                        detected.append(name)
                elif isinstance(feat, list) and feat:
                    inner = feat[0]
                    if isinstance(inner, Value) and inner.dtype == "string":
                        detected.append(name)
            except Exception:
                continue
        return detected

    # ─────────────────────────────────────────────────────────────────
    # text assembly
    # ─────────────────────────────────────────────────────────────────

    def _build_text(self, example: Dict) -> str:
        parts: List[str] = []
        for spec in self.text_fields or []:
            chunk = _extract_field(example, spec)
            if chunk:
                if self.field_sep and parts:
                    parts.append(self.field_sep)
                parts.append(chunk)
        return " ".join(parts).strip()

    def _make_id(self, split: str, idx: int, example: Dict) -> str:
        if self.id_field is not None:
            v = _extract_field(example, self.id_field)
            if v:
                return v
        for k in ("id", "uid", "_id", "example_id", "qid"):
            if k in example and example[k] is not None:
                return str(example[k])
        return f"{split}-{idx}"

    # ─────────────────────────────────────────────────────────────────
    # iteration
    # ─────────────────────────────────────────────────────────────────

    def _iter(self):
        dsd = self._load()

        # discover splits
        try:
            all_splits = list(dsd.keys())
        except Exception:
            all_splits = ["train"]
        self._detected_splits = list(all_splits)

        splits = self.split_names or all_splits

        # auto-detect fields if user didn't supply any
        if not self.text_fields:
            if self.streaming:
                raise ValueError(
                    "Auto-detection of text_fields is not supported in streaming "
                    "mode. Pass text_fields=[...] explicitly."
                )
            self.text_fields = self._autodetect_fields(dsd)
            self._auto_detected_fields = [
                s for s in self.text_fields if isinstance(s, str)
            ]
            if not self.text_fields:
                raise ValueError(
                    f"Could not auto-detect any string-valued text fields in "
                    f"{self.dataset_name!r}. Pass text_fields=[...] explicitly."
                )

        for split in splits:
            if split not in dsd:
                continue
            ds = dsd[split]
            limit = self.max_examples_per_split
            for idx, ex in enumerate(ds):
                if limit is not None and idx >= limit:
                    break
                yield split, idx, ex

    # ─────────────────────────────────────────────────────────────────
    # per-example record + quota enforcement
    # ─────────────────────────────────────────────────────────────────

    def _record(self, split: str, idx: int, ex: Dict) -> HFGenericRecord:
        raw = self._build_text(ex)
        toks = self._tokenize(raw)
        rid = self._make_id(split, idx, ex)

        if len(toks) < self.minsentencelen:
            self.skipped += 1
            return HFGenericRecord(
                split=split, original_index=idx, id=rid, tokens=toks,
                first_token=toks[0] if toks else "",
                last_token=toks[-1] if toks else "",
                kept=False, drop_reason=f"too_short_lt_{self.minsentencelen}",
                raw_text=raw,
            )

        first, last = toks[0], toks[-1]

        if self.strict:
            if self.begincounts[first] >= self.boundaryquota:
                self.dropped += 1
                return HFGenericRecord(
                    split=split, original_index=idx, id=rid, tokens=toks,
                    first_token=first, last_token=last,
                    kept=False, drop_reason=f"begin_quota_full:{first}",
                    raw_text=raw,
                )
            if self.endcounts[last] >= self.boundaryquota:
                self.dropped += 1
                return HFGenericRecord(
                    split=split, original_index=idx, id=rid, tokens=toks,
                    first_token=first, last_token=last,
                    kept=False, drop_reason=f"end_quota_full:{last}",
                    raw_text=raw,
                )
            self.begincounts[first] += 1
            self.endcounts[last] += 1

        return HFGenericRecord(
            split=split, original_index=idx, id=rid, tokens=toks,
            first_token=first, last_token=last,
            kept=True, drop_reason="", raw_text=raw,
        )

    def process(self) -> None:
        orderedpool: List[str] = []
        for split, idx, ex in self._iter():
            rec = self._record(split, idx, ex)
            self.records.append(rec)
            if rec.kept:
                self.keptrecords.append(rec)
                s = rec.tokens
                self.sentences.append(s)
                self.beginnings.append(s[0])
                self.endings.append(s[-1])
                orderedpool.extend(s[1:-1])
                self.tokens.extend(s)
            else:
                self.droppedrecords.append(rec)

        if self.uniquemiddlepool:
            seen: Set[str] = set()
            self.middlepool = []
            for w in orderedpool:
                if w not in seen:
                    seen.add(w)
                    self.middlepool.append(w)
        else:
            self.middlepool = orderedpool

        self.beginningsset = set(self.beginnings)
        self.endingsset = set(self.endings)
        self.middleset = set(self.middlepool)

        if not self.sentences:
            raise ValueError(
                f"No entities from {self.dataset_name!r} survived the "
                f"quota-boundary invariant (quota={self.boundaryquota}, "
                f"dropped={self.dropped}, skipped={self.skipped}). "
                f"Try loosening minsentencelen, raising boundaryquota, "
                f"or widening text_fields."
            )

    # ─────────────────────────────────────────────────────────────────
    # API surface (mirrors SentenceDatasetPreprocessor)
    # ─────────────────────────────────────────────────────────────────

    def tocorpus(self) -> str:
        return " ".join(self.tokens)

    def vocab(self) -> set:
        return set(self.tokens)

    def isbeginning(self, token: str) -> bool:
        return token in self.beginningsset

    def isnaturalending(self, token: str) -> bool:
        return token in self.endingsset

    def samplearbitrary(
        self,
        rngvalue: Optional[float] = None,
        rng: Optional[random.Random] = None,
    ) -> str:
        if not self.middlepool:
            return ""
        n = len(self.middlepool)
        if rngvalue is not None:
            return self.middlepool[int(rngvalue * n) % n]
        rng = rng or random
        return self.middlepool[rng.randrange(n)]

    # SentenceAwareGenerator in your file calls `sample_arbitrary`
    # (with underscore). Keep both names alive.
    sample_arbitrary = samplearbitrary
    is_beginning = isbeginning
    is_natural_ending = isnaturalending
    to_corpus = tocorpus

    def boundarybalancereport(self) -> str:
        def stats(c: Counter, label: str) -> str:
            if not c:
                return f"{label}: empty"
            vals = list(c.values())
            mn, mx = min(vals), max(vals)
            avg = sum(vals) / len(vals)
            perfect = all(v == vals[0] for v in vals)
            return (
                f"{label}: {len(c)} words, min={mn} max={mx} avg={avg:.2f} "
                f"{'perfectly balanced' if perfect else 'imbalanced'}"
            )

        return "\n".join([
            f"Boundary quota {self.boundaryquota}",
            stats(self.begincounts, "beginnings"),
            stats(self.endcounts,  "endings"),
        ])

    def summary(self) -> str:
        nsent = len(self.sentences)
        avg = sum(len(s) for s in self.sentences) / max(1, nsent)
        fields_repr = [
            (s if isinstance(s, str) else "<callable>")
            for s in (self.text_fields or [])
        ]
        return "\n".join([
            "HFSentenceDatasetPreprocessor",
            f"dataset {self.dataset_name}"
            + (f"  config {self.config_name}" if self.config_name else ""),
            f"detected splits {self._detected_splits}",
            f"text fields {fields_repr}",
            f"auto-detected {bool(self._auto_detected_fields)}",
            f"boundaryquota {self.boundaryquota}",
            f"entities kept {nsent}",
            f"dropped (quota) {self.dropped}",
            f"skipped (too short) {self.skipped}",
            f"total tokens {len(self.tokens)}",
            f"vocab size {len(self.vocab())}",
            f"unique beginnings {len(self.beginningsset)}",
            f"unique endings {len(self.endingsset)}",
            f"middle-pool size {len(self.middlepool)}",
            f"avg entity len {avg:.2f}",
            f"beginning example {self.beginnings[0] if self.beginnings else ''}",
            f"ending example {self.endings[0] if self.endings else ''}",
            f"middle example {self.middlepool[0] if self.middlepool else ''}",
            self.boundarybalancereport(),
        ])

    def auditrows(self) -> List[Dict]:
        return [asdict(r) for r in self.records]

    def keptrows(self) -> List[Dict]:
        return [asdict(r) for r in self.keptrecords]

    def droppedrows(self) -> List[Dict]:
        return [asdict(r) for r in self.droppedrecords]


# -----------------------------------------------------------------------------
# Back-compat: keep the old SQuAD class name as a thin specialisation.
# -----------------------------------------------------------------------------

class HFSquadSentenceDatasetPreprocessor(HFSentenceDatasetPreprocessor):
    """
    Drop-in replacement for the original SQuAD preprocessor. Keeps the old
    qca_mode / include_* knobs working but delegates everything to the
    generic class.
    """

    def __init__(
        self,
        dataset_name: str = "squad",
        config_name: Optional[str] = None,
        split_names: Sequence[str] = ("train", "validation"),
        lowercase: bool = True,
        minsentencelen: int = 3,
        uniquemiddlepool: bool = True,
        strict: bool = True,
        boundaryquota: int = 1,
        include_question: bool = True,
        include_context: bool = True,
        include_answer: bool = True,
        qca_mode: str = "question_context_answer",
        sep_qc: Optional[str] = None,
        sep_ca: Optional[str] = None,
        **kwargs,
    ):
        text_fields = self._squad_fields(
            qca_mode, include_question, include_context, include_answer,
        )
        super().__init__(
            dataset_name=dataset_name,
            config_name=config_name,
            text_fields=text_fields,
            split_names=split_names,
            lowercase=lowercase,
            minsentencelen=minsentencelen,
            uniquemiddlepool=uniquemiddlepool,
            strict=strict,
            boundaryquota=boundaryquota,
            **kwargs,
        )

    @staticmethod
    def _squad_fields(
        qca_mode: str,
        include_question: bool,
        include_context: bool,
        include_answer: bool,
    ) -> List[FieldSpec]:
        mode = qca_mode.lower()
        fields: List[FieldSpec] = []
        if mode == "question_only":
            if include_question: fields.append("question")
        elif mode == "question_answer":
            if include_question: fields.append("question")
            if include_answer:   fields.append("answers.text")
        elif mode == "question_context":
            if include_question: fields.append("question")
            if include_context:  fields.append("context")
        else:  # question_context_answer
            if include_question: fields.append("question")
            if include_context:  fields.append("context")
            if include_answer:   fields.append("answers.text")
        return fields


# -----------------------------------------------------------------------------
# Generic pipeline builder
# -----------------------------------------------------------------------------

def build_hf_pipeline(
    dataset_name: str,
    config_name: Optional[str] = None,
    text_fields: Optional[Sequence[FieldSpec]] = None,
    *,
    split_names: Optional[Sequence[str]] = None,
    locked: bool = True,
    ngram_n: int = 3,
    lidstone_gamma: float = 0.1,
    minsentencelen: int = 3,
    pipelinekwargs: Optional[Dict] = None,
    preprocessorkwargs: Optional[Dict] = None,
) -> Tuple[IsomorphismPipeline, HFSentenceDatasetPreprocessor]:
    """
    Generic HF-dataset → CPD → context index → pipeline.

    Anything you don't specify here can still be passed via
    ``preprocessorkwargs`` (forwarded to HFSentenceDatasetPreprocessor)
    and ``pipelinekwargs`` (forwarded to the pipeline class).
    """
    pre = HFSentenceDatasetPreprocessor(
        dataset_name=dataset_name,
        config_name=config_name,
        text_fields=text_fields,
        split_names=split_names,
        minsentencelen=minsentencelen,
        **(preprocessorkwargs or {}),
    )

    corpus = pre.tocorpus()
    cpd, vocab, tokens = build_real_cpd(
        corpus, ngram_n=int(ngram_n), lidstone_gamma=float(lidstone_gamma),
    )
    ctxidx = build_real_context_index(vocab, cpd, tokens)

    cls = LockedIsomorphismPipeline if locked else IsomorphismPipeline
    kwargs: Dict[str, Any] = dict(
        cpd=cpd, context_index=ctxidx, vocab=vocab, ngram_n=int(ngram_n),
    )
    if pipelinekwargs:
        kwargs.update(pipelinekwargs)

    pipe = cls(**kwargs)
    pipe.preprocessor = pre
    return pipe, pre


def build_hf_pipeline_from_preset(
    preset_name: str,
    *,
    locked: bool = True,
    ngram_n: int = 3,
    lidstone_gamma: float = 0.1,
    minsentencelen: int = 3,
    pipelinekwargs: Optional[Dict] = None,
    preprocessor_overrides: Optional[Dict] = None,
) -> Tuple[IsomorphismPipeline, HFSentenceDatasetPreprocessor]:
    """One-liner: preset name -> ready pipeline + preprocessor."""
    if preset_name not in HF_DATASET_PRESETS:
        raise KeyError(
            f"Unknown preset {preset_name!r}. "
            f"Available: {sorted(HF_DATASET_PRESETS)}"
        )
    spec = dict(HF_DATASET_PRESETS[preset_name])
    spec.setdefault("dataset_name", preset_name)
    if preprocessor_overrides:
        spec.update(preprocessor_overrides)

    pre = HFSentenceDatasetPreprocessor(
        minsentencelen=minsentencelen,
        **spec,
    )
    corpus = pre.tocorpus()
    cpd, vocab, tokens = build_real_cpd(
        corpus, ngram_n=int(ngram_n), lidstone_gamma=float(lidstone_gamma),
    )
    ctxidx = build_real_context_index(vocab, cpd, tokens)

    cls = LockedIsomorphismPipeline if locked else IsomorphismPipeline
    kwargs: Dict[str, Any] = dict(
        cpd=cpd, context_index=ctxidx, vocab=vocab, ngram_n=int(ngram_n),
    )
    if pipelinekwargs:
        kwargs.update(pipelinekwargs)
    pipe = cls(**kwargs)
    pipe.preprocessor = pre
    return pipe, pre


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 0. SHARED TEXT UTILITIES
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "while", "of", "in", "on", "at", "to", "for", "with", "from", "by",
    "as", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "can", "i", "me", "my", "mine", "myself",
    "we", "us", "our", "ours", "you", "your", "yours", "he", "she", "it",
    "they", "them", "their", "this", "that", "these", "those", "what",
    "which", "who", "whom", "how", "why", "where", "there", "here", "so",
    "just", "very", "really", "some", "any", "all", "no", "not", "only",
    "now", "today", "tonight", "tomorrow", "yesterday", "again", "still",
    "also", "too", "more", "less", "much", "many", "few", "feel", "feeling",
    "think", "thinking", "thought", "want", "wanted", "wants", "need",
    "needs", "about", "into", "out", "up", "down", "over", "under",
    "through", "right", "kind", "sort", "way", "ways", "thing", "things",
    "stuff", "im", "ive", "id", "dont", "cant", "wont", "isnt",
    "thats", "whats", "lot", "lots", "bit", "going", "got", "get",
    "something", "anything", "everything", "nothing",
    "anyone", "everyone", "nobody", "anybody", "everybody",
    "good", "bad", "well", "better", "worse", "best", "worst",
    "make", "makes", "made", "making", "take", "takes", "took", "taking",
    "come", "comes", "came", "coming", "give", "gives", "gave", "giving",
})


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


def _stem(word: str) -> str:
    for suffix in ("ings", "ing", "edly", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _content_tokens(text: str) -> set[str]:
    return {_stem(w) for w in _tokens(text) if w not in _STOPWORDS and len(w) > 2}


def _wrap(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    if not words:
        return [""]
    out: list[str] = []
    line = words[0]
    for w in words[1:]:
        if len(line) + 1 + len(w) <= width:
            line = f"{line} {w}"
        else:
            out.append(line)
            line = w
    out.append(line)
    return out


def _strip_prompt_echo(prompt: str, text: str) -> str:
    p = " ".join(prompt.split()).strip()
    t = " ".join(text.split()).strip()
    if not p or not t:
        return text.strip()
    if t.startswith(p):
        return t[len(p):].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 1. GEOMETRY
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coord:
    agency: float
    scale: float
    register: float

    def dist(self, other: "Coord") -> float:
        return math.sqrt(
            (self.agency - other.agency) ** 2
            + (self.scale - other.scale) ** 2
            + (self.register - other.register) ** 2
        )


@dataclass(frozen=True)
class Activity:
    name: str
    coord: Coord
    prompt: str


_ACTIVITY_TAGS: dict[str, frozenset[str]] = {
    "waiting in line": frozenset({"waiting", "pause", "patience", "public", "boredom"}),
    "taking a shower": frozenset({"body", "private", "ritual", "morning", "reflection"}),
    "doing laundry": frozenset({"chore", "repetitive", "domestic", "routine", "alone"}),
    "eating lunch alone": frozenset({"meal", "alone", "lonely", "work", "solitude", "midday"}),
    "making coffee": frozenset({"morning", "ritual", "small", "routine", "domestic"}),
    "folding clothes": frozenset({"chore", "repetitive", "domestic", "routine", "quiet"}),
    "walking the dog": frozenset({"walk", "outside", "routine", "animal", "evening"}),
    "taking out the trash": frozenset({"chore", "domestic", "small", "routine"}),
    "watering the plants": frozenset({"care", "domestic", "small", "ritual", "attention"}),
    "tidying the desk": frozenset({"transition", "work", "clearing", "preparation"}),

    "meeting someone new": frozenset({"introduction", "stranger", "social", "anxious", "nervous", "worried", "first"}),
    "catching up with a friend": frozenset({"friend", "social", "conversation", "reunion", "care"}),
    "helping a stranger": frozenset({"help", "stranger", "kindness", "moral", "interruption"}),
    "receiving criticism": frozenset({"feedback", "hurt", "criticism", "conflict", "self", "stung", "argument"}),
    "giving a compliment": frozenset({"praise", "kindness", "social", "small", "generous"}),
    "apologising": frozenset({"apology", "hurt", "conflict", "fight", "argument", "repair", "partner", "wife", "husband", "boyfriend", "girlfriend", "relationship", "family", "mother", "father", "parent", "mom", "dad", "sister", "brother", "sorry", "regret"}),
    "being ignored": frozenset({"lonely", "rejected", "hurt", "invisible", "social", "sad"}),
    "forgiving someone": frozenset({"forgive", "hurt", "conflict", "fight", "argument", "repair", "anger", "angry", "resentment"}),
    "a difficult conversation": frozenset({"conflict", "fight", "argument", "argue", "hard", "talk", "partner", "wife", "husband", "boyfriend", "girlfriend", "relationship", "family", "mother", "father", "parent", "mom", "dad", "sister", "brother", "honest", "confrontation", "upset"}),
    "being cared for": frozenset({"care", "received", "sick", "loved", "vulnerable", "partner", "family", "mother", "father", "parent", "mom", "dad"}),

    "starting a new project": frozenset({"begin", "work", "project", "new", "anxious", "nervous", "excited"}),
    "facing a deadline": frozenset({"deadline", "stress", "stressed", "work", "pressure", "anxious", "worried", "overwhelmed", "rush"}),
    "procrastinating": frozenset({"avoid", "stuck", "overwhelmed", "anxious", "worried", "delay", "resistance", "lazy"}),
    "finishing a task": frozenset({"complete", "work", "done", "transition", "satisfaction", "happy"}),
    "feeling stuck": frozenset({"stuck", "frustrated", "blocked", "problem", "overwhelmed", "spinning", "lost"}),
    "learning something new": frozenset({"learn", "study", "skill", "beginner", "confused", "growth"}),
    "making a mistake at work": frozenset({"mistake", "error", "shame", "work", "hurt", "self", "embarrassed", "boss", "manager", "colleague"}),
    "getting a promotion": frozenset({"promotion", "success", "work", "milestone", "celebrate", "happy", "boss", "manager"}),
    "being bored": frozenset({"bored", "empty", "restless", "unstimulated", "stuck"}),
    "resigning": frozenset({"quit", "leave", "work", "transition", "milestone", "decision", "boss", "manager"}),

    "going for a run": frozenset({"run", "exercise", "body", "outside", "movement", "morning"}),
    "sitting with pain": frozenset({"pain", "body", "hurt", "suffering", "endure", "physical", "sick", "ache"}),
    "a medical appointment": frozenset({"medical", "doctor", "health", "waiting", "anxious", "nervous", "worried", "scared", "body", "sick"}),
    "preparing to sleep but can't": frozenset({"sleep", "insomnia", "night", "anxious", "worried", "tired", "exhausted", "mind", "racing"}),
    "eating mindlessly": frozenset({"eat", "mindless", "distracted", "habit", "body"}),
    "meditating": frozenset({"meditate", "quiet", "attention", "stillness", "practice", "mind", "calm"}),
    "a long walk alone": frozenset({"walk", "alone", "solitary", "outside", "thinking", "wander"}),
    "recovering from illness": frozenset({"sick", "rest", "recovery", "body", "tired", "exhausted", "weak", "ill"}),
    "getting a haircut": frozenset({"waiting", "appearance", "small", "service", "passive"}),

    "a birthday": frozenset({"birthday", "milestone", "year", "time", "self", "celebrate", "happy"}),
    "moving house": frozenset({"move", "change", "transition", "home", "leaving", "milestone"}),
    "ending a friendship": frozenset({"friend", "ending", "loss", "drift", "grief", "sad", "relationship"}),
    "looking at old photos": frozenset({"memory", "past", "nostalgia", "photos", "grief", "miss", "missing", "time"}),
    "receiving bad news": frozenset({"shock", "grief", "loss", "hard", "overwhelmed", "sad", "depressed", "scared"}),
    "celebrating a milestone": frozenset({"celebrate", "milestone", "success", "joy", "happy", "achievement"}),
    "being stuck in traffic": frozenset({"traffic", "late", "stuck", "transit", "frustrated", "angry", "waiting"}),
    "watching the sunset": frozenset({"beauty", "nature", "evening", "still", "noticing", "small", "peaceful"}),
    "thinking about death": frozenset({"death", "mortality", "fear", "scared", "meaning", "existential", "grief", "dying"}),
    "feeling grateful": frozenset({"gratitude", "thankful", "appreciation", "joy", "happy", "noticing"}),
    "a disagreement online": frozenset({"argument", "argue", "fight", "conflict", "online", "stranger", "anger", "angry", "frustrated"}),
}

ACTIVITIES: list[Activity] = [
    Activity("waiting in line", Coord(-0.3, -0.6, -1.0), "I'm standing in a long queue. What might I do with this unscheduled pause?"),
    Activity("taking a shower", Coord(+0.4, -0.2, -0.5), "I'm in the shower — one of the few truly private moments. What's worth thinking about here?"),
    Activity("doing laundry", Coord(-0.1, -0.7, -0.8), "I'm loading the washing machine. How might I frame this repetitive task?"),
    Activity("eating lunch alone", Coord(+0.1, -0.6, -0.5), "I'm eating lunch by myself today. What's worth attending to during a solo meal?"),
    Activity("making coffee", Coord(+0.5, -0.1, -0.3), "I'm making my morning coffee. What's worth noticing in this small ritual?"),
    Activity("folding clothes", Coord(-0.2, -0.8, -0.7), "I'm folding a pile of laundry. How might I inhabit this quiet, repetitive work?"),
    Activity("walking the dog", Coord(+0.7, +0.2, +0.0), "I'm on the evening walk with my dog. What's worth attending to on this regular loop?"),
    Activity("taking out the trash", Coord(-0.4, -0.7, -0.8), "I'm taking out the bins. Is there anything worth noticing in this small domestic act?"),
    Activity("watering the plants", Coord(+0.5, -0.3, +0.0), "I'm watering the houseplants. What might I pay attention to right now?"),
    Activity("tidying the desk", Coord(+0.2, -0.4, -0.2), "I'm clearing off my desk before starting work. How should I approach this transition?"),

    Activity("meeting someone new", Coord(+0.3, +0.6, -0.3), "I'm about to be introduced to someone I don't know. How should I show up?"),
    Activity("catching up with a friend", Coord(+0.8, +0.4, +0.2), "I'm meeting a friend I haven't seen in months for coffee. What might I bring to the conversation?"),
    Activity("helping a stranger", Coord(+0.7, +0.3, +0.5), "I just stopped to help someone who seemed lost or struggling. How should I think about this moment?"),
    Activity("receiving criticism", Coord(-0.2, +0.5, -0.6), "Someone just gave me feedback that stung a little. How do I sit with this?"),
    Activity("giving a compliment", Coord(+0.6, +0.2, +0.3), "I'm about to tell someone something genuinely good about them. What makes this worth doing well?"),
    Activity("apologising", Coord(+0.1, +0.3, -0.7), "I'm about to apologise to someone I hurt or let down. How should I approach this?"),
    Activity("being ignored", Coord(-0.6, -0.1, -0.8), "I feel like I'm being overlooked right now. How should I hold this feeling?"),
    Activity("forgiving someone", Coord(+0.3, -0.2, +0.3), "I'm working through forgiving someone who wronged me. What does that actually require?"),
    Activity("a difficult conversation", Coord(-0.1, +0.6, -0.2), "I need to have a hard conversation with someone today. How do I prepare for it?"),
    Activity("being cared for", Coord(+0.7, -0.1, -0.5), "Someone is taking care of me right now — cooking, checking in, helping out. What do I notice?"),

    Activity("starting a new project", Coord(+0.6, +0.7, +0.3), "I'm beginning a new project I've been planning for a while. How do I start well?"),
    Activity("facing a deadline", Coord(-0.2, +0.8, -0.3), "A deadline is approaching fast. How do I think about the next few hours?"),
    Activity("procrastinating", Coord(-0.5, -0.4, -0.6), "I keep putting off something I know I need to do. What's actually happening here?"),
    Activity("finishing a task", Coord(+0.7, +0.3, +0.4), "I just completed something I've been working on. How do I close it properly?"),
    Activity("feeling stuck", Coord(-0.4, +0.2, -0.7), "I've been staring at the same problem for an hour and going nowhere. What now?"),
    Activity("learning something new", Coord(+0.5, +0.5, -0.2), "I'm in the middle of learning a new skill or concept I don't fully understand yet. How do I stay with it?"),
    Activity("making a mistake at work", Coord(-0.6, +0.5, -0.7), "I just made an error that matters. How should I respond to myself and the situation?"),
    Activity("getting a promotion", Coord(+0.9, +0.8, +0.7), "I just found out I've been promoted. How do I receive this news well?"),
    Activity("being bored", Coord(-0.3, -0.9, -0.8), "Nothing is holding my attention. I'm genuinely bored. What should I do with that?"),
    Activity("resigning", Coord(+0.2, +0.6, +0.2), "I'm about to hand in my resignation. What's worth reflecting on before I do?"),

    Activity("going for a run", Coord(+0.6, +0.8, +0.3), "I'm heading out for a run. How might I use this time beyond just the exercise?"),
    Activity("sitting with pain", Coord(-0.5, +0.3, -0.6), "I'm dealing with physical pain right now. How do I be with it rather than just against it?"),
    Activity("a medical appointment", Coord(-0.1, +0.5, -0.5), "I'm sitting in a waiting room before seeing a doctor. How do I orientate to what's coming?"),
    Activity("preparing to sleep but can't", Coord(-0.4, +0.3, -0.7), "I'm lying in bed but my mind won't settle. What might help me find rest?"),
    Activity("eating mindlessly", Coord(-0.3, -0.5, -0.7), "I realise I've been eating without really noticing. What might I return to?"),
    Activity("meditating", Coord(+0.5, -0.8, +0.2), "I've just sat down to meditate. How do I actually begin?"),
    Activity("a long walk alone", Coord(+0.6, -0.1, +0.3), "I'm taking a long solitary walk with no destination in mind. What might I think about?"),
    Activity("recovering from illness", Coord(-0.2, -0.5, -0.6), "I'm home sick and resting. How do I spend this unexpected stillness?"),
    Activity("getting a haircut", Coord(+0.2, -0.3, -0.5), "I'm sitting in the barber's chair with nothing to do but wait. What's worth attending to?"),

    Activity("a birthday", Coord(+0.6, +0.5, +0.2), "It's my birthday today. How do I think about what this marker means?"),
    Activity("moving house", Coord(+0.1, +0.7, +0.0), "I'm packing up my home to move somewhere new. What do I want to carry forward — and leave behind?"),
    Activity("ending a friendship", Coord(-0.3, +0.2, -0.3), "A friendship seems to be ending, not through conflict but through drift. How do I sit with this?"),
    Activity("looking at old photos", Coord(+0.3, -0.1, -0.3), "I've been looking through old photos. What do I want to notice, and what do I want to do with these feelings?"),
    Activity("receiving bad news", Coord(-0.8, +0.6, -0.8), "I've just received news that is genuinely bad. How do I begin to hold this?"),
    Activity("celebrating a milestone", Coord(+0.9, +0.7, +0.6), "Something significant I worked toward has happened. How do I actually let myself celebrate?"),
    Activity("being stuck in traffic", Coord(-0.4, +0.2, -0.9), "I'm stuck in traffic and running late. How do I use this time without spiralling?"),
    Activity("watching the sunset", Coord(+0.7, -0.3, +0.1), "I stopped to watch the sun go down. What's worth noticing in this ordinary miracle?"),
    Activity("thinking about death", Coord(-0.1, +0.1, +0.4), "My own mortality has come into focus today. How do I think clearly about death?"),
    Activity("feeling grateful", Coord(+0.9, +0.1, +0.3), "I feel a quiet, genuine gratitude right now. How do I honour and deepen that feeling?"),
    Activity("a disagreement online", Coord(-0.5, +0.7, -0.3), "I've just gotten into an argument with a stranger online. What's worth examining here?"),
]


class PromptGeometry:
    def __init__(self, activities: list[Activity] | None = None):
        self.activities = activities or list(ACTIVITIES)

    def all(self) -> list[Activity]:
        return list(self.activities)

    def sample(self, n: int, seed: int = 0) -> list[Activity]:
        rng = random.Random(seed)
        return rng.sample(self.activities, min(n, len(self.activities)))

    def near(self, center: Coord, radius: float) -> list[Activity]:
        return [a for a in self.activities if a.coord.dist(center) <= radius]

    def quadrant(self, **bounds: tuple[float, float]) -> list[Activity]:
        out = []
        for a in self.activities:
            ok = True
            for axis, (lo, hi) in bounds.items():
                if not lo <= getattr(a.coord, axis) <= hi:
                    ok = False
                    break
            if ok:
                out.append(a)
        return out

    def select_by_query(self, query: str, n: int) -> list[tuple[Activity, float]]:
        q = _content_tokens(query)
        if not q:
            return self._seeded_random(query, n)

        scored: list[tuple[Activity, float]] = []
        for a in self.activities:
            name_t = _content_tokens(a.name)
            prompt_t = _content_tokens(a.prompt)
            tag_t = {_stem(t) for t in _ACTIVITY_TAGS.get(a.name, frozenset())}
            score = (
                2.0 * len(q & name_t)
                + 1.0 * len(q & prompt_t)
                + 3.0 * len(q & tag_t)
            )
            scored.append((a, score))

        scored.sort(key=lambda kv: kv[1], reverse=True)
        if scored[0][1] == 0.0:
            return self._seeded_random(query, n)
        return scored[:n]

    def _seeded_random(self, query: str, n: int) -> list[tuple[Activity, float]]:
        rng = random.Random(hash(query) & 0xFFFFFFFF)
        picks = rng.sample(self.activities, min(n, len(self.activities)))
        return [(a, 0.0) for a in picks]


# ---------------------------------------------------------------------------
# 2. PARADIGMS
# ---------------------------------------------------------------------------

@dataclass
class Paradigm:
    name: str
    system_prompt: str
    lexicon: set[str]
    forbidden: set[str]
    min_lexicon_hits: int
    max_imperatives: int
    min_length_chars: int
    max_length_chars: int


PARADIGMS: dict[str, Paradigm] = {
    "communitarian": Paradigm(
        name="communitarian",
        system_prompt=(
            "You answer from a communitarian ethics. People are constituted by "
            "their relationships and obligations, not by atomized preferences. "
            "Foreground who else is affected, what is owed, what is shared. "
            "Speak plainly. Do not moralize."
        ),
        lexicon={"we", "us", "our", "neighbor", "neighbour", "share", "shared",
                 "together", "community", "owe", "obligation", "kin", "common",
                 "belong", "between", "with"},
        forbidden={"optimize", "personal brand", "maximize utility",
                   "self-actualize", "self actualize"},
        min_lexicon_hits=3, max_imperatives=2,
        min_length_chars=150, max_length_chars=900,
    ),
    "liberal": Paradigm(
        name="liberal",
        system_prompt=(
            "You answer from a liberal-individualist stance. Foreground the "
            "person's autonomy, preferences, and right to define their own "
            "good. Offer options, not orders. Trust them to decide."
        ),
        lexicon={"you", "your", "choice", "prefer", "decide", "option", "right",
                 "consent", "autonomy", "freedom", "could", "might"},
        forbidden={"must", "duty", "owe", "the community demands"},
        min_lexicon_hits=4, max_imperatives=1,
        min_length_chars=120, max_length_chars=800,
    ),
    "care": Paradigm(
        name="care",
        system_prompt=(
            "You answer from an ethics of care. Attend to the particular "
            "person and the particular other in front of them. Avoid "
            "abstract principles; ask what this person, in this moment, "
            "with these relationships, actually needs to attend to. Be warm "
            "and specific."
        ),
        lexicon={"attend", "attention", "notice", "listen", "tender",
                 "gentle", "they", "them", "particular", "specific", "care",
                 "feel", "feeling", "warmth", "soft"},
        forbidden={"in general", "universally", "always", "everyone should"},
        min_lexicon_hits=4, max_imperatives=2,
        min_length_chars=150, max_length_chars=900,
    ),
    "stoic": Paradigm(
        name="stoic",
        system_prompt=(
            "You answer in the voice of a Stoic. Distinguish what is up to "
            "the person from what is not. Direct attention to virtue, "
            "equanimity, and what can be done well in this moment. Be terse. "
            "No flourish."
        ),
        lexicon={"control", "up to you", "within", "virtue", "equanimity",
                 "accept", "attention", "present", "this moment", "discipline",
                 "what you can", "is not yours"},
        forbidden={"deserve", "unfair", "entitled", "the universe owes"},
        min_lexicon_hits=3, max_imperatives=3,
        min_length_chars=80, max_length_chars=600,
    ),
    "materialist": Paradigm(
        name="materialist",
        system_prompt=(
            "You answer from a materialist standpoint. Surface the material "
            "and labor relations behind the act: who made what possible, "
            "whose work is invisible, what conditions produced this moment. "
            "Do not lecture. Make the conditions visible."
        ),
        lexicon={"labor", "labour", "work", "made", "produced", "wage",
                 "worker", "infrastructure", "supply", "cost", "extracted",
                 "behind", "conditions", "who"},
        forbidden={"manifest", "abundance", "vibration", "energies",
                   "the universe provides"},
        min_lexicon_hits=3, max_imperatives=2,
        min_length_chars=150, max_length_chars=900,
    ),
    "anarchist": Paradigm(
        name="anarchist",
        system_prompt=(
            "You answer in an anarchist register: voluntary association, "
            "mutual aid, suspicion of hierarchy and coerced participation. "
            "Where rules are doing work, name them. Where consent is doing "
            "work, name that too. Do not romanticize."
        ),
        lexicon={"mutual", "aid", "consent", "voluntary", "free", "horizontal",
                 "refuse", "withdraw", "together", "without", "hierarchy",
                 "permission", "ask"},
        forbidden={"comply", "must obey", "authority requires",
                   "the rules are the rules"},
        min_lexicon_hits=3, max_imperatives=2,
        min_length_chars=120, max_length_chars=800,
    ),
}


# ---------------------------------------------------------------------------
# 3. SIMULATOR — direct Pi pipeline import/use
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    paradigm: str
    activity: str
    coord: Coord
    prompt: str
    text: str
    n_input_tokens: int
    n_output_tokens: int
    latency_s: float
    tokens_per_sec: float
    peak_mem_mb: float


class Simulator:
    def __init__(
        self,
        mock: bool = False,
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        *,
        pi_preset: str = "wikitext",
        pi_dataset: str | None = None,
        pi_config: str | None = None,
        pi_text_fields: list[str] | None = None,
        pi_splits: list[str] | None = None,
        pi_locked: bool = True,
        pi_ngram_n: int = 3,
        pi_min_sentence_len: int = 3,
    ):
        self.mock = mock
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.pi_preset = pi_preset
        self.pi_dataset = pi_dataset
        self.pi_config = pi_config
        self.pi_text_fields = pi_text_fields or []
        self.pi_splits = pi_splits or []
        self.pi_locked = pi_locked
        self.pi_ngram_n = int(pi_ngram_n)
        self.pi_min_sentence_len = int(pi_min_sentence_len)

        self._pipeline = None
        self._preprocessor = None
        self._generator = None

    def _ensure_model(self) -> None:
        if self.mock or self._generator is not None:
            return

        if self.pi_dataset:
            self._pipeline, self._preprocessor = build_hf_pipeline(
                dataset_name=self.pi_dataset,
                config_name=self.pi_config,
                text_fields=self.pi_text_fields or None,
                split_names=tuple(self.pi_splits) if self.pi_splits else None,
                locked=self.pi_locked,
                ngram_n=self.pi_ngram_n,
                minsentencelen=self.pi_min_sentence_len,
            )
        else:
            self._pipeline, self._preprocessor = build_hf_pipeline_from_preset(
                preset_name=self.pi_preset,
                locked=self.pi_locked,
                ngram_n=self.pi_ngram_n,
                minsentencelen=self.pi_min_sentence_len,
            )

        self._generator = SentenceAwareGenerator(
            self._pipeline,
            self._preprocessor,
        )

    def run(self, paradigm: Paradigm, activity: Activity) -> GenerationResult:
        self._ensure_model()
        if self.mock:
            return self._mock_run(paradigm, activity)
        return self._real_run(paradigm, activity)

    def _real_run(self, paradigm: Paradigm, activity: Activity) -> GenerationResult:
        prompt = f"{paradigm.system_prompt}\n\n{activity.prompt}"

        peak_before = 0.0
        peak_after = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                peak_before = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            pass

        t0 = time.perf_counter()
        raw_text = self._generator.generate_text(
            prompt=prompt,
            n_words=self.max_new_tokens,
            seed=42,
            capitalise=True,
        )
        elapsed = time.perf_counter() - t0

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak_after = torch.cuda.max_memory_allocated() / 1024**2
        except Exception:
            peak_after = 0.0

        text = _strip_prompt_echo(prompt, raw_text)
        n_in = len(prompt.split())
        n_out = len(text.split())
        peak_mb = max(peak_before, peak_after)

        return GenerationResult(
            paradigm=paradigm.name,
            activity=activity.name,
            coord=activity.coord,
            prompt=activity.prompt,
            text=text,
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            latency_s=elapsed,
            tokens_per_sec=(n_out / elapsed) if elapsed > 0 else 0.0,
            peak_mem_mb=peak_mb,
        )

    def _mock_run(self, paradigm: Paradigm, activity: Activity) -> GenerationResult:
        rng = random.Random(hash((paradigm.name, activity.name)) & 0xFFFFFFFF)
        lex = [w for w in paradigm.lexicon if " " not in w]
        rng.shuffle(lex)
        picks = lex[:6] if len(lex) >= 6 else lex
        if len(picks) < 2:
            picks = picks + ["attention", "care", "choice", "work", "present", "together"]

        sentences = [
            f"Before {activity.name.lower()}, pause and notice what is actually present.",
            f"Part of this is about you, and part of it is about {picks[0]} and {picks[1]}.",
            f"Hold the act lightly: what you {picks[2]} matters less than what you {picks[3]}.",
            f"Whatever is {picks[4]} here is not separate from the rest of your day.",
            f"This is {picks[5] if len(picks) > 5 else picks[0]} territory; treat it that way.",
        ]
        text = " ".join(sentences)
        time.sleep(0.02)
        n_out = len(text.split())
        elapsed = 0.3 + rng.random() * 0.4

        return GenerationResult(
            paradigm=paradigm.name,
            activity=activity.name,
            coord=activity.coord,
            prompt=activity.prompt,
            text=text,
            n_input_tokens=80,
            n_output_tokens=n_out,
            latency_s=elapsed,
            tokens_per_sec=n_out / elapsed,
            peak_mem_mb=0.0,
        )


# ---------------------------------------------------------------------------
# 4. COMPARATOR
# ---------------------------------------------------------------------------

_IMPERATIVE_VERBS = {
    "do", "don't", "stop", "start", "go", "be", "consider", "remember",
    "notice", "ask", "try", "make", "let", "think", "take", "give",
    "hold", "listen", "attend", "accept", "refuse",
}


def _count_imperatives(text: str) -> int:
    n = 0
    for s in _sentences(text):
        words = _WORD_RE.findall(s)
        if words and words[0].lower() in _IMPERATIVE_VERBS:
            n += 1
    return n


def _lexical_diversity(words: list[str]) -> float:
    return len(set(words)) / len(words) if words else 0.0


def _specificity(text: str) -> float:
    proper = 0
    for s in _sentences(text):
        for i, w in enumerate(_WORD_RE.findall(s)):
            if i > 0 and w[:1].isupper():
                proper += 1
    numeric = sum(1 for c in text if c.isdigit())
    return min(1.0, (proper + 0.5 * numeric) / 8.0)


@dataclass
class QualityScore:
    diversity: float
    specificity: float
    lexicon_hits: int
    forbidden_hits: int
    imperative_count: int
    length_chars: int
    in_length_bounds: bool
    lexicon_pass: bool
    forbidden_pass: bool
    imperative_pass: bool
    overall: float
    enforced: bool
    reasons: list[str] = field(default_factory=list)


class Comparator:
    def score(self, result: GenerationResult, paradigm: Paradigm) -> QualityScore:
        text = result.text
        text_low = text.lower()
        words = _tokens(text)

        single_lex = {x.lower() for x in paradigm.lexicon if " " not in x}
        multi_lex = [x.lower() for x in paradigm.lexicon if " " in x]
        lex_hits = sum(1 for w in words if w in single_lex)
        for phrase in multi_lex:
            lex_hits += text_low.count(phrase)

        forb_hits = sum(text_low.count(p.lower()) for p in paradigm.forbidden)

        diversity = _lexical_diversity(words)
        specificity = _specificity(text)
        n_imp = _count_imperatives(text)
        length = len(text)

        in_bounds = paradigm.min_length_chars <= length <= paradigm.max_length_chars
        lex_pass = lex_hits >= paradigm.min_lexicon_hits
        forb_pass = forb_hits == 0
        imp_pass = n_imp <= paradigm.max_imperatives

        reasons: list[str] = []
        if not in_bounds:
            reasons.append(f"length {length} outside [{paradigm.min_length_chars}, {paradigm.max_length_chars}]")
        if not lex_pass:
            reasons.append(f"only {lex_hits}/{paradigm.min_lexicon_hits} required paradigm-keywords")
        if not forb_pass:
            reasons.append(f"{forb_hits} forbidden tokens/phrases")
        if not imp_pass:
            reasons.append(f"{n_imp} imperatives > max {paradigm.max_imperatives}")

        enforced = in_bounds and lex_pass and forb_pass and imp_pass

        overall = (
            0.4 * (lex_hits / max(paradigm.min_lexicon_hits, 1))
            + 0.3 * diversity
            + 0.2 * specificity
            + 0.1 * (0.0 if forb_hits else 1.0)
        )

        return QualityScore(
            diversity=round(diversity, 3),
            specificity=round(specificity, 3),
            lexicon_hits=lex_hits,
            forbidden_hits=forb_hits,
            imperative_count=n_imp,
            length_chars=length,
            in_length_bounds=in_bounds,
            lexicon_pass=lex_pass,
            forbidden_pass=forb_pass,
            imperative_pass=imp_pass,
            overall=round(overall, 3),
            enforced=enforced,
            reasons=reasons,
        )

    @staticmethod
    def report(
        scored: list[tuple[GenerationResult, Paradigm, QualityScore]],
    ) -> str:
        lines: list[str] = []
        by_p: dict[str, list[tuple[GenerationResult, QualityScore]]] = {}
        for r, p, s in scored:
            by_p.setdefault(p.name, []).append((r, s))

        lines.append("=" * 78)
        lines.append("EVERYDAY GEOMETRY — generation report")
        lines.append("=" * 78)
        for pname, items in by_p.items():
            lines.append("")
            lines.append(f"PARADIGM: {pname}")
            lines.append("-" * 78)
            passes = sum(1 for _, s in items if s.enforced)
            lat = statistics.mean(r.latency_s for r, _ in items)
            tps = statistics.mean(r.tokens_per_sec for r, _ in items)
            ov = statistics.mean(s.overall for _, s in items)
            lines.append(
                f"  enforcement: {passes}/{len(items)} pass | "
                f"mean latency {lat:.2f}s | tok/s {tps:.1f} | "
                f"mean overall {ov:.3f}"
            )
            for r, s in items:
                badge = "✓" if s.enforced else "✗"
                lines.append(
                    f"  {badge} {r.activity:<28} "
                    f"L={s.length_chars:>4}  lex={s.lexicon_hits}  "
                    f"imp={s.imperative_count}  ov={s.overall:.2f}"
                )
                for why in s.reasons:
                    lines.append(f"      ! {why}")

        by_act: dict[str, list[tuple[str, GenerationResult, QualityScore]]] = {}
        for r, p, s in scored:
            by_act.setdefault(r.activity, []).append((p.name, r, s))
        cross = [(a, rows) for a, rows in by_act.items() if len(rows) >= 2]
        if cross:
            lines.append("")
            lines.append("=" * 78)
            lines.append("CROSS-PARADIGM COMPARISON (best alignment per activity)")
            lines.append("=" * 78)
            for act, rows in cross:
                best_name, best_r, best_s = max(rows, key=lambda kv: kv[2].overall)
                summary = "  ".join(f"{n}={s.overall:.2f}" for n, _, s in rows)
                lines.append("")
                lines.append(f"  {act}")
                lines.append(f"  winner: {best_name}  ({summary})")
                if not best_s.enforced:
                    lines.append("  (note: winner did not pass enforcement; best of a failing field)")
                lines.append("  " + "─" * 74)
                for para in best_r.text.split("\n"):
                    for chunk in _wrap(para, 74):
                        lines.append(f"  {chunk}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def _resolve_query(args: argparse.Namespace) -> str | None:
    if args.random:
        return None
    if args.query is not None:
        return args.query.strip() or None
    try:
        print("describe what's on your mind (or press Enter for a random sample):")
        q = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return q or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", "-q", help="user prompt to drive activity selection; if omitted, asks interactively")
    ap.add_argument("--random", action="store_true", help="ignore --query and pick activities at random")
    ap.add_argument("--paradigm", action="append", help="paradigm name(s); repeatable. default: all")
    ap.add_argument("--n", type=int, default=4, help="how many activities to use (default 4)")
    ap.add_argument("--seed", type=int, default=0, help="seed for --random sampling")
    ap.add_argument("--mock", action="store_true", help="skip model load; emit canned paradigm text")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--show-text", action="store_true", help="print full generated text per item")
    ap.add_argument("--json", type=Path, help="optional: write full results to JSON")

    ap.add_argument("--pi-preset", default="wikitext", help="Pi preset name for build_hf_pipeline_from_preset")
    ap.add_argument("--pi-dataset", help="override preset and build from a specific HF dataset instead")
    ap.add_argument("--pi-config", help="dataset config name for --pi-dataset")
    ap.add_argument("--pi-text-field", action="append", help="repeatable text field name/path for --pi-dataset")
    ap.add_argument("--pi-split", action="append", help="repeatable dataset split for --pi-dataset")
    ap.add_argument("--pi-unlocked", action="store_true", help="use unlocked pipeline instead of locked pipeline")
    ap.add_argument("--pi-ngram_n", type=int, default=3, help="n-gram order passed to the Pi builder")
    ap.add_argument("--pi-min-sentence-len", type=int, default=3, help="minimum entity length for the Pi dataset preprocessor")

    args = ap.parse_args()

    paradigm_names = args.paradigm or list(PARADIGMS.keys())
    for n in paradigm_names:
        if n not in PARADIGMS:
            print(f"unknown paradigm: {n}; known: {list(PARADIGMS)}", file=sys.stderr)
            sys.exit(2)

    geom = PromptGeometry()

    query = _resolve_query(args)
    if query is None:
        matched = [(a, 0.0) for a in geom.sample(args.n, seed=args.seed)]
        selection_mode = "random"
    else:
        matched = geom.select_by_query(query, args.n)
        selection_mode = "query"

    activities = [a for a, _ in matched]

    if selection_mode == "query":
        print(f'\nquery: "{query}"')
        print("matched activities:")
        for a, score in matched:
            label = f"(relevance {score:.1f})" if score > 0 else "(no overlap; random fallback)"
            print(f"  • {a.name:<32} {label}")
    else:
        print(f"\nrandom sample of {len(activities)} activity(ies):")
        for a, _ in matched:
            print(f"  • {a.name}")

    print(
        f"\nrunning {len(paradigm_names)} paradigm(s) × {len(activities)} activity(ies)"
        f"{' (mock mode)' if args.mock else ''}\n"
    )

    sim = Simulator(
        mock=args.mock,
        max_new_tokens=args.max_new_tokens,
        pi_preset=args.pi_preset,
        pi_dataset=args.pi_dataset,
        pi_config=args.pi_config,
        pi_text_fields=args.pi_text_field or [],
        pi_splits=args.pi_split or [],
        pi_locked=not args.pi_unlocked,
        pi_ngram_n=args.pi_ngram_n,
        pi_min_sentence_len=args.pi_min_sentence_len,
    )
    cmp_ = Comparator()
    scored: list[tuple[GenerationResult, Paradigm, QualityScore]] = []

    for pname in paradigm_names:
        p = PARADIGMS[pname]
        for act in activities:
            r = sim.run(p, act)
            s = cmp_.score(r, p)
            scored.append((r, p, s))
            if args.show_text:
                print(
                    f"\n--- {pname} | {act.name} "
                    f"@ (a={act.coord.agency:+.1f}, s={act.coord.scale:+.1f}, "
                    f"r={act.coord.register:+.1f}) ---"
                )
                print(r.text)

    print()
    print(cmp_.report(scored))

    if args.json:
        payload = {
            "query": query,
            "selection_mode": selection_mode,
            "results": [
                {"generation": asdict(r), "paradigm": p.name, "score": asdict(s)}
                for r, p, s in scored
            ],
        }
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
