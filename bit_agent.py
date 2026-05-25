"""
Run Microsoft's pretrained BitNet b1.58 2B4T (April 2025).

Setup
-----
    pip install -U "transformers>=4.52" torch accelerate

Notes
-----
- For real 1-bit speedups on CPU, use the official bitnet.cpp kernels.
  Loading through transformers runs in BF16 (dequantized at load time) and
  is the easiest way to use the model from Python.
- VRAM/RAM footprint is roughly ~5 GB BF16 / ~1.2 GB with bitnet.cpp packing.
- The model card lives at https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "microsoft/bitnet-b1.58-2B-4T"


def load(device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def chat(model, tokenizer, messages, max_new_tokens: int = 256, temperature: float = 0.7):
    """messages: list of {'role': 'system'|'user'|'assistant', 'content': str}."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    # only return the newly generated tokens
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def single_turn(prompt: str, system: str | None = None, **kw) -> str:
    model, tokenizer, _ = load()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return chat(model, tokenizer, messages, **kw)


def repl():
    """Interactive chat loop with conversation history."""
    model, tokenizer, device = load()
    print(f"BitNet b1.58 2B4T loaded on {device}. type 'exit' to quit, 'reset' to clear history.\n")

    history = [{"role": "system", "content": "You are a helpful assistant."}]
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "reset":
            history = history[:1]
            print("(history cleared)\n")
            continue

        history.append({"role": "user", "content": user})
        reply = chat(model, tokenizer, history)
        history.append({"role": "assistant", "content": reply})
        print(f"bot> {reply}\n")


"""
everyday.py — a conceptual geometry of everyday-life prompts, a
performance simulator for the generations across them, a qualitative
comparator with hard enforcement, and a set of philosophical/political
paradigms that govern the whole apparatus.

Design note (read first)
------------------------
The four pieces here are not independent. The GEOMETRY decides *which*
everyday acts count as legible. The PARADIGMS decide *how* they get
framed. The SIMULATOR measures *what happened in matter* (latency,
tokens, memory). The COMPARATOR decides *what counts as a good answer*.
Each layer is a political choice and this file is honest about that —
see the AXES note and the PARADIGMS table.

"Qualitative enforcement" here is HARD: a generation that fails a
paradigm's rubric is rejected, not softly downweighted. Strict
paradigms will have higher rejection rates. That is the point.

Hooks the model
---------------
Imports load() from x.py (your BitNet pretrained loader). Pass --mock
to run the whole framework with canned, paradigm-flavored outputs and
no model load — useful for demoing on a laptop.

Examples
--------
    python everyday.py --mock --n 4
    python everyday.py --mock --paradigm stoic --paradigm care --show-text
    python everyday.py --paradigm communitarian --n 3       # real BitNet
"""


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
# 1. GEOMETRY — a coordinate space of everyday acts
# ---------------------------------------------------------------------------
#
# AXES
#   agency    (-1 passive .. +1 initiating): does the actor receive the
#             world or shape it? "being commuted" vs "hosting".
#   scale     (-1 solitary .. +1 communal): how many others must be
#             co-present for the act to be itself?
#   register  (-1 mundane .. +1 ritual): how much does the act stand
#             apart from the flow of time?
#
# Choosing these three over (productive/unproductive, paid/unpaid,
# embodied/mediated) is already a politics. Productivity-centric axes
# would erase rest, mourning, and play. Embodiment-centric axes would
# flatten difference between solitary and communal acts. The chosen
# axes foreground *what kind of subject the actor is being asked to be*.

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


ACTIVITIES: list[Activity] = [
    Activity("brushing teeth",     Coord(-0.2, -1.0, -1.0),
             "I'm about to brush my teeth before bed. Is there anything worth thinking about while I do it?"),
    Activity("morning commute",    Coord(-0.5, -0.3, -1.0),
             "I'm on my morning commute on a crowded train. How should I orient to this stretch of the day?"),
    Activity("cooking dinner",     Coord(+0.5, +0.5, -0.5),
             "I'm cooking dinner for my partner and a friend. What's worth attending to as I cook?"),
    Activity("paying bills",       Coord(+0.0, -0.5, -1.0),
             "I sat down to pay this month's bills online. What should I think about as I do this?"),
    Activity("writing in journal", Coord(+0.8, -1.0, +0.0),
             "I'm opening my journal for tonight's entry. What might be worth writing about?"),
    Activity("arguing with partner", Coord(+0.1, +0.0, +0.3),
             "My partner and I just started arguing about household chores. How should I approach this?"),
    Activity("hosting dinner",     Coord(+1.0, +0.8, +0.5),
             "I'm hosting friends for dinner tonight. What's worth thinking about as I prepare?"),
    Activity("voting",             Coord(+0.7, +1.0, +0.5),
             "I'm walking to my polling place to vote. How should I think about this act?"),
    Activity("attending funeral",  Coord(-0.3, +1.0, +1.0),
             "I'm getting ready to attend a funeral this afternoon. How should I orient to the day?"),
    Activity("sleeping",           Coord(-1.0, -0.5, -1.0),
             "I'm getting into bed for the night. Is there anything worth attending to before I sleep?"),
    Activity("scrolling phone",    Coord(-0.7, -0.8, -1.0),
             "I'm in bed scrolling on my phone. How should I think about what I'm doing?"),
    Activity("calling parent",     Coord(+0.4, +0.3, -0.3),
             "I'm about to call my elderly parent for our weekly chat. What might I bring to the conversation?"),
]


class PromptGeometry:
    """A finite point cloud in (agency, scale, register) space."""

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
        """e.g. quadrant(scale=(0.0, 1.0)) returns communal acts only."""
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


# ---------------------------------------------------------------------------
# 2. PARADIGMS — philosophical/political alignment of the instruction
# ---------------------------------------------------------------------------
#
# Each paradigm is a triple: a stance (system prompt that conditions the
# model), a lexicon (the vocabulary that *would* show up if the stance
# is being honored), and an enforcement rubric (what we will reject).
#
# The rubric is the political teeth. A paradigm that demands solidarity
# language but never measures whether the model produced any is doing
# the soft thing. Hard enforcement = the paradigm has to actually win.

@dataclass
class Paradigm:
    name: str
    system_prompt: str
    lexicon: set[str]        # vocabulary characteristic of the stance
    forbidden: set[str]      # vocabulary that contradicts the stance
    min_lexicon_hits: int    # enforcement floor
    max_imperatives: int     # how preachy is too preachy
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
# 3. SIMULATOR — paradigm-conditioned generation with performance capture
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
    """Runs paradigm-conditioned generations and captures performance.

    Modes:
      - real:  loads BitNet via x.py and generates
      - mock:  emits canned paradigm-flavored text instantly
    """

    def __init__(self, mock: bool = False, max_new_tokens: int = 200,
                 temperature: float = 0.7):
        self.mock = mock
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self.mock or self._model is not None:
            return
        try:
            from x import load            # the file the user uploaded
        except ImportError:
            from bitnet_pretrained import load
        self._model, self._tokenizer, _ = load()

    def run(self, paradigm: Paradigm, activity: Activity) -> GenerationResult:
        self._ensure_model()
        if self.mock:
            return self._mock_run(paradigm, activity)
        return self._real_run(paradigm, activity)

    def _real_run(self, paradigm: Paradigm, activity: Activity) -> GenerationResult:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        messages = [
            {"role": "system", "content": paradigm.system_prompt},
            {"role": "user", "content": activity.prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        n_in = int(inputs["input_ids"].shape[1])

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-5),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        n_out = int(out.shape[1] - n_in)
        text = tokenizer.decode(out[0, n_in:], skip_special_tokens=True).strip()

        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2
                   if torch.cuda.is_available() else 0.0)

        return GenerationResult(
            paradigm=paradigm.name, activity=activity.name,
            coord=activity.coord, prompt=activity.prompt, text=text,
            n_input_tokens=n_in, n_output_tokens=n_out,
            latency_s=elapsed,
            tokens_per_sec=(n_out / elapsed) if elapsed > 0 else 0.0,
            peak_mem_mb=peak_mb,
        )

    def _mock_run(self, paradigm: Paradigm, activity: Activity) -> GenerationResult:
        rng = random.Random(hash((paradigm.name, activity.name)) & 0xFFFFFFFF)
        lex = [w for w in paradigm.lexicon if " " not in w]
        rng.shuffle(lex)
        picks = lex[:6] if len(lex) >= 6 else lex
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
            paradigm=paradigm.name, activity=activity.name,
            coord=activity.coord, prompt=activity.prompt, text=text,
            n_input_tokens=80, n_output_tokens=n_out,
            latency_s=elapsed, tokens_per_sec=n_out / elapsed,
            peak_mem_mb=4800.0,
        )


# ---------------------------------------------------------------------------
# 4. COMPARATOR — qualitative enforcement of the alignment rubric
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_IMPERATIVE_VERBS = {
    "do", "don't", "stop", "start", "go", "be", "consider", "remember",
    "notice", "ask", "try", "make", "let", "think", "take", "give",
    "hold", "listen", "attend", "accept", "refuse",
}


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]


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
    """Proper-noun-ish tokens + numerics, normalized to [0, 1]."""
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
    """Score one generation against its paradigm; enforce hard thresholds."""

    def score(self, result: GenerationResult, paradigm: Paradigm) -> QualityScore:
        text = result.text
        text_low = text.lower()
        words = _tokens(text)

        # lexicon: word-match for single tokens, substring for multiword
        single_lex = {x.lower() for x in paradigm.lexicon if " " not in x}
        multi_lex = [x.lower() for x in paradigm.lexicon if " " in x]
        lex_hits = sum(1 for w in words if w in single_lex)
        for phrase in multi_lex:
            lex_hits += text_low.count(phrase)

        # forbidden: substring match end-to-end (catches phrases & hyphenated)
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
            reasons.append(f"length {length} outside [{paradigm.min_length_chars}, "
                           f"{paradigm.max_length_chars}]")
        if not lex_pass:
            reasons.append(f"only {lex_hits}/{paradigm.min_lexicon_hits} "
                           f"required paradigm-keywords")
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
                    f"  {badge} {r.activity:<20} "
                    f"L={s.length_chars:>4}  lex={s.lexicon_hits}  "
                    f"imp={s.imperative_count}  ov={s.overall:.2f}"
                )
                for why in s.reasons:
                    lines.append(f"      ! {why}")

        # cross-paradigm
        by_act: dict[str, list[tuple[str, QualityScore]]] = {}
        for r, p, s in scored:
            by_act.setdefault(r.activity, []).append((p.name, s))
        cross = [(a, rows) for a, rows in by_act.items() if len(rows) >= 2]
        if cross:
            lines.append("")
            lines.append("=" * 78)
            lines.append("CROSS-PARADIGM COMPARISON (best alignment per activity)")
            lines.append("=" * 78)
            for act, rows in cross:
                best = max(rows, key=lambda kv: kv[1].overall)
                summary = "  ".join(f"{n}={s.overall:.2f}" for n, s in rows)
                lines.append(f"  {act:<22}  winner: {best[0]:<14}  {summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--paradigm", action="append",
                    help="paradigm name(s); repeatable. default: all")
    ap.add_argument("--n", type=int, default=4,
                    help="how many activities to sample (default 4)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mock", action="store_true",
                    help="skip model load; emit canned paradigm text")
    ap.add_argument("--max-new-tokens", type=int, default=180)
    ap.add_argument("--show-text", action="store_true",
                    help="print full generated text per item")
    ap.add_argument("--json", type=Path,
                    help="optional: write full results to JSON")
    args = ap.parse_args()

    paradigm_names = args.paradigm or list(PARADIGMS.keys())
    for n in paradigm_names:
        if n not in PARADIGMS:
            print(f"unknown paradigm: {n}; known: {list(PARADIGMS)}",
                  file=sys.stderr)
            sys.exit(2)

    geom = PromptGeometry()
    activities = geom.sample(args.n, seed=args.seed)
    sim = Simulator(mock=args.mock, max_new_tokens=args.max_new_tokens)
    cmp_ = Comparator()
    scored: list[tuple[GenerationResult, Paradigm, QualityScore]] = []

    print(f"running {len(paradigm_names)} paradigm(s) × "
          f"{len(activities)} activity(ies)"
          f"{' (mock mode)' if args.mock else ''}")
    for pname in paradigm_names:
        p = PARADIGMS[pname]
        for act in activities:
            r = sim.run(p, act)
            s = cmp_.score(r, p)
            scored.append((r, p, s))
            if args.show_text:
                print(f"\n--- {pname} | {act.name} "
                      f"@ (a={act.coord.agency:+.1f}, s={act.coord.scale:+.1f}, "
                      f"r={act.coord.register:+.1f}) ---")
                print(r.text)

    print()
    print(cmp_.report(scored))

    if args.json:
        payload = [
            {"generation": asdict(r), "paradigm": p.name, "score": asdict(s)}
            for r, p, s in scored
        ]
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()