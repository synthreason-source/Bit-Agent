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

Query-driven selection
----------------------
The default flow now reads a user prompt (via --query or interactively)
and selects the activities whose names and prompts most overlap with
that query, using a stop-word-filtered content-token bag with crude
suffix stemming. The point is to surface the everyday acts that
*resonate* with what's on the user's mind, then show how each paradigm
would frame each of them. Pass --random to ignore the query and pick
activities uniformly at random instead.

Hooks the model
---------------
BitNet is loaded inline via transformers — no dependency on x.py or any
other external loader file. The default checkpoint is the BF16 master
weights (microsoft/bitnet-b1.58-2B-4T-bf16), which avoids the v5
weight-conversion path that broke the packed checkpoint. Pass --mock
to skip the model load entirely and emit canned paradigm-flavored
outputs — useful for demoing on a laptop.

torch.compile is disabled at import (TORCHDYNAMO_DISABLE=1) so BitNet
runs in eager mode and doesn't need MSVC / g++ available on PATH.

Examples
--------
    python everyday.py --mock --query "I'm stuck and avoiding everything"
    python everyday.py --mock --query "my mother is unwell" --show-text
    python everyday.py --query "I can't sleep" --paradigm stoic --paradigm care
    python everyday.py --mock --random --n 4                       # old behavior
    python everyday.py                                             # asks you
"""

from __future__ import annotations

import os
# Disable torch.compile / Inductor before anything imports torch.
# BitNet's BitLinear is @torch.compile-decorated, which tries to JIT a CPU
# kernel on first forward — that needs cl.exe on Windows or g++ on Linux.
# Setting this means @torch.compile just passes through to eager mode.
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
# 0. SHARED TEXT UTILITIES — used by geometry selection and comparator
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

# Stopwords are themselves a choice — these are mostly grammatical glue
# plus a few high-frequency mental-state verbs ("feel", "think") that
# would otherwise dominate the overlap signal for almost every query.
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
    "stuff", "really", "im", "ive", "id", "dont", "cant", "wont", "isnt",
    "thats", "whats", "lot", "lots", "bit", "going", "got", "get",
    # Indefinite/generic words that pad prompts but carry no signal.
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
    """Crude suffix stripper. Not linguistically correct, just consistent.

    "running" -> "runn", "feels" -> "feel", "worked" -> "work".
    Consistency matters more than correctness: identical inputs produce
    identical stems on both sides of the comparison.
    """
    for suffix in ("ings", "ing", "edly", "ed", "ly", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[: -len(suffix)]
    return word


def _content_tokens(text: str) -> set[str]:
    """Stop-word-filtered, stemmed bag of content tokens."""
    return {_stem(w) for w in _tokens(text)
            if w not in _STOPWORDS and len(w) > 2}


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




# Concept tags per activity. These bridge user vocabulary to activity
# vocabulary — a query about "partner upset" can match "apologising" via
# the shared tag "conflict", even though no surface word overlaps.
# Tags are matched alongside the activity's name and prompt tokens; tag
# matches are weighted more heavily because they're curated.
_ACTIVITY_TAGS: dict[str, frozenset[str]] = {
    # Routine / Domestic
    "waiting in line":              frozenset({"waiting", "pause", "patience", "public", "boredom"}),
    "taking a shower":              frozenset({"body", "private", "ritual", "morning", "reflection"}),
    "doing laundry":                frozenset({"chore", "repetitive", "domestic", "routine", "alone"}),
    "eating lunch alone":           frozenset({"meal", "alone", "lonely", "work", "solitude", "midday"}),
    "making coffee":                frozenset({"morning", "ritual", "small", "routine", "domestic"}),
    "folding clothes":              frozenset({"chore", "repetitive", "domestic", "routine", "quiet"}),
    "walking the dog":              frozenset({"walk", "outside", "routine", "animal", "evening"}),
    "taking out the trash":         frozenset({"chore", "domestic", "small", "routine"}),
    "watering the plants":          frozenset({"care", "domestic", "small", "ritual", "attention"}),
    "tidying the desk":             frozenset({"transition", "work", "clearing", "preparation"}),

    # Social / Relational
    "meeting someone new":          frozenset({"introduction", "stranger", "social", "anxious", "nervous", "worried", "first"}),
    "catching up with a friend":    frozenset({"friend", "social", "conversation", "reunion", "care"}),
    "helping a stranger":           frozenset({"help", "stranger", "kindness", "moral", "interruption"}),
    "receiving criticism":          frozenset({"feedback", "hurt", "criticism", "conflict", "self", "stung", "argument"}),
    "giving a compliment":          frozenset({"praise", "kindness", "social", "small", "generous"}),
    "apologising":                  frozenset({"apology", "hurt", "conflict", "fight", "argument", "repair", "partner", "wife", "husband", "boyfriend", "girlfriend", "relationship", "family", "mother", "father", "parent", "mom", "dad", "sister", "brother", "sorry", "regret"}),
    "being ignored":                frozenset({"lonely", "rejected", "hurt", "invisible", "social", "sad"}),
    "forgiving someone":            frozenset({"forgive", "hurt", "conflict", "fight", "argument", "repair", "anger", "angry", "resentment"}),
    "a difficult conversation":     frozenset({"conflict", "fight", "argument", "argue", "hard", "talk", "partner", "wife", "husband", "boyfriend", "girlfriend", "relationship", "family", "mother", "father", "parent", "mom", "dad", "sister", "brother", "honest", "confrontation", "upset"}),
    "being cared for":              frozenset({"care", "received", "sick", "loved", "vulnerable", "partner", "family", "mother", "father", "parent", "mom", "dad"}),

    # Work / Productivity
    "starting a new project":       frozenset({"begin", "work", "project", "new", "anxious", "nervous", "excited"}),
    "facing a deadline":            frozenset({"deadline", "stress", "stressed", "work", "pressure", "anxious", "worried", "overwhelmed", "rush"}),
    "procrastinating":              frozenset({"avoid", "stuck", "overwhelmed", "anxious", "worried", "delay", "resistance", "lazy"}),
    "finishing a task":             frozenset({"complete", "work", "done", "transition", "satisfaction", "happy"}),
    "feeling stuck":                frozenset({"stuck", "frustrated", "blocked", "problem", "overwhelmed", "spinning", "lost"}),
    "learning something new":       frozenset({"learn", "study", "skill", "beginner", "confused", "growth"}),
    "making a mistake at work":     frozenset({"mistake", "error", "shame", "work", "hurt", "self", "embarrassed", "boss", "manager", "colleague"}),
    "getting a promotion":          frozenset({"promotion", "success", "work", "milestone", "celebrate", "happy", "boss", "manager"}),
    "being bored":                  frozenset({"bored", "empty", "restless", "unstimulated", "stuck"}),
    "resigning":                    frozenset({"quit", "leave", "work", "transition", "milestone", "decision", "boss", "manager"}),

    # Body / Health
    "going for a run":              frozenset({"run", "exercise", "body", "outside", "movement", "morning"}),
    "sitting with pain":            frozenset({"pain", "body", "hurt", "suffering", "endure", "physical", "sick", "ache"}),
    "a medical appointment":        frozenset({"medical", "doctor", "health", "waiting", "anxious", "nervous", "worried", "scared", "body", "sick"}),
    "preparing to sleep but can't": frozenset({"sleep", "insomnia", "night", "anxious", "worried", "tired", "exhausted", "mind", "racing"}),
    "eating mindlessly":            frozenset({"eat", "mindless", "distracted", "habit", "body"}),
    "meditating":                   frozenset({"meditate", "quiet", "attention", "stillness", "practice", "mind", "calm"}),
    "a long walk alone":            frozenset({"walk", "alone", "solitary", "outside", "thinking", "wander"}),
    "recovering from illness":      frozenset({"sick", "rest", "recovery", "body", "tired", "exhausted", "weak", "ill"}),
    "getting a haircut":            frozenset({"waiting", "appearance", "small", "service", "passive"}),

    # Milestone / Emotional
    "a birthday":                   frozenset({"birthday", "milestone", "year", "time", "self", "celebrate", "happy"}),
    "moving house":                 frozenset({"move", "change", "transition", "home", "leaving", "milestone"}),
    "ending a friendship":          frozenset({"friend", "ending", "loss", "drift", "grief", "sad", "relationship"}),
    "looking at old photos":        frozenset({"memory", "past", "nostalgia", "photos", "grief", "miss", "missing", "time"}),
    "receiving bad news":           frozenset({"shock", "grief", "loss", "hard", "overwhelmed", "sad", "depressed", "scared"}),
    "celebrating a milestone":      frozenset({"celebrate", "milestone", "success", "joy", "happy", "achievement"}),
    "being stuck in traffic":       frozenset({"traffic", "late", "stuck", "transit", "frustrated", "angry", "waiting"}),
    "watching the sunset":          frozenset({"beauty", "nature", "evening", "still", "noticing", "small", "peaceful"}),
    "thinking about death":         frozenset({"death", "mortality", "fear", "scared", "meaning", "existential", "grief", "dying"}),
    "feeling grateful":             frozenset({"gratitude", "thankful", "appreciation", "joy", "happy", "noticing"}),
    "a disagreement online":        frozenset({"argument", "argue", "fight", "conflict", "online", "stranger", "anger", "angry", "frustrated"}),
}


ACTIVITIES: list[Activity] = [
    # ── Routine / Domestic ──────────────────────────────────────────────
    Activity("waiting in line",        Coord(-0.3, -0.6, -1.0),
             "I'm standing in a long queue. What might I do with this unscheduled pause?"),
    Activity("taking a shower",        Coord(+0.4, -0.2, -0.5),
             "I'm in the shower — one of the few truly private moments. What's worth thinking about here?"),
    Activity("doing laundry",          Coord(-0.1, -0.7, -0.8),
             "I'm loading the washing machine. How might I frame this repetitive task?"),
    Activity("eating lunch alone",     Coord(+0.1, -0.6, -0.5),
             "I'm eating lunch by myself today. What's worth attending to during a solo meal?"),
    Activity("making coffee",          Coord(+0.5, -0.1, -0.3),
             "I'm making my morning coffee. What's worth noticing in this small ritual?"),
    Activity("folding clothes",        Coord(-0.2, -0.8, -0.7),
             "I'm folding a pile of laundry. How might I inhabit this quiet, repetitive work?"),
    Activity("walking the dog",        Coord(+0.7, +0.2, +0.0),
             "I'm on the evening walk with my dog. What's worth attending to on this regular loop?"),
    Activity("taking out the trash",   Coord(-0.4, -0.7, -0.8),
             "I'm taking out the bins. Is there anything worth noticing in this small domestic act?"),
    Activity("watering the plants",    Coord(+0.5, -0.3, +0.0),
             "I'm watering the houseplants. What might I pay attention to right now?"),
    Activity("tidying the desk",       Coord(+0.2, -0.4, -0.2),
             "I'm clearing off my desk before starting work. How should I approach this transition?"),

    # ── Social / Relational ─────────────────────────────────────────────
    Activity("meeting someone new",    Coord(+0.3, +0.6, -0.3),
             "I'm about to be introduced to someone I don't know. How should I show up?"),
    Activity("catching up with a friend", Coord(+0.8, +0.4, +0.2),
             "I'm meeting a friend I haven't seen in months for coffee. What might I bring to the conversation?"),
    Activity("helping a stranger",     Coord(+0.7, +0.3, +0.5),
             "I just stopped to help someone who seemed lost or struggling. How should I think about this moment?"),
    Activity("receiving criticism",    Coord(-0.2, +0.5, -0.6),
             "Someone just gave me feedback that stung a little. How do I sit with this?"),
    Activity("giving a compliment",    Coord(+0.6, +0.2, +0.3),
             "I'm about to tell someone something genuinely good about them. What makes this worth doing well?"),
    Activity("apologising",            Coord(+0.1, +0.3, -0.7),
             "I'm about to apologise to someone I hurt or let down. How should I approach this?"),
    Activity("being ignored",          Coord(-0.6, -0.1, -0.8),
             "I feel like I'm being overlooked right now. How should I hold this feeling?"),
    Activity("forgiving someone",      Coord(+0.3, -0.2, +0.3),
             "I'm working through forgiving someone who wronged me. What does that actually require?"),
    Activity("a difficult conversation", Coord(-0.1, +0.6, -0.2),
             "I need to have a hard conversation with someone today. How do I prepare for it?"),
    Activity("being cared for",        Coord(+0.7, -0.1, -0.5),
             "Someone is taking care of me right now — cooking, checking in, helping out. What do I notice?"),

    # ── Work / Productivity ─────────────────────────────────────────────
    Activity("starting a new project", Coord(+0.6, +0.7, +0.3),
             "I'm beginning a new project I've been planning for a while. How do I start well?"),
    Activity("facing a deadline",      Coord(-0.2, +0.8, -0.3),
             "A deadline is approaching fast. How do I think about the next few hours?"),
    Activity("procrastinating",        Coord(-0.5, -0.4, -0.6),
             "I keep putting off something I know I need to do. What's actually happening here?"),
    Activity("finishing a task",       Coord(+0.7, +0.3, +0.4),
             "I just completed something I've been working on. How do I close it properly?"),
    Activity("feeling stuck",          Coord(-0.4, +0.2, -0.7),
             "I've been staring at the same problem for an hour and going nowhere. What now?"),
    Activity("learning something new", Coord(+0.5, +0.5, -0.2),
             "I'm in the middle of learning a new skill or concept I don't fully understand yet. How do I stay with it?"),
    Activity("making a mistake at work", Coord(-0.6, +0.5, -0.7),
             "I just made an error that matters. How should I respond to myself and the situation?"),
    Activity("getting a promotion",    Coord(+0.9, +0.8, +0.7),
             "I just found out I've been promoted. How do I receive this news well?"),
    Activity("being bored",            Coord(-0.3, -0.9, -0.8),
             "Nothing is holding my attention. I'm genuinely bored. What should I do with that?"),
    Activity("resigning",              Coord(+0.2, +0.6, +0.2),
             "I'm about to hand in my resignation. What's worth reflecting on before I do?"),

    # ── Body / Health ───────────────────────────────────────────────────
    Activity("going for a run",        Coord(+0.6, +0.8, +0.3),
             "I'm heading out for a run. How might I use this time beyond just the exercise?"),
    Activity("sitting with pain",      Coord(-0.5, +0.3, -0.6),
             "I'm dealing with physical pain right now. How do I be with it rather than just against it?"),
    Activity("a medical appointment",  Coord(-0.1, +0.5, -0.5),
             "I'm sitting in a waiting room before seeing a doctor. How do I orientate to what's coming?"),
    Activity("preparing to sleep but can't", Coord(-0.4, +0.3, -0.7),
             "I'm lying in bed but my mind won't settle. What might help me find rest?"),
    Activity("eating mindlessly",      Coord(-0.3, -0.5, -0.7),
             "I realise I've been eating without really noticing. What might I return to?"),
    Activity("meditating",             Coord(+0.5, -0.8, +0.2),
             "I've just sat down to meditate. How do I actually begin?"),
    Activity("a long walk alone",      Coord(+0.6, -0.1, +0.3),
             "I'm taking a long solitary walk with no destination in mind. What might I think about?"),
    Activity("recovering from illness", Coord(-0.2, -0.5, -0.6),
             "I'm home sick and resting. How do I spend this unexpected stillness?"),
    Activity("getting a haircut",      Coord(+0.2, -0.3, -0.5),
             "I'm sitting in the barber's chair with nothing to do but wait. What's worth attending to?"),

    # ── Milestone / Emotional ───────────────────────────────────────────
    Activity("a birthday",             Coord(+0.6, +0.5, +0.2),
             "It's my birthday today. How do I think about what this marker means?"),
    Activity("moving house",           Coord(+0.1, +0.7, +0.0),
             "I'm packing up my home to move somewhere new. What do I want to carry forward — and leave behind?"),
    Activity("ending a friendship",    Coord(-0.3, +0.2, -0.3),
             "A friendship seems to be ending, not through conflict but through drift. How do I sit with this?"),
    Activity("looking at old photos",  Coord(+0.3, -0.1, -0.3),
             "I've been looking through old photos. What do I want to notice, and what do I want to do with these feelings?"),
    Activity("receiving bad news",     Coord(-0.8, +0.6, -0.8),
             "I've just received news that is genuinely bad. How do I begin to hold this?"),
    Activity("celebrating a milestone", Coord(+0.9, +0.7, +0.6),
             "Something significant I worked toward has happened. How do I actually let myself celebrate?"),
    Activity("being stuck in traffic", Coord(-0.4, +0.2, -0.9),
             "I'm stuck in traffic and running late. How do I use this time without spiralling?"),
    Activity("watching the sunset",    Coord(+0.7, -0.3, +0.1),
             "I stopped to watch the sun go down. What's worth noticing in this ordinary miracle?"),
    Activity("thinking about death",   Coord(-0.1, +0.1, +0.4),
             "My own mortality has come into focus today. How do I think clearly about death?"),
    Activity("feeling grateful",       Coord(+0.9, +0.1, +0.3),
             "I feel a quiet, genuine gratitude right now. How do I honour and deepen that feeling?"),
    Activity("a disagreement online",  Coord(-0.5, +0.7, -0.3),
             "I've just gotten into an argument with a stranger online. What's worth examining here?"),
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

    def select_by_query(self, query: str, n: int) -> list[tuple[Activity, float]]:
        """Rank activities by content-token overlap with the user's query.

        Three signals are combined:
          - name overlap   (weight 2.0): activity-name tokens vs query
          - prompt overlap (weight 1.0): activity-prompt tokens vs query
          - tag overlap    (weight 3.0): curated concept tags vs query

        Tags are the most reliable signal because they're curated to
        bridge user vocabulary to activity vocabulary — a query about
        "partner upset" can find "apologising" via the shared concept
        tag "conflict", with no shared surface words required.

        If no activity has any overlap at all, falls back to a random
        sample seeded by the query (so empty queries stay deterministic).
        """
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

_BITNET_CANDIDATES = (
    "microsoft/bitnet-b1.58-2B-4T",
)


def _load_bitnet_inline():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    last_err = None
    for model_id in _BITNET_CANDIDATES:
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch.bfloat16, device_map=device,
            )
            mdl.eval()
            return mdl, tok
        except Exception as e:           # noqa: BLE001 — we want to try the next id
            last_err = e
            continue
    raise RuntimeError(
        f"could not load any BitNet checkpoint from {_BITNET_CANDIDATES}: {last_err}"
    )


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
      - real:  loads BitNet inline and generates
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
        self._model, self._tokenizer = _load_bitnet_inline()

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
                    f"  {badge} {r.activity:<28} "
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
                lines.append(f"  {act:<28}  winner: {best[0]:<14}  {summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def _resolve_query(args: argparse.Namespace) -> str | None:
    """Decide what query text to use, or None if user asked for --random."""
    if args.random:
        return None
    if args.query is not None:
        return args.query.strip() or None
    # interactive fallback: ask the user
    try:
        print("describe what's on your mind (or press Enter for a random sample):")
        q = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    return q or None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--query", "-q",
                    help="user prompt to drive activity selection; "
                         "if omitted, asks interactively")
    ap.add_argument("--random", action="store_true",
                    help="ignore --query and pick activities at random")
    ap.add_argument("--paradigm", action="append",
                    help="paradigm name(s); repeatable. default: all")
    ap.add_argument("--n", type=int, default=4,
                    help="how many activities to use (default 4)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for --random sampling")
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

    print(f"\nrunning {len(paradigm_names)} paradigm(s) × "
          f"{len(activities)} activity(ies)"
          f"{' (mock mode)' if args.mock else ''}\n")

    sim = Simulator(mock=args.mock, max_new_tokens=args.max_new_tokens)
    cmp_ = Comparator()
    scored: list[tuple[GenerationResult, Paradigm, QualityScore]] = []

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
    input()
