"""Stage 7 data sources — cognitive ethics and BCF.

Synthetic preschool ethics (good/wrong judgments on concrete actions, as statements
and yes/no Q&A, EN+ES) blended with the real public-domain ethics seed supplied by
the data-prep pipeline — so stage 7 isn't a 12-line, 347-token corpus.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from src.core.data.blend import blend

# Preschool right/wrong scenarios. Phrased to slot into "It is good/wrong to {act}"
# and "Is it okay to {act}?" — kept concrete and simple (level-1 ethics/BCF seed).
_ETHIC_GOOD = {
    "en": [
        "share your toys",
        "help a friend",
        "tell the truth",
        "say thank you",
        "be kind to others",
        "wait your turn",
        "clean up your mess",
        "listen when someone talks",
        "help someone who is hurt",
        "be gentle with pets",
    ],
    "es": [
        "compartir tus juguetes",
        "ayudar a un amigo",
        "decir la verdad",
        "dar las gracias",
        "ser amable con los demás",
        "esperar tu turno",
        "recoger lo que ensucias",
        "escuchar cuando alguien habla",
        "ayudar a quien está herido",
        "tratar bien a las mascotas",
    ],
}
_ETHIC_BAD = {
    "en": [
        "hit someone",
        "lie to a friend",
        "take what is not yours",
        "be mean to others",
        "break things on purpose",
        "cheat in a game",
        "call people names",
        "push other children",
        "make a big mess and leave it",
        "ignore someone who needs help",
    ],
    "es": [
        "pegarle a alguien",
        "mentirle a un amigo",
        "tomar lo que no es tuyo",
        "ser malo con los demás",
        "romper cosas a propósito",
        "hacer trampa en un juego",
        "insultar a la gente",
        "empujar a otros niños",
        "hacer un desorden y dejarlo",
        "ignorar a quien necesita ayuda",
    ],
}
_ETHIC_T = {
    "en": {
        "good_stmt": "It is good to {a}.",
        "bad_stmt": "It is wrong to {a}.",
        "q": "User: Is it okay to {a}?\nAssistant: {yn}, it is {jud} to {a}.",
        "yes": "Yes",
        "no": "No",
        "jgood": "good",
        "jbad": "wrong",
    },
    "es": {
        "good_stmt": "Está bien {a}.",
        "bad_stmt": "Está mal {a}.",
        "q": "User: ¿Está bien {a}?\nAssistant: {yn}, {jud} {a}.",
        "yes": "Sí",
        "no": "No",
        "jgood": "está bien",
        "jbad": "está mal",
    },
}


def gen_ethics(n: int, seed: int = 1, langs: list[str] | None = None) -> Iterator[dict]:
    """Synthetic preschool ethics: good/wrong judgments on concrete actions, as
    statements and yes/no Q&A. Emits only requested languages (en/es)."""
    rng = random.Random(seed)
    pool = [lang for lang in (langs or ["en"]) if lang in _ETHIC_T] or ["en"]
    for _ in range(n):
        lang = rng.choice(pool)
        template = _ETHIC_T[lang]
        good = rng.random() < 0.5
        act = rng.choice((_ETHIC_GOOD if good else _ETHIC_BAD)[lang])
        if rng.random() < 0.5:  # statement
            text = (template["good_stmt"] if good else template["bad_stmt"]).format(a=act)
        else:  # yes/no Q&A
            text = template["q"].format(
                a=act,
                yn=(template["yes"] if good else template["no"]),
                jud=(template["jgood"] if good else template["jbad"]),
            )
        yield {"text": text, "lang": lang}


def _build_ethics(*, langs, approx_examples, extra_streamers=None, **_):
    # Real ethics seed (public-domain snippets) comes from the data-prep pipeline,
    # which supplies it via extra_streamers; top it up with synthetic to fill budget.
    real = (
        extra_streamers["ethics"]() if extra_streamers and "ethics" in extra_streamers else iter(())
    )
    return blend(real, gen_ethics(approx_examples, langs=langs), approx_examples)


SOURCES = {"ethics": _build_ethics}
