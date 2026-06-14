"""Regression tests — inference helpers (turn-boundary cleanup, sampling), the
conversational system-prompt + mood layer, tokenizer control symbols, and context/
token accounting. (Split from the old test_fixes.py.)"""

import numpy as np
from fixes_common import B

# ─────────────────────────── agent: turn-boundary cleanup (chat) ─────────────


def test_clean_answer_cuts_inline_role_tag():
    from uses.common.agent import clean_answer

    leak = "I'm not sure. User: hi Assistant: hello"
    assert clean_answer(leak) == "I'm not sure."


def test_first_stop_index_inline_and_ignores_leading():
    from uses.common.agent import first_stop_index

    assert first_stop_index("ok. User: x") == 4  # inline boundary
    assert first_stop_index("Assistant: hi") is None  # leading primed tag ignored


def test_safe_stream_len_holds_back_role_prefix():
    from uses.common.agent import safe_stream_len

    # trailing "User" could still become "User:" → held back
    assert safe_stream_len("done. User") == len("done. ")
    # plain text fully emittable
    assert safe_stream_len("hello there") == len("hello there")


def test_strip_thinking_removes_scratchpad():
    from uses.common.agent import strip_thinking

    assert strip_thinking("<think>plan</think>answer").strip() == "answer"


# ─────────────────────────── sampling: rep penalty + top_k (L6) ──────────────


def test_rep_penalty_demotes_recent_token():
    import mlx.core as mx

    from uses.chat.run_chat import sample_top_p

    logits = mx.array(np.array([1.0, 5.0, 1.0, 1.0], dtype=np.float32))  # token 1 peaks
    # temperature 0 → argmax; penalizing token 1 hard should move the choice off it.
    out = sample_top_p(logits, temperature=0.0, top_p=1.0, recent_ids=[1], rep_penalty=10.0)
    assert out != 1


def test_top_k_restricts_choices():
    import mlx.core as mx

    from uses.chat.run_chat import sample_top_p

    logits = mx.array(np.array([10.0, 9.0, -50.0, -50.0], dtype=np.float32))
    picks = {sample_top_p(logits, temperature=1.0, top_p=1.0, top_k=2) for _ in range(50)}
    assert picks <= {0, 1}  # only the top-2 are ever sampled


# ─────────────────────────── tokenizer: central control symbols (#1, C2) ─────


def test_tokenizer_symbols_include_control_and_modality():
    from src.core.modalities.vocab import CONTROL_SPECIALS, tokenizer_symbols

    syms = tokenizer_symbols(["en", "es"])
    for s in (
        "<lang:en>",
        "<lang:es>",
        "<mod:text>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
    ):
        assert s in syms
    assert "<think>" in CONTROL_SPECIALS


def test_agent_think_delimiters_match_registry():
    from src.core.modalities.vocab import REASONING_SPECIALS
    from uses.common.agent import THINK_CLOSE, THINK_OPEN

    assert [THINK_OPEN, THINK_CLOSE] == REASONING_SPECIALS


# ───────────────── system prompt + mood (conversational layer) ───────────────


def test_emotion_maps_to_mood_palette():
    from src.core.modalities.moods import MOODS, emotion_to_mood

    assert emotion_to_mood("joyful") == "happy"
    assert emotion_to_mood("terrified") == "afraid"
    assert emotion_to_mood("caring") == "caring"
    assert emotion_to_mood("totally-unknown-emotion") == "neutral"  # default
    assert emotion_to_mood(None) == "neutral"
    assert all(
        emotion_to_mood(e) in MOODS for e in ("sad", "angry", "surprised", "proud", "anxious")
    )


def test_mood_system_phrase_neutral_is_empty():
    from src.core.modalities.moods import mood_system_phrase

    assert mood_system_phrase("neutral") == ""  # default adds nothing
    assert mood_system_phrase("happy") == "(mood: happy)"
    assert mood_system_phrase("bogus") == ""  # unknown → nothing


def test_system_preamble_framing():
    from uses.common import agent

    assert agent.system_preamble(None, "neutral") == ""  # nothing → no line
    assert agent.system_preamble("Be kind.", "neutral") == "System: Be kind.\n"
    assert agent.system_preamble(None, "sad") == "System: (mood: sad)\n"
    assert agent.system_preamble("Be kind.", "happy") == "System: Be kind. (mood: happy)\n"


def test_agent_prompt_prepends_system_persona():
    from uses.common import agent

    p = agent.build_agent_prompt([], "hello", system="You are terse.")
    assert p.startswith("System: You are terse. ")  # persona ahead of tool spec
    assert "User: hello" in p and p.rstrip().endswith("Assistant:")


def test_data_enrichment_system_and_story():
    """instruct system injection yields a System line; story reframing is a NATURAL
    User→Assistant request with NO system prompt (telling a story needs no persona)."""
    import src.core.data.graded as g

    sysd = g._prepend_system("User: q\nAssistant: a", "You are kind.", "happy")
    assert sysd.startswith("System: You are kind. (mood: happy)\nUser:")
    # the story-request format the stream emits
    story = f"User: {g._STORY_PROMPTS[0]}\nAssistant: Once upon a time."
    assert not story.startswith("System:")  # no persona gate for stories
    assert "Assistant:" in story


def test_classify_mood_defaults_neutral_without_head():
    from src.core.model.mood import classify_mood

    mood, _conf = classify_mood(None, None, None, "anything")
    assert mood == "neutral"


def test_mood_tracker_neutral_without_head():
    from src.core.model.mood import MoodTracker

    t = MoodTracker(None)
    assert t.update(None, None, "I am so happy!") == "neutral"  # one msg ⇒ inertia


def test_lexicon_mood_fixes_broken_classifications():
    """The learned 11M head was near-random ('im good'→angry, 'my dog died'→caring,
    requests→emotion). The lexicon is the reliable floor: clear cues map correctly,
    requests/questions stay neutral, and negation flips a positive cue to sad."""
    from src.core.modalities.moods import lexicon_mood

    cases = {
        "im good": "happy",
        "i am so happy today": "happy",
        "thanks for your help": "happy",
        "my dog died": "sad",
        "i feel terrible": "sad",
        "i am not good": "sad",
        "i hate this": "angry",
        "im scared of the dark": "afraid",
        "tell me a story": "neutral",
        "what is 2+2": "neutral",
        "can you help me with math": "neutral",
        "how are you": "neutral",
    }
    for text, want in cases.items():
        got, _ = lexicon_mood(text)
        assert got == want, f"{text!r}: got {got}, want {want}"


def test_mood_tracker_lexicon_drives_mood_without_a_head():
    """No learned head needed: a sustained emotional tone is detected by the lexicon
    alone (the head is only an optional refinement)."""
    from src.core.model.mood import MoodTracker

    t = MoodTracker(None, alpha=0.5)
    last = "neutral"
    for _ in range(5):
        last = t.update(None, None, "i am so happy and grateful")
    assert last == "happy"
    for _ in range(8):
        last = t.update(None, None, "tell me about cats")  # neutral request
    assert last == "neutral"  # decays back


def test_mood_tracker_builds_and_decays_over_conversation(monkeypatch):
    """Conversation-aware mood: one message isn't enough (inertia), a sustained tone
    takes hold, and it decays back to neutral — emotion is the WHOLE exchange."""
    import src.core.model.mood as mood
    from src.core.model.mood import MOOD_INDEX, MOODS, MoodTracker

    happy = [0.0] * len(MOODS)
    happy[MOOD_INDEX["happy"]] = 1.0
    neutral = [0.0] * len(MOODS)
    neutral[0] = 1.0
    monkeypatch.setattr(
        mood, "mood_probs", lambda m, t, h, text, **k: happy if "joy" in text else neutral
    )
    tr = MoodTracker(head=object(), alpha=0.4)
    assert tr.update(None, None, "joy") == "neutral"  # one message ⇒ inertia
    for _ in range(4):
        m = tr.update(None, None, "joy")
    assert m == "happy"  # sustained tone takes hold
    for _ in range(6):
        m = tr.update(None, None, "calm")
    assert m == "neutral"  # decays back to default


def test_mood_tracker_reset(monkeypatch):
    import src.core.model.mood as mood
    from src.core.model.mood import MOOD_INDEX, MOODS, MoodTracker

    happy = [0.0] * len(MOODS)
    happy[MOOD_INDEX["happy"]] = 1.0
    monkeypatch.setattr(mood, "mood_probs", lambda *a, **k: happy)
    tr = MoodTracker(head=object(), alpha=0.6)
    for _ in range(5):
        tr.update(None, None, "x")
    assert tr.current() == "happy"
    tr.reset()
    assert tr.current() == "neutral"


def test_mood_head_learns_to_separate_moods():
    """The mood head should fit a tiny separable set (sanity that the classifier
    + train step are wired): loss drops over a few steps on frozen features."""
    import mlx.core as mx

    from src.core.model.mood import MoodHead, mood_loss

    head = MoodHead(d_model=32, hidden=16)
    opt = B.engine.make_optimizer(head, 1e-2, 0.0)
    rng = np.random.RandomState(0)
    # 3 clusters of features → 3 mood labels; learnable by a small MLP.
    centers = rng.randn(3, 32)
    feats = np.vstack([centers[i] + 0.05 * rng.randn(20, 32) for i in range(3)]).astype(np.float32)
    labels = np.array([i for i in range(3) for _ in range(20)], dtype=np.float32)
    h = mx.array(feats)
    y = mx.array(labels)

    def loss_fn(hd):
        return mood_loss(hd(h), y)

    lg = B.engine.value_and_grad(head, loss_fn)
    first = float(lg(head)[0].item())
    for _ in range(60):
        loss, grads = lg(head)
        B.engine.optimizer_step(opt, head, grads)
    assert float(loss.item()) < first - 0.3  # clearly learned


# ───────────────── context / token accounting (observability + billing) ──────


def test_context_report_accounting_and_billing_dict():
    from src.core.observability import ContextReport

    r = ContextReport(
        surface="chat",
        context_len=512,
        system_tokens=12,
        history_tokens=300,
        tokens_in=320,
        tokens_out=28,
        tokens_reasoning=15,
        mood="happy",
        mood_dist={"happy": 0.42, "neutral": 0.3, "caring": 0.1},
        memory_files=3,
        tps=200.0,
        params={"temp": 0.7, "top_p": 0.9},
    )
    assert r.used == 348 and r.free == 512 - 348
    assert round(r.fill_pct, 1) == round(100 * 348 / 512, 1)
    d = r.to_dict()  # billing/telemetry payload
    for k in (
        "surface",
        "tokens_in",
        "tokens_out",
        "tokens_reasoning",
        "system_tokens",
        "mood",
        "memory_files",
        "used",
        "free",
        "fill_pct",
    ):
        assert k in d
    assert d["used"] == 348 and d["mood"] == "happy"
    panel = r.render()
    assert "tokens in" in panel and "window" in panel and "mood" in panel
    assert r.render_compact().startswith("  [in 320 · out 28 · think 15")


def test_count_tokens_is_safe():
    from src.core.observability import count_tokens

    class _T:
        def encode(self, t, add_bos=True, add_eos=True):
            return list(range(len(t)))

    assert count_tokens(_T(), "abcd") == 4
    assert count_tokens(_T(), "") == 0
    assert count_tokens(None, "x") == 0  # no tokenizer ⇒ 0, never crashes
