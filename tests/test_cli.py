"""Command-line entry points: the bits that are easy to get wrong outside the engine."""

import types

import pytest

from nanoinfer.__main__ import _fit_history, _use_utf8


class FakeLLM:
    """Just enough of LLM for _fit_history: 4 tokens per message plus 3 for the template."""

    def __init__(self, max_model_len):
        self.config = types.SimpleNamespace(max_model_len=max_model_len)

    def apply_chat_template(self, history):
        return list(range(3 + 4 * len(history)))


def _args(max_tokens, system=True):
    return types.SimpleNamespace(max_tokens=max_tokens, system="sys" if system else None)


def test_short_history_is_left_alone():
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    ids, dropped = _fit_history(FakeLLM(256), history, _args(32))
    assert dropped == 0 and len(history) == 2 and len(ids) == 11


def test_long_history_drops_oldest_turns_but_keeps_the_system_message():
    """A growing conversation used to walk straight into the engine's 'prompt has N tokens;
    max_model_len is M' error and end the chat session."""
    history = [{"role": "system", "content": "sys"}]
    history += [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i)} for i in range(40)]
    newest = history[-1]
    ids, dropped = _fit_history(FakeLLM(128), history, _args(32))
    assert dropped > 0
    assert history[0]["role"] == "system"            # never dropped
    assert history[-1] is newest                     # the new question survives
    assert len(ids) <= 128 - 32                      # and there is room left to answer in
    assert all(m["content"] != "0" for m in history[1:])   # the oldest exchanges went first


def test_a_single_message_too_long_is_reported_not_raised():
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "war and peace"}]
    ids, _ = _fit_history(FakeLLM(8), history, _args(4))
    assert ids is None                               # caller prints a message and keeps chatting


def test_reply_room_is_capped_so_a_huge_max_tokens_still_leaves_a_prompt_budget():
    history = [{"role": "user", "content": "hi"}]
    ids, dropped = _fit_history(FakeLLM(256), history, _args(10_000, system=False))
    assert ids is not None and dropped == 0          # reserve is clamped to 3/4 of the window


def test_use_utf8_is_idempotent_and_never_raises():
    _use_utf8()
    _use_utf8()


@pytest.mark.parametrize("cmd", ["chat", "generate", "info"])
def test_help_works_for_every_subcommand(cmd, capsys):
    from nanoinfer.__main__ import main
    with pytest.raises(SystemExit) as e:
        main([cmd, "--help"])
    assert e.value.code == 0
    assert "--model" in capsys.readouterr().out
