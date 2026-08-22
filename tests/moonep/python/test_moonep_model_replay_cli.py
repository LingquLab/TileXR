from __future__ import annotations

import io

import pytest

from tools.moonep.model_replay_cli import resolve_shape


def test_non_interactive_mode_requires_all_five_shape_values() -> None:
    with pytest.raises(ValueError, match="missing required model replay values: EP"):
        resolve_shape(
            s="8192",
            k="16",
            hidden_size="3584",
            ep="",
            rank_size="8",
            interactive=False,
        )


def test_interactive_mode_prompts_only_for_missing_values() -> None:
    prompts = io.StringIO()
    answers = iter(("16", "16"))

    shape = resolve_shape(
        s="8192",
        k="16",
        hidden_size="3584",
        ep="",
        rank_size="",
        interactive=True,
        prompt_stream=prompts,
        input_fn=lambda: next(answers),
    )

    assert (shape.tokens_per_rank, shape.topk, shape.hidden_size) == (8192, 16, 3584)
    assert (shape.ep_size, shape.world_size) == (16, 16)
    assert prompts.getvalue().splitlines() == ["Enter EP:", "Enter R:"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("s", "0", "S must be a positive integer"),
        ("k", "33", "K must not exceed expert count 32"),
        ("hidden_size", "127", "H must be divisible by 128"),
        ("ep", "8", "EP must equal R"),
    ),
)
def test_shape_validation_matches_model_runner_contract(
    field: str, value: str, message: str
) -> None:
    values = {
        "s": "8192",
        "k": "16",
        "hidden_size": "3584",
        "ep": "16",
        "rank_size": "16",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        resolve_shape(**values, interactive=False)


def test_shape_derives_model_expert_and_capacity_contract() -> None:
    shape = resolve_shape(
        s="4096",
        k="8",
        hidden_size="7168",
        ep="16",
        rank_size="16",
        interactive=False,
    )

    assert shape.expert_count == 32
    assert shape.experts_per_rank == 2
    assert shape.ffn_hidden_size == 2048
    assert shape.token_padding == 1
    assert shape.forward_calls == 10
