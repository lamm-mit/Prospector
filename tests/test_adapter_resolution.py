"""Where an adapter chain's weights and tokenizer come from: the path rules in
`train/adapters.py` (`adapter_chain`, `resolve_base`, `tokenizer_source`), and
that no other module in the package re-derives them.

Pure path rules, so this needs neither torch nor peft; `test_adapter_guard.py`
covers the tensor-integrity check.
"""
import ast
import json
import pathlib

import pytest

from graph_reasoner.train.adapters import (adapter_chain, chat_template_for,
                                                recorded_chat_template, resolve_base,
                                                tokenizer_source)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_a_chain_is_split_stripped_and_emptied_of_blanks():
    assert adapter_chain("a,b") == ["a", "b"]
    assert adapter_chain(" a , b ,") == ["a", "b"]
    assert adapter_chain("") == [] and adapter_chain(None) == []


def test_an_adapter_dir_handed_over_as_the_model_resolves_to_its_base(tmp_path):
    """An adapter directory given as the model, with no adapter named,
    resolves to (its base, itself)."""
    d = tmp_path / "final"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-4B"}))
    assert resolve_base(str(d)) == ("Qwen/Qwen3.5-4B", str(d))


def test_an_explicit_chain_beats_the_sniff(tmp_path):
    """When a chain is named, the model path is not inspected and both are
    returned unchanged."""
    d = tmp_path / "final"
    d.mkdir()
    (d / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "somewhere/else"}))
    assert resolve_base(str(d), "a,b") == (str(d), "a,b")


def test_a_plain_model_is_left_alone():
    assert resolve_base("Qwen/Qwen3.5-4B") == ("Qwen/Qwen3.5-4B", None)


def test_the_tokenizer_comes_from_the_last_link(tmp_path):
    """The tokenizer comes from the last adapter in the chain, whose
    directory carries the tokenizer and chat template its training run saved."""
    last = tmp_path / "r2"
    last.mkdir()
    (last / "tokenizer_config.json").write_text("{}")
    assert tokenizer_source("Qwen/Qwen3.5-4B", f"{tmp_path/'r1'},{last}") == str(last)


def test_a_tokenizerless_link_falls_back_to_the_base(tmp_path):
    """An adapter directory with no tokenizer files falls back to the base
    model's tokenizer."""
    last = tmp_path / "pulled"
    last.mkdir()
    assert tokenizer_source("Qwen/Qwen3.5-4B", str(last)) == "Qwen/Qwen3.5-4B"


def test_no_adapter_means_the_model_itself():
    assert tokenizer_source("Qwen/Qwen3.5-4B") == "Qwen/Qwen3.5-4B"


def test_a_hub_id_is_not_mistaken_for_a_tokenizerless_dir():
    """The fallback applies only to a local directory, so a hub id passes
    through unchanged."""
    assert tokenizer_source("base/model", "org/adapter") == "org/adapter"


# ------------------------------------------------------------- one owner

def test_nothing_else_in_the_package_sniffs_adapter_config():
    """Only `adapters.py` holds the string constant `adapter_config.json`.
    Matched on the AST by exact value, so a mention in a docstring or comment
    does not count."""
    offenders = []
    for f in sorted((ROOT / "src/graph_reasoner").rglob("*.py")):
        if f.name == "adapters.py":
            continue
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Constant) and n.value == "adapter_config.json":
                offenders.append(f"{f.relative_to(ROOT)}:{n.lineno}")
    assert not offenders, "adapter resolution re-derived:\n  " + "\n  ".join(offenders)


@pytest.mark.parametrize("mod", ["eval/local_model.py", "agent/eval.py"])
def test_the_resolution_users_go_through_the_owner(mod):
    src = (ROOT / "src/graph_reasoner" / mod).read_text()
    assert "train.adapters" in src, \
        f"{mod} resolves adapters without importing the owner"


def _run(tmp_path, name, record=True, tokenizer=False):
    """A run directory with a `final/` adapter: its run.json recording
    plain_chatml when `record`, and tokenizer files in final/ when `tokenizer`."""
    final = tmp_path / name / "final"
    final.mkdir(parents=True)
    (final / "adapter_config.json").write_text("{}")
    if record:
        (tmp_path / name / "run.json").write_text(
            json.dumps({"schema": "recipe_run_v1", "hyperparameters": {"chat_template": "plain_chatml"}}))
    if tokenizer:
        (final / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{ own }}"}))
    return str(final)


def test_the_template_comes_from_the_last_adapter_s_record(tmp_path):
    hosted = _run(tmp_path, "hosted")                        # a record, no tokenizer: the 4B's converted adapters
    assert recorded_chat_template(hosted) == "plain_chatml"
    assert chat_template_for("Qwen/Qwen3.5-4B", hosted, None) == "plain_chatml"
    assert chat_template_for("Qwen/Qwen3.5-4B", hosted, "plain_chatml") == "plain_chatml"
    first = _run(tmp_path, "first", record=False, tokenizer=True)
    assert chat_template_for("Qwen/Qwen3.5-4B", f"{first},{hosted}", None) == "plain_chatml", "the last link's record"


def test_a_flag_that_contradicts_the_record_is_refused(tmp_path):
    hosted = _run(tmp_path, "hosted")
    with pytest.raises(SystemExit, match="trained under the chat template 'plain_chatml'.*names 'other'"):
        chat_template_for("Qwen/Qwen3.5-4B", hosted, "other")


def test_an_adapter_without_a_record_keeps_its_own_tokenizer_s_template(tmp_path):
    local = _run(tmp_path, "local", record=False, tokenizer=True)
    assert recorded_chat_template(local) is None
    assert chat_template_for("Qwen/Qwen3.5-4B", local, None) is None
    assert chat_template_for("Qwen/Qwen3.5-4B", local, "plain_chatml") == "plain_chatml"


def test_an_adapter_with_neither_record_nor_tokenizer_needs_the_flag(tmp_path):
    bare = _run(tmp_path, "bare", record=False)
    with pytest.raises(SystemExit, match="neither a run.json nor a tokenizer"):
        chat_template_for("Qwen/Qwen3.5-4B", bare, None)
    assert chat_template_for("Qwen/Qwen3.5-4B", bare, "plain_chatml") == "plain_chatml"


def test_a_base_model_and_a_hub_adapter_are_left_to_the_flag():
    assert chat_template_for("Qwen/Qwen3.5-4B", None, None) is None
    assert chat_template_for("Qwen/Qwen3.5-4B", None, "plain_chatml") == "plain_chatml"
    assert chat_template_for("Qwen/Qwen3.5-4B", "org/adapter", None) is None


def test_the_run_record_names_the_recorded_template(tmp_path):
    from graph_reasoner.agent.eval import _effective_chat_template
    hosted = _run(tmp_path, "hosted")
    assert _effective_chat_template("hf:Qwen/Qwen3.5-4B", None, adapter=hosted) == "plain_chatml"
