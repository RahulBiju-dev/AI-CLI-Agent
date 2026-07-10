import unittest
from unittest.mock import patch

from agent.core import (
    ContextWindowError,
    TOOL_CONTINUATION_PROMPT,
    _check_and_compact_history,
    _context_safety_margin,
    _estimate_messages_tokens,
    _estimate_tool_schema_tokens,
    _estimate_tokens,
    guarded_options_for_call,
    load_default_system_prompt,
    prepare_messages_for_model,
    tool_schemas_for_model,
    validate_session_options,
)
from agent.runtime_config import RuntimeConfigurationError
from tools.registry import TOOL_SCHEMAS


class TestContextBudget(unittest.TestCase):
    def test_low_vram_first_turn_with_relevant_tools_fits_before_generation(self):
        session = {"runtime_profile": "low-vram", "options": {}}
        messages = [
            {"role": "system", "content": load_default_system_prompt()},
            {"role": "user", "content": "Open Spotify and play a song"},
        ]

        tools = tool_schemas_for_model(messages, session, TOOL_SCHEMAS)
        prepared = prepare_messages_for_model(messages, session, TOOL_SCHEMAS)
        options = guarded_options_for_call(
            prepared,
            {"num_ctx": 4096, "num_predict": 768},
            tools,
        )
        names = {tool["function"]["name"] for tool in tools}
        projected = (
            _estimate_messages_tokens(prepared)
            + _estimate_tool_schema_tokens(tools)
            + _context_safety_margin(4096)
            + options["num_predict"]
        )

        self.assertLessEqual(len(tools), 10)
        self.assertIn("spotify_play", names)
        self.assertLessEqual(projected, 4096)

    def test_unicode_token_estimate_is_not_ascii_underweighted(self):
        self.assertGreaterEqual(_estimate_tokens("漢" * 100), 100)

    def test_tool_continuation_tail_remains_atomic_when_trimming(self):
        tool_call = {
            "role": "assistant",
            "content": "discarded thinking text",
            "tool_calls": [
                {"function": {"name": "read_file", "arguments": {"file_path": "README.md"}}}
            ],
        }
        tool_result = {
            "role": "tool",
            "tool_name": "read_file",
            "name": "read_file",
            "content": "bounded result",
        }
        reminder = {
            "role": "user",
            "content": TOOL_CONTINUATION_PROMPT.format(user_input="Explain the project"),
        }
        messages = [
            {"role": "system", "content": "system policy"},
            {"role": "user", "content": "old " * 4000},
            {"role": "assistant", "content": "old response " * 4000},
            tool_call,
            tool_result,
            reminder,
        ]

        prepared = prepare_messages_for_model(
            messages,
            {"options": {"num_ctx": 2048, "num_predict": 256}},
        )

        self.assertEqual(prepared[-3]["tool_calls"], tool_call["tool_calls"])
        self.assertEqual(prepared[-2], tool_result)
        self.assertEqual(prepared[-1], reminder)

    def test_num_predict_is_capped_to_remaining_context(self):
        messages = [{"role": "user", "content": "x" * 5000}]

        options = guarded_options_for_call(
            messages,
            {"num_ctx": 2048, "num_predict": 1400},
        )

        self.assertLess(options["num_predict"], 1400)
        self.assertGreaterEqual(options["num_predict"], 96)

    def test_unavoidable_context_overflow_is_controlled(self):
        messages = [{"role": "system", "content": "x" * 12000}]

        with self.assertRaises(ContextWindowError):
            guarded_options_for_call(messages, {"num_ctx": 1024, "num_predict": 256})

    def test_invalid_session_parameter_is_rejected(self):
        with self.assertRaises(RuntimeConfigurationError):
            validate_session_options({"num_ctx": 512})

    def test_history_compaction_does_not_start_secondary_ollama_call(self):
        history = [{"role": "system", "content": "policy"}]
        for index in range(5):
            history.extend([
                {"role": "user", "content": f"request {index} " + ("x" * 1200)},
                {"role": "assistant", "content": f"answer {index} " + ("y" * 1200)},
            ])
        session = {"options": {"num_ctx": 2048}}

        with patch("ollama.chat", side_effect=AssertionError("unexpected model call")):
            _check_and_compact_history(history, session)

        self.assertNotIn("_is_compacting", session)
        self.assertEqual(history[0]["role"], "system")
        self.assertEqual([m["content"] for m in history if m.get("role") == "user"][-2:], [
            "request 3 " + ("x" * 1200),
            "request 4 " + ("x" * 1200),
        ])


if __name__ == "__main__":
    unittest.main()
