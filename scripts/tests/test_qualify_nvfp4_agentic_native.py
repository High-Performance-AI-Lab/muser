import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "qualify_nvfp4_agentic_native",
    ROOT / "scripts" / "qualify_nvfp4_agentic_native.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class NativeAgenticQualificationTests(unittest.TestCase):
    def test_parses_reasoning_tool_call_and_scalars(self) -> None:
        message = MODULE.parse_atem(
            " to=self<|message|>think<|eom|>"
            "<|start|>assistant to=calc.eval<|message|>"
            "<atem:function_calls><atem:invoke name=\"calc.eval\">"
            "<atem:parameter name=\"expression\">2+2</atem:parameter>"
            "<atem:parameter name=\"rounded\">true</atem:parameter>"
            "</atem:invoke></atem:function_calls><|eot|>"
        )
        self.assertEqual(message["content"], "")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "calc.eval")
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            '{"expression":"2+2","rounded":true}',
        )

    def test_parses_plain_user_content(self) -> None:
        message = MODULE.parse_atem(" to=user<|message|>ANSWER: 4<|eot|>")
        self.assertEqual(message["content"], "ANSWER: 4")
        self.assertEqual(message["tool_calls"], [])

    def test_template_converts_arguments_to_mapping_without_mutation(self) -> None:
        original = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "x", "arguments": '{"a":1}'}}
                ],
            }
        ]
        converted = MODULE.template_messages(original)
        self.assertEqual(converted[0]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertIsInstance(
            original[0]["tool_calls"][0]["function"]["arguments"], str
        )


if __name__ == "__main__":
    unittest.main()
