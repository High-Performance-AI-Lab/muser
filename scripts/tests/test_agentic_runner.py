import importlib.util
import json
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "agentic_live_runner", ROOT / "datasets" / "agentic" / "harness" / "run.py"
)
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'


class AgenticRunnerTests(unittest.TestCase):
    def test_live_turn_has_the_same_bounded_budget_as_native_d2(self) -> None:
        captured = {}

        def open_request(request, timeout):
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            return _Response()

        with mock.patch.object(RUNNER.urllib.request, "urlopen", side_effect=open_request):
            message = RUNNER.chat_completion([{"role": "user", "content": "hi"}], [])
        self.assertEqual(message["content"], "ok")
        self.assertEqual(captured["body"]["temperature"], 0)
        self.assertEqual(captured["body"]["max_tokens"], 512)
        self.assertEqual(captured["timeout"], 120)


if __name__ == "__main__":
    unittest.main()
