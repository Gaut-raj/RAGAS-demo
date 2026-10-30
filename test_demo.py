"""Offline tests. No model credentials required."""
import asyncio
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import demo


def fixture():
    # Test-only fake data, never shipped or displayed as a recorded live result.
    return dict(schema_version=1, mode="live", created_at="test", model="test-double",
                elapsed_seconds=0, ragas_version="0.4.3", rule_checks=[],
                rows=[dict(variant=v, scores={m:1.0 for m in
                      ["Context Precision", "Context Recall", "Faithfulness"]})
                      for v in ["baseline", "degraded"]])


class DemoTests(unittest.TestCase):
    def test_retrieval_degradation_removes_required_evidence(self):
        data = demo.load_data()
        for case in data["questions"]:
            expected = set(case["expected_documents"])
            baseline = {d["id"] for d in demo.retrieve(case["question"], data["documents"], 2)}
            degraded = {d["id"] for d in demo.retrieve(case["question"], data["documents"], 1)}
            self.assertEqual(baseline, expected)
            self.assertEqual(len(degraded & expected), 1)

    def test_default_preview_needs_no_third_party_imports(self):
        result = subprocess.run([sys.executable, "-S", str(demo.ROOT / "demo.py")],
                                capture_output=True, text=True, check=True)
        self.assertIn("OFFLINE PREVIEW", result.stdout)
        self.assertIn("no generated answers or Ragas scores", result.stdout)

    def test_replay_needs_no_third_party_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            path.write_text(json.dumps(fixture()))
            result = subprocess.run([sys.executable, "-S", str(demo.ROOT / "demo.py"),
                                     "--replay", str(path)], capture_output=True, text=True, check=True)
            self.assertIn("Context Recall", result.stdout)
            self.assertIn("test-double", result.stdout)

    def test_invalid_replay_is_rejected(self):
        with self.assertRaises(ValueError):
            demo.show({"schema_version":1, "mode":"illustrative", "rows":[]})

    def test_failed_live_run_preserves_fallback(self):
        async def fail(*args):
            raise RuntimeError("provider failure")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text("existing fallback")
            with patch.dict("os.environ", {"BEDROCK_MODEL":"bedrock/test"}), \
                 patch.object(sys, "argv", ["demo.py", "--live", "--output", str(path)]), \
                 patch.object(demo, "run_live", fail), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    demo.main()
            self.assertEqual(path.read_text(), "existing fallback")

    @unittest.skipUnless(importlib.util.find_spec("ragas"), "Install requirements for Ragas API tests")
    def test_pinned_metric_api_and_orchestration(self):
        import inspect
        from importlib.metadata import version
        from ragas.llms import llm_factory
        import litellm

        self.assertEqual(version("ragas"), "0.4.3")
        llm = llm_factory("bedrock/test", provider="litellm", client=litellm.completion)
        metrics = demo.make_metrics(llm)
        inputs = dict(user_input="q", retrieved_contexts=["c"], reference="r")
        inspect.signature(metrics[0].ascore).bind(**inputs)
        inspect.signature(metrics[1].ascore).bind(**inputs)
        inspect.signature(metrics[2].ascore).bind(user_input="q", retrieved_contexts=["c"], response="a")
        inspect.signature(metrics[3].ascore).bind(evidence="e", response="a", llm=llm)

        calls = []
        async def answer(**kwargs):
            self.assertNotIn("reference", kwargs)
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Test answer"))])

        class FakeMetric:
            async def ascore(self, **kwargs):
                return SimpleNamespace(value=0.5)

        with patch.object(litellm, "acompletion", answer), \
             patch.object(demo, "make_metrics", return_value=[FakeMetric() for _ in range(4)]), \
             contextlib.redirect_stdout(io.StringIO()):
            result = asyncio.run(demo.run_live(demo.load_data(), "bedrock/test"))
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(result["rows"]), 6)
        self.assertEqual(len(result["rule_checks"]), 2)
        self.assertEqual(result["mode"], "live")


if __name__ == "__main__":
    unittest.main()
