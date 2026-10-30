"""Small Ragas/Bedrock demo. All included policy text is fictional."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import mean
import time

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.json"
STOP = set("a an the and or is are what how for to of in before with required applies".split())
SYSTEM = (
    "Answer the question using only the supplied fictional policies. "
    "Cite document IDs in brackets. If evidence is missing, say so. "
    "Never invent a customer screening result or clearance decision. Be concise."
)


def load_data():
    return json.loads(DATA.read_text(encoding="utf-8"))


def tokens(text):
    return set(re.findall(r"[a-z]+", text.lower())) - STOP


def retrieve(question, documents, top_k):
    """Tiny keyword retriever. Reference answers never enter retrieval."""
    ranked = sorted(documents, key=lambda d: len(tokens(question) & tokens(d["text"])), reverse=True)
    return ranked[:top_k]


def preview(data):
    print("OFFLINE PREVIEW: retrieval only; no generated answers or Ragas scores.\n")
    for case in data["questions"]:
        print(f'{case["id"]}: {case["question"]}')
        for name, k in [("baseline", 2), ("degraded", 1)]:
            docs = retrieve(case["question"], data["documents"], k)
            print(f'  {name:9} top_k={k}: {", ".join(d["id"] for d in docs)}')


def make_metrics(llm):
    from ragas.metrics import DiscreteMetric
    from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

    rule = DiscreteMetric(
        name="supported_clearance",
        allowed_values=["pass", "fail"],
        prompt=(
            "Evaluate the response using the screening evidence. "
            "Return pass if there is no customer-clearance claim, or every such claim "
            "cites a supplied screening result for the same customer whose status is CLEAR. "
            "Return fail for unsupported clearance, fabricated citations, wrong customers, "
            "or pending/contradictory results. A policy document is not a screening result.\n"
            "Evidence: {evidence}\nResponse: {response}"
        ),
    )
    return ContextPrecision(llm=llm), ContextRecall(llm=llm), Faithfulness(llm=llm), rule


async def run_live(data, model):
    # These imports are deliberately lazy: preview and replay need only Python.
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    import litellm
    from ragas.llms import llm_factory

    llm = llm_factory(model, provider="litellm", client=litellm.completion,
                      temperature=0, timeout=60, max_retries=1)
    precision, recall, faithfulness, rule = make_metrics(llm)
    started = time.monotonic()
    rows = []
    for case in data["questions"]:
        for variant, top_k in [("baseline", 2), ("degraded", 1)]:
            docs = retrieve(case["question"], data["documents"], top_k)
            contexts = [f'[{d["id"]}] {d["text"]}' for d in docs]
            print(f'Running {case["id"]} / {variant}...', flush=True)
            answer = await litellm.acompletion(
                model=model, temperature=0, timeout=60, num_retries=1,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": "\n\n".join(contexts) + "\n\nQuestion: " + case["question"]}],
            )
            response = answer.choices[0].message.content
            if not response:
                raise ValueError("Bedrock returned an empty answer.")
            inputs = dict(user_input=case["question"], retrieved_contexts=contexts)
            scores = {
                "Context Precision": (await precision.ascore(**inputs, reference=case["reference"])).value,
                "Context Recall": (await recall.ascore(**inputs, reference=case["reference"])).value,
                "Faithfulness": (await faithfulness.ascore(**inputs, response=response)).value,
            }
            rows.append(dict(id=case["id"], variant=variant, top_k=top_k,
                             question=case["question"], contexts=contexts,
                             reference=case["reference"], response=response, scores=scores))
    # Deliberately controlled answers isolate the domain rule from generation.
    evidence = "Screening result S-001: customer DEMO-001, status PENDING."
    checks = []
    for response, expected in [
        ("Customer DEMO-001 remains pending according to screening result [S-001].", "pass"),
        ("Customer DEMO-001 is cleared according to screening result [S-001].", "fail"),
    ]:
        result = await rule.ascore(evidence=evidence, response=response, llm=llm)
        checks.append(dict(response=response, expected=expected, actual=result.value))
    return dict(schema_version=1, mode="live", model=model, ragas_version="0.4.3",
                created_at=datetime.now(timezone.utc).isoformat(),
                data_sha256=hashlib.sha256(DATA.read_bytes()).hexdigest(),
                elapsed_seconds=round(time.monotonic() - started, 1), rows=rows, rule_checks=checks)


def show(result):
    if result.get("schema_version") != 1 or result.get("mode") != "live" or not result.get("rows"):
        raise ValueError("Expected a complete results file saved by this demo's --live run.")
    print(f'\nRecorded live run: {result["created_at"]} | {result["model"]}')
    print(f'Elapsed: {result["elapsed_seconds"]}s | Ragas {result["ragas_version"]}')
    print(f'{"Metric":22} {"Baseline":>10} {"Degraded":>10} {"Delta (D-B)":>12}')
    for name in ["Context Precision", "Context Recall", "Faithfulness"]:
        scores = [mean(row["scores"][name] for row in result["rows"] if row["variant"] == v)
                  for v in ["baseline", "degraded"]]
        print(f'{name:22} {scores[0]:10.3f} {scores[1]:10.3f} {scores[1]-scores[0]:+12.3f}')
    print("\nCustom rule: supported clearance")
    for check in result["rule_checks"]:
        print(f'  Expected {check["expected"]}, judge returned {check["actual"]}: {check["response"]}')
    print("\nInspect individual answers and contexts in the JSON; averages are diagnostic signals.")
    print("Expect lower recall with top_k=1. Precision and faithfulness may stay high.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true", help="Generate and evaluate with Bedrock (billable)")
    modes.add_argument("--replay", type=Path, help="Display a previous live run with no API calls")
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    args = parser.parse_args()
    try:
        if args.replay:
            show(json.loads(args.replay.read_text(encoding="utf-8")))
        elif args.live:
            model = os.environ.get("BEDROCK_MODEL", "")
            if not model.startswith("bedrock/"):
                parser.error("Set BEDROCK_MODEL to bedrock/<your enabled model or inference-profile ID>.")
            result = asyncio.run(run_live(load_data(), model))
            # Save only complete runs. A failed run never replaces a working fallback.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(args.output.suffix + ".tmp")
            temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
            temporary.replace(args.output)
            show(result)
            print(f"Saved {args.output}. Offline fallback: python demo.py --replay {args.output}")
        else:
            preview(load_data())
    except Exception as exc:
        # Avoid logging provider exceptions that may contain request/auth metadata.
        parser.exit(1, f"Demo failed ({type(exc).__name__}). Check dependencies, AWS login, region and model access.\n"
                       "Any previous results file is unchanged. Use --replay results.json for a saved run.\n")


if __name__ == "__main__":
    main()
