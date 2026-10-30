# RAGAS-demo

A small terminal demo for the Project Hephaestus presentation: retrieve fictional
policy text, generate an answer with Amazon Bedrock, and evaluate it with Ragas.

**One script. Five short documents. Three questions. No database or web app.**
The documents, customer ID and screening result are invented for this demo and
must not be treated as real compliance policy.

## 1. Preview immediately (no credentials or packages)

Requires Python 3.11 or 3.12; tested locally with Python 3.12.

```bash
git clone https://github.com/Gaut-raj/RAGAS-demo.git
cd RAGAS-demo
python demo.py
```

This prints the retrieved document IDs for each question. It does **not** generate
answers or pretend to produce Ragas scores.

## 2. Run the live evaluation

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Authenticate with your normal AWS profile/SSO setup. Use a Bedrock chat model or
inference profile available to your account that supports tool/structured output
through LiteLLM. Model availability and IAM permissions depend on your account.
The same model generates answers and judges them, keeping setup small.

macOS/Linux:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION_NAME=us-east-1
export BEDROCK_MODEL='bedrock/YOUR_ENABLED_MODEL_OR_INFERENCE_PROFILE_ID'
python demo.py --live
```

Windows PowerShell:

```powershell
$env:AWS_PROFILE = 'your-profile'
$env:AWS_REGION_NAME = 'us-east-1'
$env:BEDROCK_MODEL = 'bedrock/YOUR_ENABLED_MODEL_OR_INFERENCE_PROFILE_ID'
python demo.py --live
```

Replace the model placeholder with your enabled model ID. No OpenAI API key is
needed. Use `aws sso login --profile your-profile` first if your profile uses SSO.
Live runs send the bundled synthetic text to Bedrock and incur model charges.

## What to show in about five minutes

1. Open `data.json`: five fictional policies and three hand-authored questions
   with reference answers. Each question needs two documents.
2. Run `python demo.py` to show the simple keyword retriever. Baseline retrieves
   two documents; degraded retrieves only one. The prompt and model stay fixed.
3. Run `python demo.py --live` and compare the printed mean scores:
   - **Context Precision:** are useful chunks ranked first?
   - **Context Recall:** do the chunks cover the reference answer?
   - **Faithfulness:** does the answer stay grounded in the supplied chunks?
4. Show the **supported clearance** custom rule on two fixed answers to a
   synthetic PENDING screening result. Expected labels are `pass` and `fail`;
   the table displays the judge's actual verdicts.
5. Open `results.json` to inspect individual answers, contexts and scores.

The expected change is **lower Context Recall** for the degraded retriever.
Context Precision can stay high because its single document is relevant.
Faithfulness can also stay high if the answer honestly says evidence is missing.
Judge outputs vary: inspect the cases if the expected difference does not appear.
There are no fabricated scores or assertions that every degraded metric must fall.

Ragas 0.4.3 uses `DiscreteMetric` for this Aspect Critic-style binary domain rule.
The demo deliberately omits embeddings, Response Relevancy, synthetic testset
generation and agent traces to keep the first walkthrough small.

## Offline fallback

After one successful live run:

```bash
python demo.py --replay results.json
```

Replay uses only Python and makes no model calls. The file includes the model ID,
Ragas version, dataset hash, timestamp and elapsed time. Failed live runs do not
overwrite an existing fallback. Results stay local and are ignored by Git.
No recorded Bedrock results ship with this repository: run once with your account
before presenting. Without that file, the no-credential retrieval preview remains
available, but is not a scored evaluation fallback.

## Tests

```bash
python -m unittest -v
```

Tests cover retrieval coverage, offline replay, failure handling and the live
orchestration using test doubles. With dependencies installed, they also check
the pinned Ragas metric APIs. They make no AWS calls. Six generated answers and
three metrics per answer can still require many judge calls; measure runtime and
cost with your selected model before presenting. An eight-minute live run is not
guaranteed.

## Files

- `demo.py`: retrieval, generation, evaluation and replay.
- `data.json`: fictional policy documents and reviewed-for-demo reference answers.
- `requirements.txt`: pinned direct dependencies.
- `test_demo.py`: offline tests.

## References

- [Ragas 0.4.3 Bedrock setup](https://docs.ragas.io/en/v0.4.3/howtos/customizations/customize_models/#aws-bedrock)
- [Ragas metric catalog](https://docs.ragas.io/en/v0.4.3/concepts/metrics/available_metrics/)
- [v0.4 migration and custom metrics](https://docs.ragas.io/en/v0.4.3/howtos/migrations/migrate_from_v03_to_v04/)
