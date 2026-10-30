# RAGAS-demo

## Requirements

- Python 3.11 or 3.12.
- For live evaluation: AWS credentials and access to a Bedrock chat model or inference profile that supports structured output through LiteLLM.
- Use only the bundled fictional data or other synthetic, non-confidential data.

## Download

```bash
git clone https://github.com/Gaut-raj/RAGAS-demo.git
cd RAGAS-demo
```

## Preview without AWS

```bash
python demo.py
```

This shows retrieved document IDs only. No dependencies, credentials or model calls are required.

## Install dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Or activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

## Configure AWS

Replace the profile, region and model placeholders with values for your AWS account.

macOS/Linux:

```bash
export AWS_PROFILE=your-profile
export AWS_REGION_NAME=us-east-1
export BEDROCK_MODEL='bedrock/YOUR_ENABLED_MODEL_OR_INFERENCE_PROFILE_ID'
```

Windows PowerShell:

```powershell
$env:AWS_PROFILE = 'your-profile'
$env:AWS_REGION_NAME = 'us-east-1'
$env:BEDROCK_MODEL = 'bedrock/YOUR_ENABLED_MODEL_OR_INFERENCE_PROFILE_ID'
```

If your profile uses AWS SSO, sign in first:

```bash
aws sso login --profile your-profile
```

## Run the evaluation

```bash
python demo.py --live
```

Live evaluation incurs Bedrock model charges. Inspect the printed summary and
`results.json` for individual answers, retrieved contexts and scores.

To save to a different file:

```bash
python demo.py --live --output results-run2.json
```

## Replay saved results offline

After completing a successful live run:

```bash
python demo.py --replay results.json
```

Replay requires only Python and makes no model calls. No saved live results are
included in the repository. Failed live runs leave an existing results file unchanged.

## Run tests

With the dependencies installed:

```bash
python -m unittest -v
```

Tests make no AWS calls.

## Troubleshooting

- If live evaluation fails, check your AWS login, region, model ID and Bedrock permissions.
- If replay fails, confirm the path points to a results file from a completed live run.
- To see all command options, run `python demo.py --help`.
