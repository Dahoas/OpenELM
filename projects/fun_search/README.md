# Evolving RL Policies as Code

This is branch of the OpenELM repo explores using LLMs as policy improvement operators over rl policies represented as code.

## Installation

0. Create a python venv to install OpenELM and pytorch (assumes python version==3.9, CUDA version >= 12.1). Make sure pip is fully upgraded.

```bash
python -m venv elm
source elm/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
```

1. Install the gpt-query repository locally:

```bash
git clone https://github.com/Dahoas/gpt-query
cd gpt-query
pip install -e .
```

2. Clone the OpenELM repo and install the `rl_env` branch:

```bash
git clone https://github.com/Dahoas/OpenELM
cd OpenELM
git checkout rl_env
pip install -e .
```

3. Set your OPENAI_API key and test the installation:

```bash
export OPENAI_API=YOUR_API_KEY
python run_fun_search.py
```

Results for the current ELM run are tracked under `logs/elm/RUN_TIME/database.jsonl`.
