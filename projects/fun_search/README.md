# Evolving RL Policies as Code

This is branch of the OpenELM repo explores using LLMs as policy improvement operators over rl policies represented as code.

## Installation

0. Clone the repo and checkout the `rl_env` branch:

```bash
git clone https://github.com/Dahoas/OpenELM
git checkout rl_env
```

2. Create a python venv to install OpenELM and pytorch (assumes CUDA version >= 12.1):

```bash
python -m venv elm
source elm/bin/activate
pip install torch torchvision torchaudio
```

2. Install the gpt-query repository locally:

```bash
git clone https://github.com/Dahoas/gpt-query
cd gpt-query
pip install -e .
```

3. Install OpenELM with `rl_env` requirements:

```bash
cd ~/PATH_TO_OpenELM/OpenELM
pip install -e .
```

4. Set your OPENAI_API key and test the installation:

```bash
export OPENAI_API=YOUR_API_KEY
python run_fun_search.py
```

Results for the current ELM run are tracked under `logs/elm/RUN_TIME/database.jsonl`.
