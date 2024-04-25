# Run Local LLama-3 Guide

1. Make sure `gpt-query` and `rl_env` branch of `OpenELM` are fully updated.

```bash
cd gpt-query
git pull
cd ../OpenELM
git checkout rl_env
git pull
```

2. Pip install vllm

`pip install vllm`

3. Download desired Llama-3 from huggingface

Note: this step requires you to fill out a form on HF and wait for approval (takes ~1 hour)

```python
from transformers import AutoModelForCausalLM
model_path = "casperhansen/llama-3-70b-instruct-awq" if 70B else "meta-llama/Meta-Llama-3-8B-Instruct"  # Use quantized 70B version
AutoModelForCausalLM.from_pretrained(model_path)
```

4. Spin up server hosting Llama-3

See [here](https://github.com/Dahoas/gpt-query/blob/master/projects/vllm_examples/scripts/local_serve.sh) for an example script hosting quantized Llama-3 70B. 
This can be run as `bash scripts/local_serve.sh &> llama_logs.txt &`

**IMPORTANT**: Run `echo hostname > hostname.txt` to get the endpoint. This is how the FunSearch process communicates with the Llama-3 server.

5. Set correct model name and endpoint in `run_fun_search.py`

For example, see lines [31-32](https://github.com/Dahoas/OpenELM/blob/c7fe3592652957590938e6a310b2ba4fc4eb796c/projects/fun_search/run_fun_search.py#L31).

6. Run FunSearch

`python run_fun_search.py`
