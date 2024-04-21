import json


policy_path = "/storage/home/hcoda1/6/ahavrilla3/p-wliao60/alex/repos/OpenELM/projects/fun_search/init_policies/door_key/offline_analysis/policy_1.py"
with open(policy_path, "r") as f:
    lines = f.readlines()
src = "\n".join(lines)
policy = {"src": src}

jsonl_path = "/storage/home/hcoda1/6/ahavrilla3/p-wliao60/alex/repos/OpenELM/projects/fun_search/init_policies/door_key/jsonls/init.jsonl"
with open(jsonl_path, "w") as f:
    json.dump(policy, f)