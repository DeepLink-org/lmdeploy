import numpy as np
import json

num_experts = 12
np.random.seed(42)  # 保持可复现

# 随机整数负载：每个专家处理 80~160 个 token
weight = np.random.randint(low=80, high=161, size=num_experts).tolist()

data = {
    "num_groups": 4,
    "num_nodes": 1,
    "weight": weight
}

with open("/nvme1/liudongyan/workspace/lmdeploy/ep_mapping_json_path_logicexp12.json", "w") as f:
    json.dump(data, f, indent=2)
print("JSON 写入完成, weight 总和 =", sum(weight))
