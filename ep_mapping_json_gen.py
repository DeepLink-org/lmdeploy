import numpy as np
import json

# 配置参数
num_experts = 256
layers = 12  # 层数
np.random.seed(42)  # 保持可复现

# 随机生成 weight：形状为 [layers, num_experts]，每个专家处理 80~160 个 token
weight = np.random.randint(low=80, high=161, size=(layers, num_experts)).tolist()

# 构造数据
data = {
    "num_groups": 4,
    "num_nodes": 1,
    "weight": weight  # weight 的形状为 [layers, num_experts]
}

# 写入 JSON 文件
output_path = "/opt/workspace/workspace/lmdeploy_internLM/lmdeploy/ep_mapping_json_decode.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

# 打印信息
print("JSON 写入完成, weight 总和 =", np.sum(weight))