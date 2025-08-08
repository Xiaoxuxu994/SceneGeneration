import os

def generate_hostfile():
    # 从环境变量或云API获取节点IP列表（示例：AWS EKS）
    node_ips = os.getenv("MASTER_ADDR", "").split(",")  # 格式: "10.0.1.2,10.0.2.3"
    gpus_per_node = int(os.getenv("NGPUS_PER_NODE", "8"))
    
    with open("deepspeed_hostfile", "w") as f:
        for ip in node_ips:
            f.write(f"{ip} slots={gpus_per_node}\n")
    
    print("Generated hostfile:")
    print(open("deepspeed_hostfile").read())

if __name__ == "__main__":
    generate_hostfile()