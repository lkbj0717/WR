import os
import pickle

# 设置评估结果文件的路径
pkl_file_path = r".\checkpoints\cifar10\eval_result.pkl"

# 检查文件是否存在
if not os.path.exists(pkl_file_path):
    print(f"文件 {pkl_file_path} 不存在，请检查路径是否正确。")
else:
    # 加载评估结果的 pkl 文件
    with open(pkl_file_path, 'rb') as f:
        eval_data = pickle.load(f)

    # 打印 pkl 文件中的数据结构
    if isinstance(eval_data, list):
        print(f"评估结果包含 {len(eval_data)} 条数据")
        # 打印每个数据项的具体结构
        for idx, item in enumerate(eval_data):
            print(f"第 {idx + 1} 条数据: {item}")
    else:
        print("pkl 文件中的数据格式不是列表类型，请检查文件内容。")
