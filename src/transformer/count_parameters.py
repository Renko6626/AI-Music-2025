import os
from typing import Any, Dict

import torch


def _extract_state_dict(obj: Dict[str, Any]) -> Dict[str, Any] | None:
    """从常见 checkpoint 字段里提取 state_dict。"""
    for key in ("state_dict", "model_state_dict", "module", "model"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value
    return None


def _count_state_dict_tensors(state_dict: Dict[str, Any]) -> int:
    """统计 state_dict 中所有张量的参数数量。"""
    return sum(t.numel() for t in state_dict.values() if torch.is_tensor(t))


def count_model_parameters(model_path: str) -> int:
    """统计 .pth 文件中的参数数量，兼容多种保存格式。"""
    checkpoint = torch.load(model_path, map_location="cpu")

    if hasattr(checkpoint, "parameters"):
        return sum(p.numel() for p in checkpoint.parameters())

    if isinstance(checkpoint, dict):
        state_dict = _extract_state_dict(checkpoint)
        if state_dict is not None:
            return _count_state_dict_tensors(state_dict)

        # 直接就是 state_dict 的情况
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return _count_state_dict_tensors(checkpoint)

    raise ValueError(f"不支持的 checkpoint 格式: {type(checkpoint)}")


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints_gpt"

    if os.path.isdir(target):
        for filename in os.listdir(target):
            if filename.endswith(".pth"):
                model_path = os.path.join(target, filename)
                try:
                    param_count = count_model_parameters(model_path)
                    print(f"模型文件: {filename}, 参数数量: {param_count/1000000:.2f}M")
                except Exception as exc:  # noqa: BLE001
                    print(f"模型文件: {filename}, 统计失败: {exc}")
    else:
        param_count = count_model_parameters(target)
        print(f"模型文件: {os.path.basename(target)}, 参数数量: {param_count}")
