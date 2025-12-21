import numpy as np
import torch
from typing import Union, Callable

# 假设你的 GPTMusicEvaluator 保存在 .transformer.gpt_evaluator
from transformer.gpt_evaluator import GPTMusicEvaluator
def create_gpt_objective(
    model_source: Union[str, GPTMusicEvaluator], 
    device: str = None,
    mode: str = "gaussian",  # 模式: 'linear', 'gaussian', 'inverse'
    target_loss: float = 2.0, # 预期的最佳 Loss (高斯模式的峰值)
    tolerance: float = 1.0    # 容忍度 (高斯模式的方差 / 线性模式的范围)
) -> Callable[[np.ndarray], float]:
    """
    创建一个归一化后的 GPT 评分函数。
    
    Args:
        model_source: 模型路径或实例。
        mode: 归一化模式。
            - 'inverse': 旧模式 1/loss (不推荐，无上限)。
            - 'linear': 线性映射到 [0, 1]。小于 min_loss 得 1，大于 max_loss 得 0。
            - 'gaussian': 【推荐】钟形曲线。只有 Loss 接近 target_loss 时得 1，太高或太低都得 0。
                          这能完美解决 "Loss过低是偷懒" 的问题。
        target_loss: 
            - Linear模式: 定义 'min_loss' (得到1.0分的阈值)。
            - Gaussian模式: 定义 'mu' (钟形曲线的中心，得到1.0分的位置)。
        tolerance:
            - Linear模式: 定义范围 (max_loss = target_loss + tolerance)。
            - Gaussian模式: 定义 'sigma' (曲线宽度)。
    """
    
    # 1. 初始化模型
    if isinstance(model_source, str):
        evaluator = GPTMusicEvaluator(model_source, device=device)
    elif isinstance(model_source, GPTMusicEvaluator):
        evaluator = model_source
    else:
        raise ValueError("Invalid model source")

    # 2. 定义不同的归一化逻辑
    
    def _score_linear(loss):
        """线性映射: Loss <= 2.0 得满分, Loss >= 5.0 得0分"""
        min_loss = target_loss
        max_loss = target_loss + tolerance
        
        # 归一化公式: (max - x) / (max - min)
        score = (max_loss - loss) / (max_loss - min_loss)
        return np.clip(score, 0.0, 1.0)

    def _score_gaussian(loss):
        """
        高斯映射: 奖励 '恰到好处' 的 Loss。
        如果 Loss 太高(烂旋律) -> 分低
        如果 Loss 太低(偷懒的全长音) -> 分低 (惩罚!)
        只有 Loss 在 target_loss 附近 -> 分高
        """
        # 公式: exp( - (x - mu)^2 / (2 * sigma^2) )
        numerator = (loss - target_loss) ** 2
        denominator = 2 * (tolerance ** 2)
        return np.exp(- numerator / denominator)

    def _score_inverse(loss):
        """旧模式: 1/x (无界，危险)"""
        return 1.0 / (loss + 1e-6)

    # 选择策略
    if mode == "linear":
        calc_score = _score_linear
    elif mode == "gaussian":
        calc_score = _score_gaussian
    else:
        calc_score = _score_inverse

    # 3. 闭包函数
    def gpt_score_func(sequence_data: np.ndarray) -> float:
        if isinstance(sequence_data, np.ndarray):
            seq_list = sequence_data.tolist()
        else:
            seq_list = list(sequence_data)
            
        # 这里我们需要 Evaluator 提供原始 Loss，而不是 1/Loss
        # 我们可以调用 evaluate 方法并反推，或者修改 evaluator 增加 get_loss 接口
        # 为了不改 evaluator 代码，我们临时反算一下: fitness = 1/loss -> loss = 1/fitness
        fitness = evaluator.get_fitness_score(seq_list) 
        
        # 处理 loss 为 0 或 极小的情况
        if fitness > 1e6: 
            raw_loss = 0.0
        else:
            raw_loss = 1.0 / (fitness + 1e-9)

        # 应用归一化
        final_score = calc_score(raw_loss)
        return float(final_score)

    return gpt_score_func