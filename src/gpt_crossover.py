import numpy as np
import random
from typing import Tuple, List

# 导入框架接口
from GA.ga_framework import CrossoverStrategy, Individual
from transformer.gpt_evaluator import GPTMusicEvaluator

# 假设你的个体类是 MusicIndividual
# 为了类型提示方便，这里引用一下，实际运行时动态传入即可
try:
    from main_modular import MusicIndividual
except ImportError:
    pass # 忽略导入错误，只要接口一致即可

class StructureAwareCrossover(CrossoverStrategy):
    """
    【结构感知交叉】
    只在音乐结构的关键节点（如小节线）进行切分和拼接。
    防止在切分音中间打断节奏。
    """
    def __init__(self, cut_points: List[int] = [8, 16, 24]):
        """
        cut_points: 允许切分的位置索引。
        对于 32 个音符 (4/4拍, 4小节), 推荐 [8, 16, 24]。
        """
        self.cut_points = cut_points

    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        p1_data = parent1.data
        p2_data = parent2.data
        
        # 随机选择一个合法的切分点
        # 如果序列太短无法切分，回退到中间切分
        valid_points = [cp for cp in self.cut_points if cp < len(p1_data)]
        if not valid_points:
            point = len(p1_data) // 2
        else:
            point = random.choice(valid_points)
        
        # 物理拼接
        c1_data = np.concatenate([p1_data[:point], p2_data[point:]])
        c2_data = np.concatenate([p2_data[:point], p1_data[point:]])
        
        # 返回新个体 (使用父类的 class 来实例化，保持多态)
        # 假设 Individual 的子类构造函数接收 data
        return parent1.__class__(c1_data), parent2.__class__(c2_data)


class GPTLogitMixingCrossover(CrossoverStrategy):
    """
    【GPT 软引导交叉】(Logit Mixing)
    使用 GPT 生成 Child 的后半段。
    生成时，将 Parent B 的音符作为 '建议' (Logit Bias) 注入到模型中。
    结果：Child 听起来像 Parent A 的自然延续，但旋律走向模仿 Parent B。
    """
    def __init__(self, model: GPTMusicEvaluator, alpha: float = 4.0, fixed_split: int = None):
        """
        alpha: 混合强度。越高越像 Parent B，越低越像 GPT 自由发挥。建议 3.0 - 5.0。
        fixed_split: 如果指定，总是从该位置开始生成。如果不指定，随机在 [8, 16, 24] 选择。
        """
        self.model = model
        self.alpha = alpha
        self.fixed_split = fixed_split

    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        p1_data = parent1.data
        p2_data = parent2.data
        seq_len = len(p1_data)
        
        # 1. 确定切分点
        if self.fixed_split:
            split_idx = self.fixed_split
        else:
            # 随机选择 [8, 16, 24] 中的一个，或者随机范围
            # 建议保留足够的 Context (至少 8)
            split_idx = random.choice([8, 16, 24])
            if split_idx >= seq_len: split_idx = seq_len // 2

        # 2. 生成 Child 1: Context=A, Target=B
        # 这一步比较耗时，但能保证连贯性
        try:
            c1_list = self.model.generate_with_logit_bias(
                prompt_sequence=p1_data[:split_idx].tolist(),
                target_sequence=p2_data.tolist(),
                start_idx=split_idx,
                alpha=self.alpha
            )
            c1_data = np.array(c1_list[:seq_len])
        except Exception as e:
            print(f"[GPTCrossover] Error: {e}, fallback to simple splice.")
            c1_data = np.concatenate([p1_data[:split_idx], p2_data[split_idx:]])

        # 3. 生成 Child 2: Context=B, Target=A
        try:
            c2_list = self.model.generate_with_logit_bias(
                prompt_sequence=p2_data[:split_idx].tolist(),
                target_sequence=p1_data.tolist(),
                start_idx=split_idx,
                alpha=self.alpha
            )
            c2_data = np.array(c2_list[:seq_len])
        except:
            c2_data = np.concatenate([p2_data[:split_idx], p1_data[split_idx:]])

        return parent1.__class__(c1_data), parent2.__class__(c2_data)


class CompositeCrossover(CrossoverStrategy):
    """
    【组合交叉策略】
    允许按概率混合多种交叉方式。
    例如: 70% 概率用结构交叉 (快), 30% 概率用 GPT 交叉 (好)。
    """
    def __init__(self):
        self.strategies = []
        self.weights = []

    def register(self, strategy: CrossoverStrategy, weight: float):
        self.strategies.append(strategy)
        self.weights.append(weight)

    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if not self.strategies:
            return parent1.copy(), parent2.copy()
        
        # 轮盘赌选择策略
        strat = random.choices(self.strategies, weights=self.weights, k=1)[0]
        return strat.cross(parent1, parent2)