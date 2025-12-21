from typing import List
import numpy as np
from .ga_framework import Evaluator, Individual
class RuleBasedEvaluator(Evaluator):
    def evaluate(self, population: List[Individual]) -> np.array:
        scores = []
        for ind in population:
            gene = ind.data
            # 简单的规则：有效音符越多越好，且在 C 大调音阶内越好
            active_notes = np.sum(gene > 1) # 假设 0=Rest, 1=Hold
            if active_notes == 0:
                scores.append(0.0)
                continue
            
            # C Major Scale: 0, 2, 4, 5, 7, 9, 11 (mod 12)
            c_scale = {0, 2, 4, 5, 7, 9, 11}
            in_key = sum(1 for x in gene if x > 1 and (x % 12) in c_scale)
            
            score = (active_notes / 32.0) * 0.4 + (in_key / active_notes) * 0.6
            scores.append(score)
        return np.array(scores)

class BasicRules:
    """
    一些通用的乐理规则函数，可以直接添加到 RuleBasedEvaluator 中。
    """
    
    @staticmethod
    def pitch_in_key_c_major(grid: np.ndarray) -> float:
        """奖励 C 大调音阶内的音符"""
        # C Major: 0, 2, 4, 5, 7, 9, 11
        scale_set = {0, 2, 4, 5, 7, 9, 11}
        notes = grid[grid > 1] # 排除 Rest(0) 和 Hold(1)
        if len(notes) == 0: return 0.0
        
        count = sum(1 for n in notes if (n % 12) in scale_set)
        return count / len(notes)

    @staticmethod
    def rhythmic_variety(grid: np.ndarray) -> float:
        """奖励节奏变化，惩罚过多休止或全屏延音"""
        # 计算非 Hold 的事件数量 (即 Note On 或 Rest)
        changes = np.sum(grid != 1)
        # 我们希望变化率适中，比如占 20%~80%
        ratio = changes / len(grid)
        if 0.2 <= ratio <= 0.8:
            return 1.0
        else:
            return 0.2

    @staticmethod
    def smooth_contour(grid: np.ndarray) -> float:
        """奖励平滑的音程进行，惩罚大跳"""
        notes = grid[grid > 1]
        if len(notes) < 2: return 0.0
        
        jumps = 0
        for k in range(len(notes) - 1):
            diff = abs(notes[k] - notes[k+1])
            if diff > 7: # 大于纯五度
                jumps += 1
        
        # 跳跃越少分越高
        return max(0, 1.0 - (jumps * 0.1))

