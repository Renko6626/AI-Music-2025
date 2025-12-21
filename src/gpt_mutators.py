import numpy as np
import random
import torch
from typing import List, Optional

# 导入基类
from GA.ga_framework import MutationStrategy, Individual
# 导入你的 GPT 封装器类型 (用于类型提示)
from transformer.gpt_evaluator import GPTMusicEvaluator

# 假设有一些全局配置，比如音符范围
# 如果 MusicConfig 不可用，这里定义一些默认值
try:
    from MusicRep import MusicConfig
    PITCH_MIN = MusicConfig.PITCH_MIN
    PITCH_MAX = MusicConfig.PITCH_MAX
    HOLD_VAL = MusicConfig.HOLD_VAL
except ImportError:
    PITCH_MIN = 0
    PITCH_MAX = 127
    HOLD_VAL = 1

class GPTMutationBase(MutationStrategy):
    """GPT 变异基类，持有模型引用"""
    def __init__(self, model: GPTMusicEvaluator):
        self.model = model

    def _create_new_individual(self, original_ind: Individual, new_data: np.ndarray) -> Individual:
        """工厂方法：创建一个与原个体类型相同的新个体"""
        # 使用 original_ind.__class__ 确保返回的是 MusicIndividual 等具体子类
        new_ind = original_ind.__class__(new_data)
        return new_ind

class GPTSuffixMutation(GPTMutationBase):
    """
    【后缀重生成变异】(Suffix Regeneration)
    保留前半段动机，切断后半段，让 GPT 基于前文续写出新的结尾。
    适用于：打破局部最优，寻找全新的乐句发展方向。
    """
    def __init__(self, model: GPTMusicEvaluator, temperature: float = 1.0, min_context: int = 4):
        super().__init__(model)
        self.temp = temperature
        self.min_context = min_context

    def mutate(self, individual: Individual) -> Individual:
        gene = individual.data
        seq_len = len(gene)
        
        # 1. 随机选择切分点 (保留至少 min_context 个音符)
        # 切分点范围: [min_context, seq_len - 4]
        if seq_len <= self.min_context + 4:
            return individual
            
        cut_point = np.random.randint(self.min_context, seq_len - 4)
        prefix = gene[:cut_point].tolist()
        
        # 2. 调用 GPT 续写
        try:
            # generate 返回的是完整序列 (prompt + new)
            new_seq_list = self.model.generate(
                prompt_sequence=prefix,
                max_new_tokens=seq_len - cut_point,
                temperature=self.temp,
                top_k=20
            )
            
            # 3. 截断/补齐 (防止模型生成长度不对)
            new_data = np.array(new_seq_list[:seq_len])
            if len(new_data) < seq_len:
                # 极端情况补 0
                padding = np.zeros(seq_len - len(new_data), dtype=int)
                new_data = np.concatenate([new_data, padding])
                
            return self._create_new_individual(individual, new_data)
            
        except Exception as e:
            print(f"[GPTSuffixMutation] Failed: {e}")
            return individual

class GPTRejectionSamplingMutation(GPTMutationBase):
    """
    【拒绝采样变异】(Rejection Sampling / In-filling)
    遮挡中间一段，让 GPT 盲写 K 个候选片段。
    然后计算 K 个完整序列的困惑度，选择拼接最顺滑（Loss最低）的那个。
    适用于：修复衔接不当的乐段，优化局部流畅度。
    """
    def __init__(self, model: GPTMusicEvaluator, k: int = 5, temperature: float = 1.2, max_mask_len: int = 6):
        super().__init__(model)
        self.k = k
        self.temp = temperature
        self.max_mask_len = max_mask_len

    def mutate(self, individual: Individual) -> Individual:
        gene = individual.data
        seq_len = len(gene)
        
        # 1. 确定 Mask 区域
        # 长度 2 到 max_mask_len
        mask_len = np.random.randint(2, self.max_mask_len + 1)
        
        # 保证头尾留空
        safe_high = seq_len - mask_len - 2
        if safe_high <= 2:
            return individual
            
        start_idx = np.random.randint(2, safe_high)
        end_idx = start_idx + mask_len
        
        prefix = gene[:start_idx].tolist()
        suffix = gene[end_idx:].tolist()
        
        candidates = []
        
        # 2. 生成 K 个候选
        # 这里用循环调用，因为 Evaluator 接口目前是单条生成的
        for _ in range(self.k):
            try:
                # 高温采样 (1.2) 以保证 K 个结果不同
                gen_full = self.model.generate(
                    prompt_sequence=prefix,
                    max_new_tokens=mask_len,
                    temperature=self.temp,
                    top_k=20
                )
                
                # 截取中间生成的部分
                middle_part = gen_full[start_idx : start_idx + mask_len]
                
                # 拼接完整
                full_seq = prefix + middle_part + suffix
                
                if len(full_seq) == seq_len:
                    candidates.append(full_seq)
            except:
                continue
        
        if not candidates:
            return individual
            
        # 3. 评估 (Ranking)
        # 批量计算 Fitness (1/Loss)
        candidates_arr = np.array(candidates)
        scores = self.model.evaluate(candidates_arr)
        
        # 4. 择优
        best_idx = np.argmax(scores)
        best_data = candidates_arr[best_idx]
        
        # Debug info (可选)
        # print(f"Rejection Sampling: Best Score {scores[best_idx]:.2f} vs Avg {np.mean(scores):.2f}")
        
        return self._create_new_individual(individual, best_data)

class GPTVerifiedPointMutation(GPTMutationBase):
    """
    【验证式点变异】(Verified Point Mutation)
    随机修改一个音符，然后用 GPT 快速检查。
    如果修改后的序列 Loss 飙升（说明改得太离谱），则拒绝此次修改；否则接受。
    适用于：在保持高流畅度的前提下引入微小扰动。
    """
    def __init__(self, model: GPTMusicEvaluator, prob_change: float = 0.5):
        super().__init__(model)
        self.prob_change = prob_change # 执行变异的概率(在外部调度器之上再加一层控制)

    def mutate(self, individual: Individual) -> Individual:
        # 先获取当前分数为基准 (如果个体里已经缓存了 fitness 最好，没有就算一下)
        # 为了节省计算，我们采用“宽容策略”：只要新分数的 Loss 不超过某个阈值，或者比旧分数差得不多，就接受。
        
        gene = individual.data.copy()
        seq_len = len(gene)
        
        # 1. 随机改动
        idx = np.random.randint(0, seq_len - 1) # 不改最后一个，因为它是结束
        original_val = gene[idx]
        
        # 随机操作：变成延音，或者移调
        if np.random.random() < 0.2:
            gene[idx] = HOLD_VAL
        else:
            shift = np.random.choice([-2, -1, 1, 2])
            val = gene[idx] + shift
            gene[idx] = max(PITCH_MIN, min(PITCH_MAX, val))
            
        if gene[idx] == original_val:
            return individual

        # 2. 验证 (Verification)
        # 我们比较新旧序列的 Fitness
        # 注意：这里会带来 2 次推理开销 (如果 individual.fitness 是空的)
        
        # 计算新序列分数
        new_fitness = self.model.get_fitness_score(gene.tolist())
        
        # 获取旧序列分数 (尝试读取缓存)
        old_fitness = individual.fitness
        if old_fitness == 0.0: # 如果没缓存
             old_fitness = self.model.get_fitness_score(individual.data.tolist())
        
        # 3. 决策逻辑
        # 如果新分数比旧分数低太多 (例如降低了 20% 以上)，则拒绝
        # ratio = new / old. 如果 ratio < 0.8，说明变差了很多
        ratio = new_fitness / (old_fitness + 1e-9)
        
        if ratio > 0.8:
            # 接受变异
            ind = self._create_new_individual(individual, gene)
            # 顺便把算好的 fitness 存进去，省得下一轮 evaluator 再算一遍
            ind.fitness = new_fitness 
            return ind
        else:
            # 拒绝变异，返回原样
            return individual