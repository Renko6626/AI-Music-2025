import os
import numpy as np
import torch
import random
from typing import List

# --- 1. 导入基础设施 ---
from MusicRep import MelodySequence, Synthesizer, SineStrategy,StringStrategy, MusicConfig, fixGrid
from GA.ga_framework import (
    GAEngine, 
    MutationScheduler, 
    MultiRuleEvaluator, 
    Individual, 
    SelectionStrategy, 
    CrossoverStrategy, 
    MutationStrategy,
    TournamentSelection,
    MusicIndividual
)
from GA.default_mutators import (
    TranspositionMutation,

)

# --- 2. 导入 GPT 相关模块 ---
from transformer.gpt_evaluator import GPTMusicEvaluator
from gpt_rule import create_gpt_objective
from gpt_mutators import (
    GPTSuffixMutation, 
    GPTRejectionSamplingMutation, 
    GPTVerifiedPointMutation
)
from gpt_crossover import (
    StructureAwareCrossover, 
    GPTLogitMixingCrossover, 
    CompositeCrossover
)

# ==========================================
# 配置参数
# ==========================================
NANO_MODEL_PATH = "./transformer/final_models/MelodyGPT_nano.pth"       # 用于评估
STANDARD_MODEL_PATH = "./transformer/final_models/MelodyGPT_standard.pth" # 用于生成
OUTPUT_DIR = "evolution_results"
POP_SIZE = 50
N_GENERATIONS = 100
SAVE_INTERVAL = 2  # 每隔多少代保存一次音频

# ==========================================
# 定义一些基础乐理规则 (Fallback)
# ==========================================
class SimpleMusicRules:
    @staticmethod
    def pitch_in_key(grid):
        """奖励 C 大调音符, 而且音符音高不能太高或者太低"""
        # 0=Rest, 1=Hold
        notes = grid[grid > 1]
        if len(notes) == 0: return 0.0
        c_scale = {0, 2, 4, 5, 7, 9, 11} # C D E F G A B
        in_key = sum(1 for x in notes if (((x-2) % 12) in c_scale and 60 <= x <= 84))
        return in_key / len(notes)

    @staticmethod
    def rhythmic_variety(grid):
        """惩罚过多的休止符或过多的连续 Hold"""
        rests = np.sum(grid == 0)
        if rests > 4: return 0.0 # 太多休止
        return 1.0

# ==========================================


class AntiBoringRules:
    @staticmethod
    def sufficient_activity(grid: np.ndarray) -> float:
        """
        规则：奖励音符密度。如果密度太低（懒惰），得分归零。
        假设 grid 中: 0=Rest, 1=Hold, >=2=Pitch
        """
        # 计算起音（Attack）的数量，即非 Hold 且非 Rest 的音符
        # 或者如果你允许休止符，可以只计算非 Hold
        
        # 统计实际按下的音符数量 (Attacks)
        n_attacks = np.sum(grid >= 2)
        
        # 32个步长里，假设至少要有 6 个音符才算是一段旋律
        # (平均每个小节 1.5 个音符，这已经很宽容了)
        min_attacks = 6
        
        if n_attacks < min_attacks:
            # 惩罚：不仅不给分，甚至可以给负分（如果你的GA支持）
            # 这里返回 0.0，让它在加权求和中处于劣势
            return 0.0 
        
        # 如果达到了最低标准，可以给满分，或者根据密度线性奖励
        # 这里给 1.0，表示“通过了活跃度检查”
        return 1.0

    @staticmethod
    def max_hold_length(grid: np.ndarray) -> float:
        """
        规则：惩罚过长的连续长音。
        防止出现一个音拖 16 拍的情况。
        """
        current_hold_len = 0
        max_len = 0
        
        for token in grid:
            if token == 1: # Hold
                current_hold_len += 1
            else:
                max_len = max(max_len, current_hold_len)
                current_hold_len = 0
        max_len = max(max_len, current_hold_len)
        
        # 设定阈值：比如最长不能超过 8 个单位 (即 4 拍/全音符)
        limit = 3
        
        if max_len > limit:
            # 超过限制，给予惩罚。
            # 比如每超 1 个单位，扣一点分
            return max(0.0, 1.0 - (max_len - limit) * 0.1)
        
        return 1.0
class MusicTheoryRules:
    @staticmethod
    def pitch_entropy(grid: np.ndarray) -> float:
        """
        规则：奖励音高多样性 (香农熵)。
        防止旋律只在 1-2 个音符之间来回跳，或者全是长音。
        """
        # 提取所有音高 (排除 Rest=0, Hold=1)
        pitches = grid[grid >= 2]
        
        if len(pitches) == 0:
            return 0.0
            
        # 计算每个音高出现的概率
        unique, counts = np.unique(pitches, return_counts=True)
        probs = counts / len(pitches)
        
        # 计算熵: -sum(p * log(p))
        entropy = -np.sum(probs * np.log2(probs + 1e-9))
        
        # 归一化：假设理想的熵在 2.0 到 4.0 之间
        # 我们用 tanh 把它映射到 0~1
        # 一个只有 1 种音高的序列熵为 0
        return np.tanh(entropy/2)

# ==========================================
# 主流程
# ==========================================
def main():
    # 0. 环境初始化
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Initializing Hybrid Evolutionary System on {device}...")

    # --- 1. 加载模型 (双模型架构) ---
    print("🤖 Loading Models...")
    
    # Model A: The Critic (Nano) - 仅用于算分，省显存，速度快
    # 如果没有单独的 nano 权重，也可以用 standard 代替
    if os.path.exists(NANO_MODEL_PATH):
        nano_evaluator = GPTMusicEvaluator(NANO_MODEL_PATH, device=device)
        print("   ✅ Nano Model (Critic) loaded.")
    else:
        print("   ⚠️ Nano model not found, falling back to Standard for critique.")
        nano_evaluator = None 

    # Model B: The Artist (Standard) - 用于生成和复杂变异
    standard_evaluator = GPTMusicEvaluator(STANDARD_MODEL_PATH, device=device)
    print("   ✅ Standard Model (Artist) loaded.")

    # 如果刚才没加载到 Nano，就让 Artist 兼任 Critic
    if nano_evaluator is None:
        nano_evaluator = standard_evaluator

    # --- 2. 配置评估器 (Evaluator) ---
    print("⚖️ Configuring Evaluator...")
    evaluator = MultiRuleEvaluator()
    
    # A. 规则打分 (权重 1.0)
    evaluator.register(SimpleMusicRules.pitch_in_key, weight=4.0, name="InKey")
    evaluator.register(SimpleMusicRules.rhythmic_variety, weight=3.0, name="Rhythm")
    
    
    # B. GPT 困惑度打分 (权重 4.0 - 核心指标)
    # 使用 Nano 模型进行快速评估
    gpt_metric = create_gpt_objective(
    nano_evaluator, 
    mode="linear", 
    target_loss=1.5,  # 认为低于 1.5 的都是完美旋律
    tolerance=4.0     # 5.5 - 1.5 = 4.0
    )
# 注册时权重可以给 1.0，因为它现在和规则一样都是 0~1 了
    evaluator.register(gpt_metric, weight=2.0, name="GPT_Perplexity")

    # C. 反无聊规则 (权重 10.0, 作为补丁)
    evaluator.register(AntiBoringRules.sufficient_activity, weight=10.0, name="Activity")
    evaluator.register(AntiBoringRules.max_hold_length, weight=4.0, name="MaxHold")
    evaluator.register(MusicTheoryRules.pitch_entropy, weight=6.0, name="PitchEntropy")

    # --- 3. 配置变异调度器 (Mutation) ---
    print("🧬 Configuring Mutations...")
    scheduler = MutationScheduler()
    
    # A. [Standard] 拒绝采样 (修复衔接) - 权重 4.0 (主力)
    scheduler.register(
        GPTRejectionSamplingMutation(standard_evaluator, k=10, max_mask_len=6),
        weight=4.0, name="GPT_Infill"
    )
    
    # B. [Standard] 后缀重生成 (探索新意) - 权重 2.0
    scheduler.register(
        GPTSuffixMutation(standard_evaluator, temperature=2.0),
        weight=1.0, name="GPT_Suffix"
    )
    
    # C. [Standard] 验证式微调 (保守优化) - 权重 2.0
    scheduler.register(
        GPTVerifiedPointMutation(nano_evaluator), # 注意：这里也可以用 Nano 来验证以加速
        weight=2.0, name="GPT_Verify"
    )
    
    # D. 传统移调 (保持调性多样性) - 权重 1.0
    scheduler.register(TranspositionMutation(), weight=1.0, name="Transpose")

    # --- 4. 配置交叉策略 (Crossover) ---
    print("⚔️ Configuring Crossover...")
    composite_cross = CompositeCrossover()
    
    # A. 结构交叉 (快速，保留小节) - 60%
    composite_cross.register(StructureAwareCrossover([8, 16, 24]), weight=0.6)
    
    # B. [Standard] GPT 软引导交叉 (高质量融合) - 40%
    # 这需要 standard_evaluator 实现了 generate_with_logit_bias
    
    composite_cross.register(
        GPTLogitMixingCrossover(standard_evaluator, alpha=2.5), 
        weight=0.4
    )
    

    # --- 5. 定义初始种群工厂 (Seeding) ---
    print("🌱 Configuring Factory...")
    def gpt_seeded_factory():
        """
        混合初始化：
        1. 随机产生前 4 个音符 (Motif)
        2. 用 Standard GPT 续写剩下 28 个音符
        """
        start_pitch = random.randint(60, 72) # C4-C5
        # 构造一个简短的动机
        """
        motif = [start_pitch, MusicConfig.HOLD_VAL, start_pitch + random.choice([-2, 2, 4]), MusicConfig.HOLD_VAL]
        
        # 让 Standard 模型续写
        try:
            full_seq = standard_evaluator.generate(
                prompt_sequence=motif,
                max_new_tokens=64 - len(motif),
                temperature=1.2 # 初始种群多样性要高
            )
            # 截取前64个
            grid = np.array(full_seq[:64])
        except:
            # 降级方案
            grid = MelodySequence.from_random().grid
        """
        grid=MelodySequence.from_random().grid
            
        return MusicIndividual(fixGrid(grid))

    # --- 6. 组装引擎 ---
    engine = GAEngine(
        pop_size=POP_SIZE,
        n_generations=N_GENERATIONS,
        evaluator=evaluator,
        selection_strat=TournamentSelection(k=3),
        crossover_strat=composite_cross,
        mutation_scheduler=scheduler,
        individual_factory=gpt_seeded_factory,
        repair_func=fixGrid,
        elite_ratio=0.1
    )

    # --- 7. 运行进化循环 (带音频监听) ---
    print(f"\n🎼 Starting Evolution for {N_GENERATIONS} generations...")
    
    # 初始化音频合成器
    synth = Synthesizer(strategy=StringStrategy())
    
    engine.initialize()
    
    for gen in range(N_GENERATIONS):
        # 执行一步进化
        engine.step(gen)
        
        # 获取当前最优
        best_ind = engine.best_individual
        
        # 每隔 N 代保存一次
        if (gen + 1) % SAVE_INTERVAL == 0 or gen == 0:
            filename_base = os.path.join(OUTPUT_DIR, f"gen_{gen+1:03d}_fit_{best_ind.fitness:.2f}")
            
            # 转换为 MelodySequence
            melody = MelodySequence(best_ind.data)
            
            # 保存图片 (如果有实现 save_staff)
            melody.save_staff(filename_base + ".png")
            
            # 保存音频
            try:
                synth.render(best_ind.data, bpm=120, output_path=filename_base + ".wav")
                # 同时也保存 MIDI 以便后续分析
                melody.save_midi(filename_base + ".mid")
                print(f"   💾 Snapshot saved: {filename_base}.wav")
            except Exception as e:
                print(f"   ⚠️ Audio render failed: {e}")

    print("\n🎉 Evolution Complete!")
    print(f"🏆 Final Best Fitness: {engine.best_individual.fitness:.4f}")
    print(f"🎶 Sequence: {engine.best_individual.data}")

if __name__ == "__main__":
    main()