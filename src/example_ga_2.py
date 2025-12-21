import numpy as np
import os
import copy

# 导入你的现有工具库
from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy, fixGrid
# 假设 BasicRules 还在 GA.py 里，或者你可以直接在这里定义
from GA.default_rules import BasicRules 

# 导入我们新写的框架
from GA.ga_framework import (
    GAEngine, 
    MutationScheduler, 
    Individual, 
    Evaluator, 
    SelectionStrategy, 
    CrossoverStrategy, 
    MutationStrategy,
    MusicIndividual,
    MultiRuleEvaluator
)
from GA.default_mutators import (
    TranspositionMutation, 
    PointMutation, 
    InversionMutation
)
from GA.default_crossover import OnePointCrossover
from GA.ga_framework import TournamentSelection



# ==========================================
# 主程序
# ==========================================

def run_example():
    # 0. 基础设置
    np.random.seed(int.from_bytes("SYBNB!".encode()[:4], 'big'))
    EXAMPLE_PATH = "example_outputs/ga_modular/"
    if not os.path.exists(EXAMPLE_PATH):
        os.makedirs(EXAMPLE_PATH)

    # 1. 配置评估器 (Evaluator)
    evaluator = MultiRuleEvaluator()
    evaluator.register(BasicRules.smooth_contour, weight=1.0)
    evaluator.register(BasicRules.rhythmic_variety, weight=0.5)
    evaluator.register(BasicRules.pitch_in_key_c_major, weight=1.0)

    # 自定义规则
    def my_custom_rule(grid: np.ndarray) -> float:
        notes = grid[grid > 1]
        unique_notes = len(set(notes))
        return np.tanh(unique_notes / 12.0)
    
    evaluator.register(my_custom_rule, weight=0.8)
    print("✅ Evaluator configured with rules.")

    # 2. 配置变异调度器 (Mutation Scheduler)
    scheduler = MutationScheduler()
    
    # 注册算子并分配权重
    # 比如：点变异权重 10，移调权重 0 (根据你旧代码的配置)
    # 如果你想启用移调，只需把 weight 改为非 0 即可
    scheduler.register(PointMutation(prob=0.1), weight=10.0, name="PointMut")
    scheduler.register(TranspositionMutation(), weight=0.0, name="Transpose") 
    
    print("✅ Mutation Scheduler configured.")

    # 3. 定义工厂函数和修复函数
    def music_factory():
        # 生成随机个体的逻辑
        return MusicIndividual(MelodySequence.from_random().grid)

    def music_repair(data):
        # 调用 MusicRep 中的修复逻辑
        return fixGrid(data)

    # 4. 组装引擎 (Engine Assembly)
    engine = GAEngine(
        pop_size=100,
        n_generations=500,
        evaluator=evaluator,
        selection_strat=TournamentSelection(k=3),
        crossover_strat=OnePointCrossover(),
        mutation_scheduler=scheduler,
        individual_factory=music_factory,
        repair_func=music_repair,
        elite_ratio=0.2
    )

    print("🚀 Starting Modular GA Engine...")

    # 5. 运行优化
    best_ind = engine.run() # 返回的是 MusicIndividual 对象

    # 6. 后处理与输出
    best_melody_grid = best_ind.data
    best_melody = MelodySequence(best_melody_grid)
    
    print(f"🏆 Best Fitness: {best_ind.fitness:.4f}")
    print("最优旋律序列:", best_melody.grid)

    # 导出
    best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody.png"))
    
    synth = Synthesizer(strategy=SineStrategy())
    synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody.wav"))
    print(f"✅ Result saved to {EXAMPLE_PATH}")

if __name__ == "__main__":
    run_example()