import numpy as np
import os

from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy, fixGrid
from GA.evaluator import build_classical_evaluator
from GA.ga_framework import (
    GAEngine,
    MutationScheduler,
    MusicIndividual,
    TournamentSelection,
)
from GA.default_mutators import (
    TranspositionMutation,
    PointMutation,
    InversionMutation,
)
from GA.default_crossover import OnePointCrossover


def run_example():
    # 为了结果可复现，添加随机数种子
    np.random.seed(int.from_bytes("SYBNB!".encode()[:4], 'big'))

    EXAMPLE_PATH = "example_outputs/ga_example/"
    if not os.path.exists(EXAMPLE_PATH):
        os.makedirs(EXAMPLE_PATH)

    # 1. 构建古典评估器（新版 MultiRuleEvaluator）
    evaluator = build_classical_evaluator()

    # 2. 配置变异调度器
    scheduler = MutationScheduler()
    scheduler.register(PointMutation(prob=0.1), weight=10.0, name="PointMut")
    scheduler.register(TranspositionMutation(), weight=0.0, name="Transpose")
    scheduler.register(InversionMutation(), weight=0.0, name="Inversion")

    # 3. 工厂与修复函数
    def music_factory():
        return MusicIndividual(MelodySequence.from_random().grid)

    def music_repair(data):
        return fixGrid(data)

    # 4. 组装并运行 GA 引擎
    engine = GAEngine(
        pop_size=100,
        n_generations=500,
        evaluator=evaluator,
        selection_strat=TournamentSelection(k=3),
        crossover_strat=OnePointCrossover(),
        mutation_scheduler=scheduler,
        individual_factory=music_factory,
        repair_func=music_repair,
        elite_ratio=0.2,
    )

    best_ind = engine.run()

    # 5. 输出与渲染
    best_melody_grid = best_ind.data
    best_melody = MelodySequence(best_melody_grid)
    print("最优旋律序列的音符网格:", best_melody.grid)

    best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody_classical.png"))

    synth = Synthesizer(strategy=SineStrategy())
    synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody_classical.wav"))
    print("已保存最优旋律的合成音频为 best_melody_classical.wav")


if __name__ == "__main__":
    run_example()
