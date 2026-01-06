from GA import MusicEvaluator, RuleBasedEvaluator, BasicRules, MusicGeneticOptimizer, PentatonicRules, ClassicalRules
import numpy as np
from MusicRep import MelodySequence, Synthesizer, MusicConfig, SineStrategy
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

EXAMPLE_PATH="example_outputs/ga_example/"
if not os.path.exists(EXAMPLE_PATH):
    os.makedirs(EXAMPLE_PATH)

class CustomEvaluator(RuleBasedEvaluator):
    "自定义评估器，记录每一轮每个指标的评分情况和总的评分情况"
    def __init__(self):
        super().__init__()
        self.rule_names=[]
        self.records = []

    def add_rule(self, rule_func, weight, name=None):
        super().add_rule(rule_func, weight)
        if name is None:
            name = rule_func.__name__
        self.rule_names.append(name)
        return self
    
    def evaluate(self, population_grid: np.ndarray) -> np.ndarray:
        pop_size = len(population_grid)
        scores = np.zeros((pop_size, len(self.rules)), dtype=np.float64)
        total_scores = np.zeros(pop_size, dtype=np.float64)
        weights = np.array(self.weights)

        for i in range(pop_size):
            for j, rule in enumerate(self.rules):
                scores[i, j] = rule(population_grid[i])
            total_scores[i] = np.dot(scores[i], weights)

        # 记录当前轮次的评分情况
        mean_scores = np.mean(scores, axis=0)
        mean_total_score = np.mean(total_scores)
        best_individual = np.argmax(total_scores)
        best_scores = scores[best_individual]
        best_total_score = total_scores[best_individual]
        self.records.append(dict(
            mean_scores=mean_scores,
            mean_total_score=mean_total_score,
            best_scores=best_scores,
            best_total_score=best_total_score
        ))
            
        return total_scores

def run_and_save_pentatonic_ga():
    np.random.seed(int.from_bytes("SYBNB!".encode()[:4],'big'))
    evaluator = CustomEvaluator()

    # ===== Pentatonic Evaluator =====
    evaluator.add_rule(PentatonicRules.pentatonic_fit, weight=2.0)
    evaluator.add_rule(PentatonicRules.stepwise_motion_preference, weight=1.4)
    evaluator.add_rule(PentatonicRules.melodic_flow, weight=1.0)
    evaluator.add_rule(PentatonicRules.contour_variation_reward, weight=0.9)  
    evaluator.add_rule(PentatonicRules.overlong_note_penalty, weight=1.3)    
    evaluator.add_rule(PentatonicRules.rest_sparsity_penalty, weight=1.1)
    evaluator.add_rule(PentatonicRules.note_density_target, weight=1.0)
    evaluator.add_rule(PentatonicRules.register_balance, weight=0.8)

    ga_optimizer = MusicGeneticOptimizer(
        pop_size=100,
        n_generations=800,
        elite_ratio=0.2,
        prob_point_mutation=0.1,
        prob_transposition=0,
        prob_retrograde=0,
        prob_inversion=0,
        evaluator_model=evaluator,
        device='cpu'
    )

    ga_optimizer._initialize_population()
    ga_optimizer.fit()
    # 获取最优个体
    best_melody_grid = ga_optimizer.best_individual_
    best_melody = MelodySequence(best_melody_grid)

    # 输出结果
    print("最优旋律序列的音符网格:", best_melody.grid)
    best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody_pentatonic.png"))
    synth = Synthesizer(strategy=SineStrategy())
    synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody_pentatonic.wav"))
    print("已保存最优旋律的合成音频为 best_melody_pentatonic.wav")

    # 保存结果
    np.savez(os.path.join(EXAMPLE_PATH, "pentatonic_ga_records.npz"), records=evaluator.records, rule_names=evaluator.rule_names, weights=evaluator.weights)


def run_and_save_classical_ga():
    np.random.seed(int.from_bytes("SYBNB!".encode()[:4],'big'))
    evaluator = CustomEvaluator()

    # ===== Layer A =====
    evaluator.add_rule(ClassicalRules.key_fit_best_of_24, weight=1.4)
    evaluator.add_rule(ClassicalRules.interval_stepwise_preference, weight=1.2)
    evaluator.add_rule(ClassicalRules.rest_and_long_rest_penalty, weight=1.3)
    evaluator.add_rule(ClassicalRules.melodic_range_score, weight=0.6)

    # ===== Layer B =====
    evaluator.add_rule(ClassicalRules.phrase_start_on_strongbeat, weight=0.6)
    evaluator.add_rule(ClassicalRules.cadence_end_stable_and_long, weight=1.6)
    evaluator.add_rule(ClassicalRules.motif_repetition_interval, weight=0.9)
    evaluator.add_rule(ClassicalRules.syncopation_penalty, weight=1.1)
    evaluator.add_rule(ClassicalRules.chromatic_semitone_overuse_penalty, weight=0.6)

    # ===== Layer C =====
    evaluator.add_rule(ClassicalRules.note_density_target, weight=1.0)
    evaluator.add_rule(ClassicalRules.leading_tone_resolution, weight=0.9)

    ga_optimizer = MusicGeneticOptimizer(
        pop_size=100,
        n_generations=800,
        elite_ratio=0.2,
        prob_point_mutation=0.1,
        prob_transposition=0,
        prob_retrograde=0,
        prob_inversion=0,
        evaluator_model=evaluator,
        device='cpu'
    )

    ga_optimizer._initialize_population()
    ga_optimizer.fit()
    # 获取最优个体
    best_melody_grid = ga_optimizer.best_individual_
    best_melody = MelodySequence(best_melody_grid)

    # 输出结果
    print("最优旋律序列的音符网格:", best_melody.grid)
    best_melody.save_staff(os.path.join(EXAMPLE_PATH, "best_melody_classical.png"))
    synth = Synthesizer(strategy=SineStrategy())
    synth.render(best_melody.grid, bpm=120, output_path=os.path.join(EXAMPLE_PATH, "best_melody_classical.wav"))
    print("已保存最优旋律的合成音频为 best_melody_classical.wav")
    
    # 保存结果
    np.savez(os.path.join(EXAMPLE_PATH, "classical_ga_records.npz"), records=evaluator.records, rule_names=evaluator.rule_names, weights=evaluator.weights)


def plot_pentatonic_ga_scores():
    # 读取结果
    data = np.load(os.path.join(EXAMPLE_PATH, "pentatonic_ga_records.npz"), allow_pickle=True)
    records = data['records'].tolist()
    rule_names = data['rule_names'].tolist()
    weights = data['weights'].tolist()
    # 绘图
    generations = len(records)
    x_generations = np.arange(1, generations + 1)
    mean_scores_over_gens = np.array([record['mean_scores'] for record in records])
    best_scores_over_gens = np.array([record['best_scores'] for record in records])
    mean_total_scores_over_gens = np.array([record['mean_total_score'] for record in records])
    best_total_scores_over_gens = np.array([record['best_total_score'] for record in records])

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    colormap=plt.get_cmap("tab10")
    colors = [colormap(i) for i in range(len(rule_names))]
    for i, rule_name in enumerate(rule_names):
        # plt.plot(x_generations, mean_scores_over_gens[:, i], label=f'Mean {rule_name}', color=colors[i], linestyle='--')
        plt.plot(x_generations, best_scores_over_gens[:, i], label=f'Best {rule_name} (w={weights[i]:.1f})', color=colors[i])
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Pentatonic GA Rule Scores Over Generations')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    # plt.plot(x_generations, mean_total_scores_over_gens, label='Mean Total Score')
    plt.plot(x_generations, best_total_scores_over_gens, label='Best Total Score')
    plt.title('Total Scores Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Score')
    plt.legend()
    plt.grid()

    # # 放大绘制200-800代的细节
    # ax=plt.gca()
    # ax_inset = inset_axes(ax, width="40%", height="30%", loc='lower right', borderpad=2)
    # ax_inset.plot(x_generations, best_total_scores_over_gens, label='Best Total Score')
    # ax_inset.set_xlim(200, 800)
    # ax_inset.set_ylim(np.min(best_total_scores_over_gens[200:])-0.01, np.max(best_total_scores_over_gens[200:])+0.01)
    # ax_inset.set_title('Detail (Gen 200-800)')
    # ax_inset.grid()
    # ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    plt.tight_layout()
    plt.savefig(os.path.join(EXAMPLE_PATH, "pentatonic_ga_scores.png"))
    plt.close()


def plot_classical_ga_scores():
    # 读取结果
    data = np.load(os.path.join(EXAMPLE_PATH, "classical_ga_records.npz"), allow_pickle=True)
    records = data['records'].tolist()
    rule_names = data['rule_names'].tolist()
    weights = data['weights'].tolist()
    # 绘图
    generations = len(records)
    x_generations = np.arange(1, generations + 1)
    mean_scores_over_gens = np.array([record['mean_scores'] for record in records])
    best_scores_over_gens = np.array([record['best_scores'] for record in records])
    mean_total_scores_over_gens = np.array([record['mean_total_score'] for record in records])
    best_total_scores_over_gens = np.array([record['best_total_score'] for record in records])

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    colormap=plt.get_cmap("tab10")
    colors = [colormap(i) for i in range(len(rule_names))]
    for i, rule_name in enumerate(rule_names):
        # plt.plot(x_generations, mean_scores_over_gens[:, i], label=f'Mean {rule_name}', color=colors[i], linestyle='--')
        plt.plot(x_generations, best_scores_over_gens[:, i], label=f'Best {rule_name} (w={weights[i]:.1f})', color=colors[i])
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Classical GA Rule Scores Over Generations')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    # plt.plot(x_generations, mean_total_scores_over_gens, label='Mean Total Score')
    plt.plot(x_generations, best_total_scores_over_gens, label='Best Total Score')
    plt.title('Total Scores Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Score')
    plt.legend()
    plt.grid()

    # # 放大绘制200-800代的细节
    # ax=plt.gca()
    # ax_inset = inset_axes(ax, width="40%", height="30%", loc='lower right', borderpad=2)
    # ax_inset.plot(x_generations, best_total_scores_over_gens, label='Best Total Score')
    # ax_inset.set_xlim(200, 800)
    # ax_inset.set_ylim(np.min(best_total_scores_over_gens[200:])-0.01, np.max(best_total_scores_over_gens[200:])+0.01)
    # ax_inset.set_title('Detail (Gen 200-800)')
    # ax_inset.grid()
    # ax.indicate_inset_zoom(ax_inset, edgecolor="black")

    plt.tight_layout()
    plt.savefig(os.path.join(EXAMPLE_PATH, "classical_ga_scores.png"))
    plt.close()

if __name__ == "__main__":
    if not os.path.exists(os.path.join(EXAMPLE_PATH, "pentatonic_ga_records.npz")):
        run_and_save_pentatonic_ga()
    if not os.path.exists(os.path.join(EXAMPLE_PATH, "classical_ga_records.npz")):
        run_and_save_classical_ga()
    plot_pentatonic_ga_scores()
    plot_classical_ga_scores()
