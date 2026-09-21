<!-- omit from toc -->
# TAGA：基于Transformer增强的遗传算法作曲系统

<!--
- 这是一份面向使用者的readme，不是面向开发者的readme，注意措辞。
- 骇死助教！！！
-->

<!-- omit from toc -->
## 目录

- [项目概述](#项目概述)
- [快速上手](#快速上手)
  - [安装环境](#安装环境)
  - [运行目录](#运行目录)
  - [纯遗传算法作曲](#纯遗传算法作曲)
  - [纯Transformer生成](#纯transformer生成)
  - [混合作曲](#混合作曲)
- [类和数据结构](#类和数据结构)
  - [音乐基础(MusicRep库)](#音乐基础musicrep库)
  - [遗传算法(GA库)](#遗传算法ga库)
  - [Transformer模型(Transformer库)](#transformer模型transformer库)
  - [GPT增强的GA组件](#gpt增强的ga组件)
  - [导入全部](#导入全部)
- [自己组装](#自己组装)
- [运行和输出](#运行和输出)
  - [运行](#运行)
  - [输出为WAV音频](#输出为wav音频)
    - [输出为MIDI序列](#输出为midi序列)
    - [输出为五线谱](#输出为五线谱)
- [项目结构](#项目结构)
  - [项目结构总体概述](#项目结构总体概述)
  - [MusicRep库](#musicrep库)
    - [核心类和函数](#核心类和函数)
  - [GA 库](#ga-库)
    - [快速示例](#快速示例)
  - [transformer库](#transformer库)
    - [核心代码](#核心代码)
    - [运行/示例](#运行示例)
  - [VAE库（旧）](#vae库旧)
    - [核心代码](#核心代码-1)
    - [运行/示例](#运行示例-1)
- [数据集与复现](#数据集与复现)
  - [三份数据集一览](#三份数据集一览)
  - [路线 A：GigaMIDI](#路线-agigamidi)
  - [路线 B：古典 MIDI](#路线-b古典-midi)
  - [路线 C：GiantMIDI-Piano](#路线-cgiantmidi-piano)
  - [删掉了哪些文件](#删掉了哪些文件)
- [分工和致谢](#分工和致谢)

## 项目概述

- 这是我们的音乐与数学大作业，我们的主题是遗传算法作曲。
- 我们以遗传算法为全局优化骨架，利用预训练Transformer模型捕捉深层音乐语义，帮助我们生成初始种群、进行变异和交配操作，并为种群提供更具音乐性的适应度评估。

<!--
## 任务要求

> 本部分直接取自教学网中的作业要求。
> 使用者大概不需要了解我们开发者心中的顾虑吧，这些规则怪谈对他们不适用
> 骇死助教！！！

**机器作曲·遗传算法**

1. 采用下述方法**之一**产生初始种群：
   a) 从具有相同节拍的若干歌曲、乐曲中选取10\~20个长度相等（例如4小节）的片段。
   b) 随机产生：给定乐音体系$$S=\{F_3,\sharp{F}_3,\dots,\sharp{F}_5,G_5\}$$随机选取$S$中的音级，配以不同的的时值，产生10\~20段4/4拍、4小节的“旋律”，其中音符的最短时值为八分音符。
2. 根据课上介绍，搜索相关参考文献．在任何一个软件平台上实现遗传算法。遗传操作应包括交叉(crossover)、变异(mutation)以及对旋律进行的移调、倒影、逆行变换等。
3. 探索建立适应度函数(fitness function)，用以指导旋律进化的方向。
4. 把初始种群作为遗传算法的输入，对其进行遗传迭代，看是否能够得到*较好*的音乐片段。
5. 真实、客观、准确地描述你所完成的各项工作及得到的实际结果，形成完整的实验报告。着重讨论**适应度函数的选取**对于最终产生旋律的音乐特性之间的联系，以及对算法本身效率的影响。
-->

## 快速上手

### 安装环境

1. 前往[Python官网](https://www.python.org/downloads/)下载Python安装包，随后双击运行该安装包进行安装。安装时全部选择默认选项，但是一定要勾上“将Python添加到环境变量”的选框。
2. [项目主页](https://github.com/Renko6626/AI-Music-2025)右上角的绿色Code按钮，再点击`Download ZIP`按钮下载本项目源代码，解压到你的电脑中。
3. 在AI-Music-2025文件夹下，Shift加右键，选择“在此处打开命令窗口”（也可以是powershell窗口）。输入命令`pip install -r requirements.txt`并回车执行，pip将自动安装运行本项目所需的全部依赖包。
4. 如果你的计算机存在显卡，并且你希望使用GPU加速模型运行，请前往[PyTorch官网](https://pytorch.org/get-started/locally/)的“Get Started”页面，选择适合你电脑的CUDA版本，然后复制对应的pip安装命令，在命令行窗口中执行安装PyTorch。

### 运行目录

- 所有示例脚本均位于`src/`文件夹下。你需要将命令行的当前目录切换到`src/`，然后运行对应的脚本文件。

```bash
~/AI-Music-2025 $ cd src
~/AI-Music-2025/src $ python ...
```

### 纯遗传算法作曲

- 如果需要使用五声调式评估规则，运行`src/example_ga_classical.py`。运行结束后，查看`example_outputs/ga_example/`文件夹，你将看到生成的MIDI文件和对应的WAV音频。
- 如果需要使用古典评估规则，运行`src/example_ga_pentatonic.py`。运行结束后，查看`example_outputs/ga_example/`文件夹，你将看到生成的MIDI文件和对应的WAV音频。

### 纯Transformer生成

- 运行`src/example_transformer.py`，你将进入控制台交互界面。
- 首先，在choose mode选项中输入`g`并回车，进入生成模式；
- 然后，命令行将提示你输入prompt tokens，这是音频的起始片段。
  - prompt tokens是一个以空格或半角逗号分隔的整数序列，和MIDI音高编码存在微小的差异。0表示休止，1表示延音，2-129表示音高（C4=62，D4=64，E4=66，依此类推）。
  - prompt tokens中一个token对应八分音符时值。不要留空。
- 接下来你可以自定义生成的tokens数量、transformer的温度和top-k采样参数。如果你不知道这些是什么，直接留空。
- 等待片刻，程序将生成完整的音频序列，前64个tokens会显示在控制台。你可以选择将生成的音频保存为WAV文件。

### 混合作曲

- 直接运行`src/hybrid_evaluation.py`。运行结束后，你可以在`evolution_results/`文件夹中找到遗传过程中每一代音频对应的WAV文件和MIDI曲谱。

## 类和数据结构

使用混合策略进行作曲时，需要接触各色的类和数据结构。如果你需要搭建适合自己的混合作曲系统，理解这些类和数据结构是必要的。

### 音乐基础(MusicRep库)

- **网格数据**：一个numpy整数数组或Python整数列表。每个元素代表一个时间步（八分音符），值为MIDI音高编号或休止/延音标记（0/1），典型长度为32（4小节×4拍×八分音符）。
- **MelodySequence类**（来自`MusicRep/melody_sequence.py`）：封装了网格数据的类，提供了多种转换和渲染方法，如导出MIDI、渲染WAV、生成五线谱等。
- **Synthesizer类**（来自`MusicRep/synthesizer.py`）：一个简易的纯Python合成器，支持多种音色策略，可将网格数据直接渲染为WAV音频文件。音色策略包括正弦波、方波和拨弦等，需要从`MusicRep.synthesizer`导入。
- **MusicConfig类**（来自`MusicRep/music_config.py`）：定义了音乐表示的常数，如音域范围、节拍信息等。一般情况下不需要修改，如何你需要调整你生成的音乐的基本属性，可以修改这个类中的常数。
- **fixGrid函数**（来自`MusicRep/fix_grid.py`）：一个辅助函数，用于修复网格数据中的语法错误，如孤立的延音符号等。遗传算法引擎的构造函数总是需要传入这个函数作为参数。

### 遗传算法(GA库)

- **MusicIndividual类**（来自`GA/ga_framework.py`）：遗传算法中的个体类，封装了网格数据和适应度分数。遗传算法的种群由多个MusicIndividual实例组成。
- **MultiRuleEvaluator类**（来自`GA/ga_framework.py`）：多规则评估器，允许用户组合多个评价规则来评估个体的适应度。可以通过`register`方法添加自定义规则函数，并为每个规则分配权重。
- **MutationScheduler类**（来自`GA/ga_framework.py`）：遗传算法中的变异调度器。可以自定义变异策略，如移调、倒影、点变异等。基础的变异策略可以从`GA/default_mutators.py`导入，自定义变异策略需要继承MutationStrategy类。
- TournamentSelection类（来自`GA/ga_framework.py`）：锦标赛选择器。一般不需要修改，如果需要自定义选择策略，可以继承SelectionStrategy类。
- OnePointCrossover类（来自`GA/default_crossovers.py`）：单点交叉操作。一般不需要修改，如果需要自定义交叉策略，可以继承CrossoverStrategy类。
- **GAEngine类**（来自`GA/ga_framework.py`）：遗传算法引擎，负责整个遗传算法的运行流程。所有的遗传算法相关操作都以这个类为中心进行。

### Transformer模型(Transformer库)

- **tokens序列**：一个整数列表，和网格数据一样，表示音乐旋律。可以通过`MelodySequence.to_remi_tokens()`方法从网格数据生成tokens序列，也可以调用`transformer.tokens_to_melodygrid()`函数将tokens序列转换回网格数据。
- MusicGPT类（来自`transformer/model.py`）：一个torch神经网络模型。
- **GPTMusicEvaluator类**（来自`transformer/gpt_evaluator.py`）：将预训练的MusicGPT模型封装为遗传算法可用的评估器。提供了批量评估和生成方法，可以直接用于遗传算法的适应度评估和变异操作。

### GPT增强的GA组件

- **StructureAwareCrossover类**（来自`src/gpt_crossover.py`）：只在小节线处分割交叉点的交叉操作。
- **GPTLogitMixingCrossover类**（来自`src/gpt_crossover.py`）：基于GPT模型的软引导融合交叉操作。它利用GPT模型的预测分布来指导交叉过程，生成更符合音乐语义的后代个体。
- **CompositeCrossover类**（来自`src/gpt_crossover.py`）：复合交叉操作。它结合了传统的单点交叉和GPT引导的交叉，先进行单点交叉，然后再应用GPT引导的融合交叉，以增强后代个体的音乐性。如果你需要使用GPT增强的交叉，你可能只需要关注此类。
- **GPTSuffixMutation类**（来自`src/gpt_mutators.py`）：基于GPT模型的后缀生成变异操作。它利用GPT模型根据个体的前缀部分生成新的后缀，从而实现变异。
- **GPTRejectionSamplingMutation类**（来自`src/gpt_mutators.py`）：基于GPT模型的拒绝采样变异操作。它使用GPT模型生成候选变异，并根据适应度分数决定是否接受该变异。
- **GPTVerifiedPointMutation类**（来自`src/gpt_mutators.py`）：基于GPT模型的验证点变异操作。它在进行点变异时，利用GPT模型评估变异后的个体，如果适应度提高则接受变异，否则拒绝。
- **create_gpt_objective函数**（来自`src/gpt_rule.py`）：基于一个GPT模型，创建一个归一化后的 GPT 评分函数。

### 导入全部

在使用这些类和函数之前，你需要先导入对应的模块。以下代码一次性导入了以上全部在遗传作曲中可能会用到的类和函数：

``` python
from MusicRep import (
    MelodySequence, 
    Synthesizer,
    MusicConfig,
    SineStrategy,
    fixGrid
)
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
    InversionMutation,
    PointMutation
)
from GA.default_crossovers import OnePointCrossover
from GA.default_evaluator import (
    BasicRules,
    PentatonicEvaluator,
    ClassicalEvaluator,
    build_basic_evaluator,
    build_pentatonic_evaluator,
    build_classical_evaluator
)
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
```

如果你的程序主入口在项目根目录下，而不是在`src/`文件夹中，你需要在导入语句前加上以下代码，以确保Python解释器能够正确找到这些模块：

``` python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
```

## 自己组装

组装一个遗传算法作曲系统需要包含以下必须要素：
- **评价器（Evaluator）**：用于评估个体适应度的组件，通常是一个MultiRuleEvaluator类，结合多个评价规则。这些评价规则可以：
  - 在`MusicRep.evaluator`中预定义，包括基础、五声或古典规则；
  - 由将GPTMusicEvaluator类输入到create_gpt_objective函数中得到，利用预训练的Transformer模型进行评估；
  - 是任何一个函数，输入网格数据，输出一个0到1之间浮点数分数（越高越好）。
- **变异器（Mutator）**：用于对个体进行变异操作的组件，这些组件需要注册在一个额外的MutationScheduler调度器中。可以：
  - 使用基础的变异策略，如移调、倒影和点变异，来自`GA/default_mutators.py`；
  - 使用GPT增强的变异策略，如GPTSuffixMutation、GPTRejectionSamplingMutation和GPTVerifiedPointMutation，来自`src/gpt_mutators.py`；
  - 继承于`GA.ga_framework`中的MutationStrategy类，编写自定义变异策略。
- **交叉器（Crossover）**：用于生成后代个体的组件。可以：
  - 使用基础的OnePointCrossover类，进行单点交叉；
  - 使用StructureAwareCrossover类，只在小节线处分割交叉点；
  - 使用GPTLogitMixingCrossover类，进行GPT引导的融合交叉；
  - 使用CompositeCrossover类，将多种交叉策略结合起来使用；
  - 继承于`GA.ga_framework`中的CrossoverStrategy类，编写自定义交叉策略。
- **选择器（Selector）**：用于从当前种群中选择个体进行繁殖的组件。可以：
  - 使用基础的TournamentSelection类，进行锦标赛选择；
  - 继承于`GA.ga_framework`中的SelectionStrategy类，编写自定义选择策略。
- **“个体工厂”(Individual Factory)**：一个用于生成初始个体的函数，没有输入参数，返回一个MusicIndividual实例。
  - 很遗憾，没有现成的个体工厂可以直接使用。你需要自己编写一个函数。
  - 一个最为简单的个体工厂可以在一行以内实现：`individual_factory = lambda: MusicIndividual(fixgrid(MelodySequence.from_random().grid))`，它会生成一个随机旋律作为个体。
  - 更好的做法是，先随机生成开始的几个音，然后使用GPTMusicEvaluator类中的`generate`方法续写剩余音符，生成更有音乐性的初始个体。可以参考`src/hybrid_evaluation.py`中的第247~273行代码。

在准备好这些组件后，你还需要小小地纠结一下，确定以下参数：

- **种群大小**（pop_size）：每一代中个体的数量，通常在20到100之间。
  - 种群大小越大，搜索空间越广，最终得到的旋律越好，但计算开销也越大。
- **迭代轮数**（n_generations）：遗传算法运行的总轮数，一般取为100。
  - 基本原则是，当适应度不再发生变化时，种群可能丧失了多样性，算法可以停止。
- **精英比例**（elite_ratio）：每一代中保留的精英个体比例，通常在0.05到0.2之间。
  - 该参数决定了每一代中最优秀的个体有多少会直接传递到下一代，
  - 如果设置过高，可能导致种群过早收敛，失去多样性；
  - 如果设置过低，则可能导致优秀基因丢失，收敛变慢。


一切就绪后，直接将它们传入GAEngine类的构造函数中，组装成一个完整的遗传算法作曲系统。

``` python
engine = GAEngine(
    pop_size=50,
    n_generations=100,
    elite_ratio=0.1,
    individual_factory=individual_factory,
    evaluator=evaluator,
    mutator_scheduler=mutator_scheduler,
    crossover=crossover,
    selector=selector
    repair_func=fixGrid
)
```



## 运行和输出

### 运行

- 直接调用`engine.run()`。这个方法最省时省力，它将运行遗传算法，直接返回最终的最优个体。
- 在一个循环中反复调用`engine.step(generation_idx)`。这个方法允许你手动控制遗传算法的每一代运行过程。使用该方法，你可以提前终止，或者在超过设定代数之后继续运行；你也可以获得每一代中的非最优个体，评估整个种群的适应度分布……一切只限制于你的想象力。


### 输出为WAV音频

将网格数据渲染为WAV音频，可以使用两种方法：
1. 使用MusicRep库中的MelodySequence类的`render_wav`方法，将MIDI序列渲染为WAV音频。该方法依赖于`midi2audio`和`fluidsynth`，需要在系统中安装FluidSynth声卡驱动，并提供一个SoundFont文件路径。这个方法可以生成高质量的音频，但需要额外的依赖和配置，尽量不要使用。
2. 使用MusicRep库中的Synthesizer类，将网格数据直接渲染为WAV音频。该方法是纯Python实现的简易合成器，不依赖外部声卡驱动，适合在无声卡环境下使用。虽然音质不如第一种方法，但足以满足基本需求。

#### 输出为MIDI序列

- 使用MelodySequence类的`save_midi(path)`方法，将网格数据保存为MIDI文件。该方法会自动合并延音符号，生成符合MIDI标准的文件。
- 如果需要再后续交由专业的DAW软件处理该旋律片段，可以使用该方法输出MIDI序列。

#### 输出为五线谱

- 使用MelodySequence类的`render_staff(output)`方法，将网格数据渲染为五线谱图像。该方法依赖于`music21`库，并且需要配置外部环境（MuseScore或LilyPond）来生成图像文件。

## 项目结构

### 项目结构总体概述

- `MusicRep/` : 音乐表示相关代码, 主要用于旋律的不同编码和转换, 以及把旋律进行播放
    - `fix_grid.py`：修复旋律序列中的语法错误。
    - `melody_sequence.py`：旋律序列类，支持网格数组、MIDI对象、REMI token 互转，也支持渲染为MIDI/WAV/五线谱。
    - `synthesizer.py`：纯 Python 实现的简易合成器，使用不同的音色将网格序列直接渲染为 WAV 音频。
- `GA/` : 遗传算法相关代码
    - `ga_framework.py`：一个通用的遗传算法框架，**包含了遗传算法本身的全部代码**，同时定义了评估器和变异调度器的接口。
    - `evaluator.py`：基于`ga_framework.py`中定义的框架，构建了多种规则评估器。**包含了五声调式评估器和古典评估器的全部代码**。
    - `default_crossovers.py`：定义了单点交叉操作。
    - `default_mutators.py`：定义了移调、倒影和点变异。
- `transformer/`：包含GPT评估器、模型定义、训练脚本和数据预处理，主要用于生成旋律片段和辅助遗传算法种群初始化。
    - `model.py`：MusicGPT 模型定义。
    - `trainv2.py`：GPT 训练脚本与数据集切片加载器。
    - `gpt_evaluator.py`：GPT 评估器，将预训练的 GPT 模型封装为遗传算法可用的评估器。
    - `dataset/preprocess.py`：从 MIDI 提取主旋律，生成 GPT 训练数据集。
    - `final_models/MelodyGPT_nano.pth`：Nano模型权重。Nano模型更小更快，直接参与评估旋律的好坏，也用于拒绝采样变异和微调变异。
    - `final_models/MelodyGPT_standard.pth`：Standard模型权重。Standard模型生成的旋律更丰富，用于后缀生成变异和软引导融合交叉。
    - 其它文件：辅助脚本和模块，如token与网格转换等。与遗传算法无关，可以忽略。
- `VAE/`：基于 GRU 的变分自编码器与评估（已弃用）
  - `vae_evaluator.py`：加载训练好的 GRU-VAE，计算潜空间风格相似度评分。
  - `model/`：`gru.py` 定义 GRU-VAE 主体。
  - `train/`：数据预处理、模型训练与日志（tensorboard events）存放位置。

### MusicRep库

MusicRep 是一个用于表示和处理旋律的python库，它支持我们将会使用的音乐表示方法(网格用于遗传算法、remi token用于Transformer等)。此外，它还包含了一个简易的MIDI合成器，可以将旋律直接渲染为WAV音频文件。

#### 核心类和函数

- `MusicConfig`：乐理与时间网格常数（音域 F3~G5，4/4 拍×4 小节，八分音符精度，共 32 步）。

- `MelodySequence`(来自`melody_sequence.py`)：
    - 处理遗传算法所用的音乐序列, 数据结构是一个32长度的整数数组。
    - 元素位置代表时间步（八分音符），值代表音高（MIDI 编号）或休止0/延音1。
    - 主要方法：
        - `from_random()`：生成一条符合音域的随机旋律。
        - `to_midi_object()` / `save_midi(path)`：合并延音，导出 `miditoolkit.MidiFile` 或直接存成本地文件。
        - `to_remi_tokens()`：生成简化 REMI token 序列（Bar/Pos/Pitch/Dur），主要用来以后喂给Transformer。
        - `render_wav(output_wav, soundfont_path=None)`：将MIDI序列渲染为音频文件（依赖 `midi2audio`/`fluidsynth`，可选）。
        - `render_staff(output)`：输出五线谱（依赖`music21`且需要配置外部环境，可选）。

- `Synthesizer`（来自`synthesizer.py`）：一个纯 Python 简易合成器，因为服务器没有声卡，按网格渲染 WAV，不依赖 MIDI 播放。
    - 指定合成音色：
        - 在创建时传入 `strategy` 参数，选择不同的合成策略（如正弦波、方波、拨弦等）。
    - 可选策略：
        - `SineStrategy`：合成正弦波和对应的谐波，模拟琴类乐器。
        - `SquareStrategy`：合成方波，模拟一些电子合成器。
        - `StringStrategy`：模拟拨弦，模拟吉他等弦乐器音色。
    - 主要方法：
        - `render(grid_sequence, bpm=120, output_path="output.wav")`：将 0/1/音高网格直接合成为 WAV。

- `fixGrid(grid_sequence)`：辅助函数，修正输入网格中的不合基本语法的音符（如孤儿hold音符，rest之后的hold等）。


### GA 库

- `ga_framework.py`：遗传算法框架，包含遗传算法引擎和对接遗传算法的所有接口类，并额外实现了其中的一部分。
  - 类 `GAEngine`：遗传算法引擎，负责整个遗传算法的运行流程。
    - `run()`：运行完整的遗传算法，返回最终的最优个体。
    - `step(generation_idx)`：运行一代遗传算法，允许手动控制每一代的运行过程。
  - 接口抽象类：包括个体类`MusicIndividual`、变异策略接口`MutationStrategy`、选择策略接口`SelectionStrategy`、交叉策略接口`CrossoverStrategy`等。
  - 类 `MultiRuleEvaluator`：多规则评估器，允许用户组合多个评价规则来评估个体的适应度。
  - 类 `MutationScheduler`：变异调度器，允许用户注册多个变异策略，并在遗传过程中随机选择应用。
  - 类 `TournamentSelection`：锦标赛选择器，实现了基于锦标赛的个体选择策略。
- `evaluator.py`：基于`ga_framework.py`中定义的框架，构建了多种规则评估器，例如五声调式评估器和古典评估器。
  - `BasicRules`：基础评估规则集合，包括音高在音域内、节奏多样性、音程平滑等规则。
  - `PentatonicEvaluator`：五声调式评估器，结合了多个五声调式相关的规则。
  - `ClassicalEvaluator`：古典评估器，结合了多个古典音乐相关的规则。
  - `build_basic_evaluator()`：构建一个预设的基础评估器实例。
  - `build_pentatonic_evaluator()`：构建一个预设的五声调式评估器实例。
  - `build_classical_evaluator()`：构建一个预设的古典评估器实例。
- `default_mutators.py`：定义了基础的变异策略，如移调、倒影和点变异。
  - 类 `TransposeMutation`：移调变异策略，实现了对个体进行随机移调的操作。
  - 类 `InversionMutation`：倒影变异策略，实现了对个体进行倒影变换的操作。
  - 类 `PointMutation`：点变异策略，实现了对个体进行随机点变异的操作。

#### 快速示例
```python
scheduler = MutationScheduler()
scheduler.register(PointMutation(prob=0.1), weight=10.0, name="PointMut")

engine = GAEngine(
    pop_size=100,
    n_generations=500,
    evaluator=build_classical_evaluator(),
    selection_strat=TournamentSelection(k=3),
    crossover_strat=OnePointCrossover(),
    mutation_scheduler=scheduler,
    individual_factory=lambda: MusicIndividual(fixgrid(MelodySequence.from_random().grid)),
    repair_func=fixGrid,
    elite_ratio=0.1,
)

best_melody_grid = engine.run().data
MelodySequence(best_melody_grid).save_midi("best_melody.mid")
```

### transformer库

GPT 自回归模型。可以用于直接生成旋律，也可以用于辅助GA算法，生成初始种群并提供适应度函数。

#### 核心代码

- `model.py`：定义GPT模型
  - 类 `MusicGPT`：模型本体
    - `forward(idx, targets=None)` 前向传播，获得logits和CE损失；
    - `generate(idx, max_new_tokens, temperature, top_k)` 自回归地生成新的token，用于音频序列的延伸。
- `gpt_evaluator.py`：评估GPT模型的好坏
  - 类 `GPTMusicEvaluator`：模型评估器
    - `evaluate(population_grid)` 批量评估大量音频序列的适应度，其中适应度定义为损失的倒数。
    - `get_fitness_score(sequence)` 评估单一音频序列的适应度；
    - `generate(prompt_sequence, ...)` 生成音频序列。
- `train.py`：训练GPT模型
  - `MusicGPTDataset`：数据集；
  - `train(config, resume_path)` ：训练并保存模型
- `dataset/preprocess.py` ：从原始音频文件中读取数据集，并进行数据增强，保存结果为文件。

#### 运行/示例

- 数据预处理：`python src/transformer/dataset/preprocess.py`
- 训练：`python src/transformer/train.py --resume latest`（可不带 `--resume`）
- 评估/生成：
```python
from transformer.gpt_evaluator import GPTMusicEvaluator
eva = GPTMusicEvaluator("./transformer/checkpoints_gpt/music_gpt_v1_best.pth")
scores = eva.evaluate(pop_grid)  # pop_grid: [B, T]
new_seq = eva.generate([130], max_new_tokens=128, temperature=1.0, top_k=20)
```

### VAE库（旧）

- GRU 版离散序列 VAE，学习旋律潜空间并用“风格距离”给 GA 打分，附带 Transformer-VAE 备选实现。
- 但是VAE模型的**表现非常差**，远不如GPT模型，因此**已弃用**。这里仍然保留该模块的代码，供有兴趣的同学参考。

#### 核心代码
- `model/gru.py`和`train/preprocess.py`：数据预处理。读取一个MIDI文件，生成增强后的数据，保存到文件。
- `train/model.py`：定义了各式各样的VAE模型
- `train/dataset.py`：只是下载和保存数据集
- `train/train.py`：非常复杂的训练流程，总之可以训练一个VAE模型出来
- `train/vae_evaluator.py`：对接`GA/evaluator.py/MusicEvaluator`基类，对一段音符片段进行评估。强烈建议将本文件中的`MusicEvaluator`改名，并显式地继承所有evaluator的基类！

#### 运行/示例
- 下载+解压 MIDI：`python src/VAE/train/preprocess.py`
- 生成 32 长度数据集：`python src/VAE/train/dataset.py`
- 训练：`python src/VAE/train/train.py`
- 评估打分：
```python
from VAE.vae_evaluator import MusicEvaluator
import torch
eva = MusicEvaluator("checkpoints/vae_gru_bach_v2_best.pth")
eva.set_target_style(torch.load("classical_dataset.pt")[:1024])
score = eva.get_style_fitness(your_seq)
```

## 数据集与复现

> **本项目已结课。** 为节省磁盘空间，所有原始数据集与预处理中间产物已于 2026-09-22 从工作目录清除
> （共约 24.8 GB）。代码、训练日志和模型权重都原样保留
> （`transformer/checkpoints_gpt/`、`transformer/final_models/`、`VAE/train/checkpoints/`）。
> **预处理脚本一个没动，路径也没变**，照下面的步骤重新拉一次数据就能复现全部训练。

### 三份数据集一览

| 数据集 | 出处 | 许可 | 谁在用 |
| --- | --- | --- | --- |
| **GigaMIDI v2.0.0** | HuggingFace [`Metacreation/GigaMIDI`](https://huggingface.co/datasets/Metacreation/GigaMIDI)。**gated**：要先在数据集页面填表、勾同意条款才能下载 | CC BY-NC 4.0，仅限非商业的研究与教学 | `trainv2.py` 的 nano / standard / heavy / v3_final 四档 |
| **MIDI Classical Music**（4796 个古典 MIDI） | HuggingFace [`drengskapur/midi-classical-music`](https://huggingface.co/datasets/drengskapur/midi-classical-music)，当年是从 `hf-mirror.com` 镜像克隆的 | MIT | VAE 全部模型，以及 transformer 的早期古典模型 |
| **GiantMIDI-Piano**（`surname_checked_midis` v1.2，7237 个钢琴 MIDI） | 归档名 `surname_checked_midis_v1.2.zip`，包内文件日期 2022-01-20 | 见上游 | `train.py` 训出的 `music_gpt_v1_*` |

### 路线 A：GigaMIDI

主力模型（`music_gpt_gigamidi_v3_*`）走的就是这条线，也是最大的一份，原始数据约 22 GB。

1. 在 HuggingFace 上通过 `Metacreation/GigaMIDI` 的 gated 申请，然后 `huggingface-cli login`。
2. 下载两个文件到 `src/transformer/dataset/gigamidi/`：
   - `Final_GigaMIDI_V2.0_Final.zip`（5.1 GB）
   - `Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv`（1.2 GB），
     **下载后必须改名成 `metadata.csv`** —— `preprocess.py` 里 `METADATA_CSV_PATH` 写死了这个名字。
3. 就地解压。外层 zip 解出来是一个 `Final_GigaMIDI_V1.1_Final/` 目录，里面**还套着三个 zip**，
   要继续就地解开：

   ```
   Final_GigaMIDI_V1.1_Final/training-V1.1-80%.zip
   Final_GigaMIDI_V1.1_Final/test-V1.1-10%.zip
   Final_GigaMIDI_V1.1_Final/validation-V1.1-10%.zip
   ```

   解完的层级必须正好长成下面这样，因为 `metadata.csv` 第一列存的就是这个相对路径：

   ```
   src/transformer/dataset/gigamidi/Final_GigaMIDI_V1.1_Final/training-V1.1-80%/no-drums/4/81a8984f….mid
   ```

   > ⚠️ **这里最容易踩坑。** `preprocess.py` 里 `DATASET_BASE_DIR = "."`，路径直接拼在 gigamidi
   > 目录下，而且对找不到的文件是 `continue` 静默跳过的。层级解错了**不会报任何错**，
   > 只会安静地产出一个几乎是空的数据集。跑完务必拿第 5 步的行数对一下再开训。

4. 生成训练数据集：

   ```bash
   cd src/transformer/dataset/gigamidi
   python preprocess.py        # 输出 ./gigamidi_processed_nodrums_v3/
   ```

   脚本只取路径里同时含 `no-drums` 和 `training`/`test`/`validation` 的条目，用 NOMML 启发式
   挑主旋律轨（阈值 12），再做清洗：最短 32 token、最多 8 个连续休止、音符密度 ≥0.3、
   音高种类 ≥5、音域限定 A0–C8（21–108）。输出是 HF `datasets` 的 `save_to_disk` 格式。
   `gigamidi_processed_nodrums` 和 `_v2` 是同一个脚本改参数跑出来的早期版本。

5. **对数基准**（当年 v3 的实际产出，共 256 MB）：

   | split | 行数 | 大小 |
   | --- | --- | --- |
   | train | 122,530 | 204 MB |
   | test | 15,125 | 26 MB |
   | validation | 15,312 | 26 MB |

6. 训练：`python src/transformer/trainv2.py`

   `trainv2.py` 里 `CONFIG_nano` / `CONFIG_standard` / `CONFIG_heavy` 的 `data_path` 都指向
   `./dataset/gigamidi/gigamidi_processed_nodrums_v3`；但 `CONFIG_v3`（`music_gpt_gigamidi_v3_final`）
   指向的是**不带后缀**的 `gigamidi_processed_nodrums`，要复现那一档得先把目录名对上。
   第 130 行的 `CONFIG=CONFIG_heavy` 决定实际跑哪一档。

### 路线 B：古典 MIDI

VAE 和 transformer 的早期古典模型共用这份。注意当年它在仓库里存了**两份一模一样的拷贝**
（`transformer/dataset/data/` 和 `VAE/train/midi_dataset_local/data/`，都是 4796 个同名文件），
重建时拉一份、另一处做软链接就行。

1. 拉数据集：

   ```bash
   cd src/VAE/train
   git clone https://huggingface.co/datasets/drengskapur/midi-classical-music midi_dataset_local
   # 当年走的是镜像：https://hf-mirror.com/datasets/drengskapur/midi-classical-music
   ```

   > `src/VAE/train/midi_dataset_local` 现在是个**空目录占位**。它当年被误当作 gitlink
   > （mode 160000，且没有 `.gitmodules`）提交进了仓库，所以这个空壳必须留着，否则
   > `git status` 会一直显示一条删除。`git clone` 允许克隆进已存在的空目录，直接按上面跑就行。

2. VAE 侧：

   ```bash
   cd src/VAE/train
   python preprocess.py     # ./midi_dataset_local/data → classical_dataset.pt（约 690 MB）
   python train.py
   ```

3. transformer 侧（早期古典模型）：

   ```bash
   cd src/transformer/dataset
   ln -s ../../VAE/train/midi_dataset_local/data ./data    # 或者直接拷一份
   python preprocess.py     # → classical_gpt_dataset_smart_v2.pt（约 86 MB）
   ```

   `gpt_evaluator.py` 的自测块要用 `classical_gpt_dataset_smart_v2.pt`；
   `test_mini_classical.py` 要用 `classical_gpt_dataset_smart.pt`（v1，同一脚本改
   `OUTPUT_FILE` 跑出来的）。

### 路线 C：GiantMIDI-Piano

`train.py`（注意不是 `trainv2.py`）训出的 `music_gpt_v1_*` 用的是这份。

1. 取到 `surname_checked_midis_v1.2.zip`，解压后把 7237 个 MIDI 放进
   `src/transformer/dataset/gp_dataset/`。
2. 预处理：

   ```bash
   cd src/transformer/dataset
   python gp_process.py     # → giant_piano_mind_v1.pt（约 1.1 GB）
   ```

3. 训练：`python src/transformer/train.py --resume latest`
   （`train.py` 的 `data_path` 写死 `./dataset/giant_piano_mind_v1.pt`。）

### 删掉了哪些文件

列在这里存档，免得以后怀疑是不是丢了东西。

| 路径（相对仓库根） | 大小 | 怎么拿回来 |
| --- | --- | --- |
| `src/transformer/dataset/gigamidi/Final_GigaMIDI_V2.0_Final.zip` | 5.1 GB | 路线 A 第 2 步 |
| `src/transformer/dataset/gigamidi/metadata.csv` | 1.2 GB | 路线 A 第 2 步 |
| `src/transformer/dataset/gigamidi/train/`、`test/` | 13 GB + 1.6 GB | 路线 A 第 3 步（旧版 V1.1 的解压残留） |
| `src/transformer/dataset/gigamidi/gigamidi_processed_nodrums{,_v2,_v3}/` | 共 914 MB | 路线 A 第 4 步 |
| `src/transformer/dataset/data/` | 122 MB | 路线 B 第 3 步 |
| `src/transformer/dataset/gp_dataset/` + `surname_checked_midis_v1.2.zip` | 197 MB + 129 MB | 路线 C 第 1 步 |
| `src/transformer/dataset/giant_piano_mind_v1.pt` | 1.1 GB | 路线 C 第 2 步 |
| `src/transformer/dataset/classical_gpt_dataset_smart{,_v2}.pt` | 共 171 MB | 路线 B 第 3 步 |
| `src/VAE/train/classical_dataset.pt` | 690 MB | 路线 B 第 2 步 |
| `src/VAE/train/midi_dataset_local/` | 155 MB | 路线 B 第 1 步 |

**一个都没动的**：全部 `.py`、训练日志（`logs_gpt*/`）、模型权重
（`transformer/checkpoints_gpt/` 4.3 GB、`transformer/final_models/` 452 MB、
`VAE/train/checkpoints/` 471 MB），以及 GigaMIDI 的数据卡 `gigamidi/README.md`。

> 📌 `src/transformer/dataset/gigamidi/` 下的 `preprocess.py` 和 `test.py` 原本**不在版本库里**
> —— `.gitignore` 第 236 行把整个 `gigamidi/` 目录都挡掉了。它们是路线 A 唯一的实现，
> 已用 `git add -f` 强制入库，以后再清目录不会跟着数据一起消失。

## 分工和致谢

<!--
- 感谢ybSun撰写项目代码主体的贡献；
- 感谢ybSun课题组为模型训练提供算力支持；
- 感谢ZZK的火腿肠对ybSun的精神支持；
- 感谢全体组员容忍我写出如此抽象的致谢skr\~。
- 骇死助教！！！
-->

- (组长)孙韫博：
  - 开发遗传算法引擎
  - 提出TAGA架构和相应变异与遗传算子
  - 在服务器上训练Transformer模型并进行测试
  - 编写了文章中算法架构的部分
- 夏省玘：
  - 协助开发完善代码，撰写说明文档
  - 进行消融实验测试
  - 编写了乐谱的可视化程序
- 邵柏睿：
  - 构建符合乐理的遗传算法规则，完善传统遗传算法程序
  - 纯遗传算法部分的实验和文章撰写
- 赵泽凯：
  - 编写文章的结论部分
  - 排版并修改润色文章
- 于秋雨：
  - 文章的导论与背景部分
  - 相关领域的文献调研

