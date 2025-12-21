import torch
import torch.nn.functional as F
import numpy as np
import random
import os

from .model import MusicGPT, GPT_CONFIG_nano

# 采用绝对路径导入 MusicRep（如果可用）
MUSICREP_AVAILABLE = False
try:
    from MusicRep.melody_sqeuence import MelodySequence
    from MusicRep.synthesizer import Synthesizer, StringStrategy
    MUSICREP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: 'MusicRep' library not found ({e}). WAV rendering will be disabled.")
    MUSICREP_AVAILABLE = False

class GPTMusicEvaluator:
    """Evaluator wrapper for trained MusicGPT: fitness + generation."""

    def __init__(self, model_path: str, device: str | None = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚖️ Using device: {self.device}")

        # Load checkpoint
        print(f"   Loading checkpoint from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Recover config
        self.config = checkpoint.get("config", {})
        if not self.config:
            print("Warning: Model config not found in checkpoint. Using default GPT_CONFIG.")
            self.config["model_config"] = GPT_CONFIG_nano

        # Ensure vocab size for BOS/EOS
        #self.config["model_config"]["vocab_size"] = 130

        # Build model
        self.model = MusicGPT(self.config["model_config"])

        # Load weights (strip DataParallel prefixes if any)
        state_dict = checkpoint["model_state_dict"]
        unwrapped_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(unwrapped_state_dict)

        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded and ready.")

    # ---------- Fitness ----------
    def evaluate(self, population_grid: np.ndarray) -> np.ndarray:
        """Batch fitness: higher is better. population_grid shape [B, T]."""
        scores = []
        with torch.no_grad():
            for seq in population_grid:
                seq_tensor = torch.tensor(seq, dtype=torch.long, device=self.device).unsqueeze(0)
                if seq_tensor.size(1) < 2:
                    scores.append(0.0)
                    continue
                X = seq_tensor[:, :-1]
                Y = seq_tensor[:, 1:]
                _, loss = self.model(X, targets=Y)
                scores.append(1.0 / (loss.item() + 1e-6))
        return np.array(scores, dtype=np.float32)

    # ---------- Single fitness helper ----------
    def get_fitness_score(self, sequence: list[int]) -> float:
        return float(self.evaluate(np.array([sequence]))[0])

    # ---------- Generation ----------
    def generate(self, prompt_sequence: list[int], max_new_tokens: int, temperature: float = 0.8, top_k: int = 20) -> list[int]:
        prompt_tensor = torch.tensor(prompt_sequence, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            generated_tensor = self.model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        return generated_tensor.squeeze(0).tolist()
    def generate_with_logit_bias(self, prompt_sequence: list[int], target_sequence: list[int], start_idx: int, alpha: float = 3.0, temperature: float = 1.0, top_k: int = 20) -> list[int]:
        """
        高效版的 Logit Bias 生成，支持 Batch 操作（虽然 GA 中通常 B=1，但这样写扩展性更好）。
        
        优化点：
        1. 向量化操作：避免 Python 循环，直接使用 Tensor 索引注入 Bias。
        2. 显存优化：及时裁剪 context。
        
        Args:
            prompt_sequence: 父代 A 的前半段 (Context) [List or 1D Tensor]
            target_sequence: 父代 B 的完整序列 (用于提供 Bias) [List or 1D Tensor]
            start_idx: 生成起始位置
            alpha: 混合强度
        """
        # 1. 数据准备 (转为 Tensor 并确保维度为 (B, T))
        # 这里的 B=1，但代码逻辑兼容 B>1
        device = self.device
        
        # 如果传入的是 list，转 tensor；如果是 tensor，确保在 device 上
        if isinstance(prompt_sequence, list):
            current_seq = torch.tensor([prompt_sequence], dtype=torch.long, device=device)
        else:
            current_seq = prompt_sequence.to(device).unsqueeze(0) if prompt_sequence.dim() == 1 else prompt_sequence

        if isinstance(target_sequence, list):
            target_seq = torch.tensor([target_sequence], dtype=torch.long, device=device)
        else:
            target_seq = target_sequence.to(device).unsqueeze(0) if target_sequence.dim() == 1 else target_sequence

        block_size = self.config["model_config"]["block_size"]
        total_len = target_seq.size(1)
        batch_size = current_seq.size(0)

        # 预先生成 batch 索引 [0, 1, ... B-1]，用于后续的高级索引
        batch_indices = torch.arange(batch_size, device=device)

        self.model.eval()
        with torch.no_grad():
            for i in range(start_idx, total_len):
                # 2. Context 裁剪 (滑动窗口)
                # 这一步至关重要，防止序列过长导致报错，同时减少计算量
                if current_seq.size(1) <= block_size:
                    idx_cond = current_seq
                else:
                    idx_cond = current_seq[:, -block_size:]

                # 3. Forward Pass
                # Model returns (logits, loss)
                logits, _ = self.model(idx_cond)
                # 取最后一个 token 的 logits: (B, Vocab_Size)
                next_token_logits = logits[:, -1, :]

                # 4. 向量化注入 Bias (Vectorized Bias Injection)
                # 获取当前步 Target 序列对应的 Token
                # shape: (B,)
                if i < target_seq.size(1):
                    target_tokens = target_seq[:, i]
                    
                    # 只有合法的 token 才加 bias (过滤 PAD 或越界值)
                    # 假设 vocab_size 内的都是合法 token
                    vocab_size = next_token_logits.size(-1)
                    valid_mask = (target_tokens >= 0) & (target_tokens < vocab_size)
                    
                    # 核心优化：利用高级索引直接修改，无 Python 循环
                    # next_token_logits[b, target_token[b]] += alpha
                    if valid_mask.all():
                        next_token_logits[batch_indices, target_tokens] += alpha
                    else:
                        # 处理部分 batch 可能 finish 或 padding 的情况
                        valid_indices = batch_indices[valid_mask]
                        valid_targets = target_tokens[valid_mask]
                        next_token_logits[valid_indices, valid_targets] += alpha

                # 5. 采样 (Sampling)
                # Temperature
                next_token_logits = next_token_logits / temperature

                # Top-K (Vectorized)
                if top_k is not None:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    # 将小于第 k 大概率的值设为 -inf
                    # v[:, [-1]] 会自动广播到 (B, Vocab)
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

                # Softmax & Multinomial
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1) # (B, 1)

                # 6. 拼接
                current_seq = torch.cat((current_seq, next_token), dim=1)

        self.model.train()
        
        # 返回 List[int] (如果是单条)
        return current_seq[0].tolist()
        

# ---------- Utility: tokens -> MusicRep grid ----------
def tokens_to_melodygrid(tokens):
    grid = np.array(tokens)
    pitches = grid[grid >= 2]
    grid[grid >= 2] = pitches - 2
    return grid


if __name__ == "__main__":
    MODEL_PATH = "./transformer/checkpoints_gpt/music_gpt_v1_best.pth"
    DATA_PATH = "./transformer/dataset/classical_gpt_dataset_smart_v2.pt"
    BOS_TOKEN = 130
    OUTPUT_FOLDER = "example_outputs/transformer_generated/"

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH}. Please check the path.")

    evaluator = GPTMusicEvaluator(model_path=MODEL_PATH)

    print("\n" + "=" * 50)
    print("--- 1. Testing Fitness Evaluation ---")
    print("=" * 50)

    dataset = torch.load(DATA_PATH, weights_only=False)
    good_sequence = dataset[10][:32]
    bad_sequence = [random.randint(0, 129) for _ in range(32)]
    boring_sequence = [60, 1, 62, 1, 64, 1, 62, 1] * 4

    good_fitness = evaluator.get_fitness_score(list(good_sequence))
    bad_fitness = evaluator.get_fitness_score(bad_sequence)
    boring_fitness = evaluator.get_fitness_score(boring_sequence)

    print(f"Fitness of a REAL music snippet:   {good_fitness:.4f}")
    print(f"Fitness of a BORING sequence:     {boring_fitness:.4f}")
    print(f"Fitness of RANDOM NOISE:          {bad_fitness:.4f}")

    print("\n" + "=" * 50)
    print("--- 2. Testing Melody Generation ---")
    print("=" * 50)

    prompt_from_scratch = [BOS_TOKEN]
    conservative_melody = evaluator.generate(
        prompt_sequence=prompt_from_scratch,
        max_new_tokens=128,
        temperature=1.5,
        top_k=20,
    )
    print(f"Generated sequence (first 32 tokens): {conservative_melody[1:33]}")

    c_major_prompt = [BOS_TOKEN, 60 + 2, 64 + 2, 67 + 2]
    creative_melody = evaluator.generate(
        prompt_sequence=c_major_prompt,
        max_new_tokens=128,
        temperature=1.5,
        top_k=50,
    )
    print(f"Generated sequence (first 32 tokens): {creative_melody[0:128]}")

    if MUSICREP_AVAILABLE:
        print("\n" + "=" * 50)
        print("--- 3. Rendering to WAV files ---")
        print("=" * 50)

        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        synth = Synthesizer(strategy=StringStrategy())

        conservative_grid = tokens_to_melodygrid(conservative_melody)
        conservative_path = os.path.join(OUTPUT_FOLDER, "conservative_melody.wav")
        synth.render(conservative_grid, bpm=100, output_path=conservative_path)
        print(f"✅ Conservative melody saved to: {conservative_path}")

        creative_grid = tokens_to_melodygrid(creative_melody)
        creative_path = os.path.join(OUTPUT_FOLDER, "creative_melody.wav")
        synth.render(creative_grid, bpm=100, output_path=creative_path)
        print(f"✅ Creative melody saved to: {creative_path}")
    