import os
import pandas as pd
from pathlib import Path
from datasets import Dataset, DatasetDict
from symusic import Score
import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import warnings
import ast
import numpy as np


# ==============================================================================
# 1. 配置 (几乎不需要修改)
# ==============================================================================
METADATA_CSV_PATH = "metadata.csv" 
DATASET_BASE_DIR = "."
OUTPUT_DIR = "./gigamidi_processed_nodrums_v3"
CHUNK_SIZE = 1000
NUM_PROCESSES = cpu_count() - 1 or 1

warnings.filterwarnings("ignore", category=UserWarning, module='symusic.*')

# ==============================================================================
# 2. “主旋律侦探”函数 (与之前相同)
# ==============================================================================
def find_melody_track_index_from_paper(sample):
    # ... (此处省略，与上一版本完全相同)
    required_keys = ["NOMML", "instrument_group (expressive)", 
                     "note_density (expressive)", "loopability (expressive)"]
    if not all(isinstance(sample.get(key), list) and sample.get(key) for key in required_keys):
        return None
    nomml_scores = sample["NOMML"]
    if not nomml_scores or not isinstance(nomml_scores, list): return None
    NOMML_THRESHOLD = 12
    expressive_candidates = []
    metadata_len = min(
        len(sample["instrument_group (expressive)"]),
        len(sample["note_density (expressive)"]),
        len(sample["loopability (expressive)"])
    )
    for i in range(min(len(nomml_scores), metadata_len)):
        if nomml_scores[i] >= NOMML_THRESHOLD:
            candidate_info = {
                "track_index": i, "instrument_group": sample["instrument_group (expressive)"][i],
                "note_density": sample["note_density (expressive)"][i], "loopability": sample["loopability (expressive)"][i]
            }
            expressive_candidates.append(candidate_info)
    if not expressive_candidates: return None
    if len(expressive_candidates) == 1: return expressive_candidates[0]["track_index"]
    melodic_scores = {
        "Piano": 5, "Guitar": 5, "Reed": 5, "Strings": 5, "Synth Lead": 5, "Vocal": 6, "Brass": 4, "Pipe": 3,
        "Ethnic": 3, "Ensemble": 1, "Synth Pad": 1, "Chromatic Percussion": 1, "Bass": -2, "Synth Effects": -5,
        "Drums": -10, "Percussive": -10,
    }
    best_track_index, max_final_score = -1, -float('inf')
    for candidate in expressive_candidates:
        group, note_density, loopability = candidate["instrument_group"], candidate["note_density"], candidate["loopability"]
        tendency_score = melodic_scores.get(group, 1)
        final_score = (tendency_score * note_density) * (1.1 - loopability)
        if final_score > max_final_score:
            max_final_score, best_track_index = final_score, candidate["track_index"]
    return best_track_index


# ==============================================================================
# 3. 为单个进程设计的工作函数 (Worker Function) (与之前相同)
# ==============================================================================
def process_chunk(chunk_df):
    """
    处理一个数据块：加载 MIDI -> 单轨化(最高音) -> 转化为 Token -> 数据清洗
    """
    processed_data = []
    
    # ============================
    # 1. 核心常量配置
    # ============================
    TOKEN_REST = 0
    TOKEN_SUSTAIN = 1
    PITCH_OFFSET = 2
    
    # 清洗阈值
    MIN_SEQ_LEN = 32          # 最终序列允许的最短长度 (Token 数)
    MAX_CONSECUTIVE_REST = 8 # 允许最大的连续休止符数量 (建议 16=2小节，若想激进压缩可改为 4)
    MIN_NOTE_DENSITY = 0.3    # 最小音符密度 (有声部分占比至少 10%)
    MIN_UNIQUE_PITCHES = 5    # 最小音高种类 (防止单调重复)
    VALID_PITCH_RANGE = (21, 108) # 有效音域 (钢琴 88 键 A0-C8)

    for _, row in chunk_df.iterrows():
        # 获取路径
        relative_path = row.iloc[0]
        midi_file_path = Path(DATASET_BASE_DIR) / relative_path
        
        if not midi_file_path.exists(): 
            continue
        
        # 获取元数据和旋律轨索引
        metadata_dict = row.to_dict()
        melody_track_idx = find_melody_track_index_from_paper(metadata_dict)
        
        if melody_track_idx is None: 
            continue
        
        try:
            # 加载 MIDI
            score = Score(midi_file_path)
            if melody_track_idx >= len(score.tracks): 
                continue
            
            melody_track = score.tracks[melody_track_idx]
            notes = melody_track.notes
            if not notes: 
                continue
            
            # 计算时间步 (Grid Quantization)
            tpqn = score.tpq
            if tpqn == 0: continue
            
            eighth_note_ticks = tpqn / 2.0
            total_ticks = max(n.end for n in notes)
            total_steps = math.ceil(total_ticks / eighth_note_ticks)
            
            # 基础长度过滤 (防止极短文件)
            if total_steps < MIN_SEQ_LEN: 
                continue
            # 防止内存爆炸 (过滤极长文件，例如超过 50000 步)
            if total_steps > 50000:
                continue

            # ==================================================================
            # 2. 单轨化逻辑：最高音优先 (High Pitch Priority)
            # ==================================================================
            
            # step_pitches: 记录每个时间步的真实 MIDI 音高 (-1 代表休止)
            # step_is_onset: 记录该时间步是否是音符的起始点 (True=NoteOn, False=Sustain)
            step_pitches = [-1] * total_steps
            step_is_onset = [False] * total_steps
            
            for note in notes:
                start_step = math.floor(note.start / eighth_note_ticks)
                end_step = math.ceil(note.end / eighth_note_ticks)
                
                # 越界保护
                if start_step >= total_steps: continue
                real_end = min(end_step, total_steps)
                
                for step in range(start_step, real_end):
                    current_highest = step_pitches[step]
                    
                    # 逻辑 A: 新音符比当前位置高 -> 无条件覆盖
                    if note.pitch > current_highest:
                        step_pitches[step] = note.pitch
                        # 只有在音符的起始步，才标记为 Onset
                        step_is_onset[step] = (step == start_step)
                    
                    # 逻辑 B: 音高相同，但这是新音符的起始 (同音反复 Re-articulation) -> 强制标记为 Onset
                    elif note.pitch == current_highest:
                        if step == start_step:
                            step_is_onset[step] = True

            # ==================================================================
            # 3. 原始 Token 序列生成
            # ==================================================================
            raw_tokens = []
            for step in range(total_steps):
                pitch = step_pitches[step]
                is_onset = step_is_onset[step]
                
                if pitch == -1:
                    token = TOKEN_REST
                elif is_onset:
                    # 检查音域，过滤超低/超高噪音
                    if VALID_PITCH_RANGE[0] <= pitch <= VALID_PITCH_RANGE[1]:
                        token = pitch + PITCH_OFFSET
                    else:
                        token = TOKEN_REST 
                else:
                    token = TOKEN_SUSTAIN
                raw_tokens.append(token)

            # ==================================================================
            # 4. 数据清洗流水线 (Data Cleaning Pipeline)
            # ==================================================================
            
            arr = np.array(raw_tokens)
            
            # --- Cleaning A: 去头去尾 (Trim Silence) ---
            non_rest_indices = np.where(arr != TOKEN_REST)[0]
            if len(non_rest_indices) == 0: 
                continue 
            
            start_idx = non_rest_indices[0]
            end_idx = non_rest_indices[-1]
            trimmed_tokens = raw_tokens[start_idx : end_idx + 1]
            
            if len(trimmed_tokens) < MIN_SEQ_LEN: 
                continue

            # --- Cleaning B: 压缩冗长休止符 (Smart Silence Compression) ---
            compressed_tokens = []
            rest_count = 0
            
            for token in trimmed_tokens:
                if token == TOKEN_REST:
                    rest_count += 1
                    # 只有未超过阈值时才添加
                    if rest_count <= MAX_CONSECUTIVE_REST:
                        compressed_tokens.append(token)
                else:
                    rest_count = 0
                    compressed_tokens.append(token)
            
            # --- Cleaning C: 质量检测 (Quality Check) ---
            final_arr = np.array(compressed_tokens)
            
            # C1. 长度复查
            if len(compressed_tokens) < MIN_SEQ_LEN: 
                continue

            # C2. 音符密度检测 (Density)
            # 计算 Note-On 和 Sustain 的总占比
            sound_tokens = final_arr[final_arr != TOKEN_REST]
            density = len(sound_tokens) / len(final_arr)
            if density < MIN_NOTE_DENSITY:
                continue

            # C3. 音高丰富度检测 (Variety)
            # 只统计 Note-On 的音高
            pitch_tokens = final_arr[final_arr >= PITCH_OFFSET]
            unique_pitches = len(np.unique(pitch_tokens))
            if unique_pitches < MIN_UNIQUE_PITCHES:
                continue

            # ==================================================================
            # 5. 保存结果
            # ==================================================================
            processed_data.append({
                "input_ids": compressed_tokens, 
                "source_file": str(relative_path), 
                "split": row['split']
            })
            
        except Exception:
            # 遇到任何解析错误跳过文件，不中断进程
            continue
            
    return processed_data
# ==============================================================================
# 4. 主执行流程 (*** 主要改动在这里 ***)
# ==============================================================================
def main():
    print("Step 1: Loading metadata CSV...")
    df = pd.read_csv(METADATA_CSV_PATH)
    # 假设 CSV 的第一列是文件路径
    path_column_name = df.columns[0] 
    print(f"Loaded {len(df)} total metadata entries.")

    print("\nStep 2: Filtering metadata based on path...")
    
    # 定义我们想要的路径关键词
    # 路径中必须包含 'no-drums'
    # 并且必须包含 'training', 'test', 或 'validation' 中的一个
    def filter_paths(path):
        if not isinstance(path, str):
            return False
        # 使用 Path 对象来方便地处理路径部分
        p = Path(path)
        parts = p.parts
        
        has_split = any(part.startswith(s) for part in parts for s in ["training", "test", "validation"])
        has_nodrums = "no-drums" in parts
        
        return has_split and has_nodrums

    # 应用筛选器
    original_count = len(df)
    df_filtered = df[df[path_column_name].apply(filter_paths)].copy() # 使用 .copy() 避免 SettingWithCopyWarning
    print(f"Filtered down to {len(df_filtered)} entries matching the criteria (from {original_count}).")

    # *** 新增：从路径中提取 train/test/validation 划分信息 ***
    def get_split_from_path(path):
        if "training" in path:
            return "train"
        elif "test" in path:
            return "test"
        elif "validation" in path:
            return "validation"
        return "unknown"
        
    df_filtered['split'] = df_filtered[path_column_name].apply(get_split_from_path)
    
    print("Assigned splits to the data:")
    print(df_filtered['split'].value_counts())


    print("\nStep 3: Preparing data for parallel processing...")
    # (与之前类似，但现在处理的是过滤后的 DataFrame)
    for col in ["NOMML", "instrument_group (expressive)", "note_density (expressive)", "loopability (expressive)"]:
        if col in df_filtered.columns and isinstance(df_filtered[col].iloc[0], str):
            print(f"Converting column '{col}' from string to list...")
            df_filtered[col] = df_filtered[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    chunks = [df_filtered.iloc[i:i + CHUNK_SIZE] for i in range(0, len(df_filtered), CHUNK_SIZE)]
    print(f"Split filtered data into {len(chunks)} chunks.")

    all_processed_data = []
    
    print("\nStep 4: Starting multiprocessing pool...")
    with Pool(processes=NUM_PROCESSES) as pool:
        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for result_chunk in pool.imap_unordered(process_chunk, chunks):
                all_processed_data.extend(result_chunk)
                pbar.update()

    print(f"\nStep 5: Processing complete. Successfully processed {len(all_processed_data)} files.")
    if not all_processed_data:
        print("No data was processed successfully.")
        return

    print("\nStep 6: Converting to Hugging Face DatasetDict...")
    
    # 根据我们之前添加的 'split' 列来组织数据
    train_data = [d for d in all_processed_data if d['split'] == 'train']
    test_data = [d for d in all_processed_data if d['split'] == 'test']
    validation_data = [d for d in all_processed_data if d['split'] == 'validation']

    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'test': Dataset.from_list(test_data),
        'validation': Dataset.from_list(validation_data)
    })

    print("Final dataset structure:")
    print(dataset_dict)
    
    dataset_dict.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved successfully to {OUTPUT_DIR}")

    print("\n--- Verification ---")
    reloaded_dataset = DatasetDict.load_from_disk(OUTPUT_DIR)
    print("Dataset reloaded instantly!")
    print(reloaded_dataset)
    print("A sample from the training set:", reloaded_dataset['train'][0]['input_ids'][:30])

if __name__ == "__main__":
    main()