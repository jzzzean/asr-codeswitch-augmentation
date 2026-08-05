# -*- coding: utf-8 -*-
import os
import random
import pandas as pd
import csv
from tqdm import tqdm
import traceback
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

# --- 配置參數 ---
NUM_SAMPLES_TO_GENERATE = 200
NUM_ZH_SAMPLES = 1000
NUM_EN_SAMPLES = 500
TARGET_SR = 16000

# 論文方法的關鍵參數
MIN_SNIPPET_LENGTH = 1.0  # 最短片段長度（秒）- 降低要求
TARGET_SNIPPET_LENGTH = 3.5  # 目標片段長度（秒）
MAX_SNIPPET_LENGTH = 5.0  # 最長片段長度（秒）
MAX_TOTAL_LENGTH = 30.0   # Whisper的最大長度限制（秒）
MIN_TOTAL_LENGTH = 15.0   # 最小總長度，確保有足夠的語碼轉換

from data_augmentation import add_background_noise, apply_perturbation

# Local paths. Set ASR_DATA_DIR to keep datasets outside the repository.
DATA_DIR = Path(os.environ.get("ASR_DATA_DIR", "data"))
AUG_DATA_BASE_PATH = str(DATA_DIR / "augmented_data")
CONCAT_OUTPUT_PATH = os.path.join(AUG_DATA_BASE_PATH, "test_concat")
NOISY_OUTPUT_PATH = os.path.join(AUG_DATA_BASE_PATH, "test_noisy")
PERTURBED_OUTPUT_PATH = os.path.join(AUG_DATA_BASE_PATH, "test_perturbed")

NOISE_DIR = str(DATA_DIR / "raw_data" / "noise")
NOISE_FILE_PATH = os.path.join(NOISE_DIR, "NOISE.wav")

ZH_TSV_PATH = str(DATA_DIR / "raw_data" / "common_voice_zh" / "transcript" / "train.tsv")
ZH_AUDIO_DIR = str(DATA_DIR / "raw_data" / "common_voice_zh" / "audio")
EN_TSV_PATH = str(DATA_DIR / "raw_data" / "common_voice_en" / "transcript" / "train.tsv")
EN_AUDIO_DIR = str(DATA_DIR / "raw_data" / "common_voice_en" / "audio")

def build_audio_index(base_dir):
    """建立音檔索引"""
    audio_index = {}
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.endswith(('.wav', '.mp3', '.flac')):
                audio_index[f] = os.path.join(root, f)
    return audio_index

def create_snippet(audio_path, text, target_length=None):
    """
    從音檔中創建指定長度的片段（模擬fine-grained alignment的結果）
    
    Args:
        audio_path: 音檔路徑
        text: 對應文字
        target_length: 目標長度（秒），如果為None則使用整個音檔
    
    Returns:
        tuple: (audio_data, actual_duration, text) 或 None
    """
    try:
        y, sr = librosa.load(audio_path, sr=TARGET_SR)
        original_duration = len(y) / TARGET_SR
        
        # 如果音檔太短，直接跳過
        if original_duration < MIN_SNIPPET_LENGTH:
            return None
        
        # 如果沒有指定目標長度，或者原音檔就在合適範圍內
        if target_length is None:
            if original_duration <= MAX_SNIPPET_LENGTH:
                return y, original_duration, text
            else:
                # 音檔過長，隨機截取
                target_length = min(MAX_SNIPPET_LENGTH, 
                                  max(MIN_SNIPPET_LENGTH, TARGET_SNIPPET_LENGTH))
        
        # 確保目標長度不超過原音檔長度
        target_length = min(target_length, original_duration)
        
        # 隨機選擇起始位置
        target_samples = int(target_length * TARGET_SR)
        max_start = max(0, len(y) - target_samples)
        start_sample = random.randint(0, max_start)
        
        snippet = y[start_sample:start_sample + target_samples]
        actual_duration = len(snippet) / TARGET_SR
        
        return snippet, actual_duration, text
        
    except Exception as e:
        print(f"創建片段失敗 {audio_path}: {e}")
        return None

def recursive_concatenation(zh_samples, en_samples, target_id):
    """
    遞迴拼接方法：持續添加片段直到達到目標長度
    
    Args:
        zh_samples: 中文樣本列表
        en_samples: 英文樣本列表
        target_id: 目標ID（用於文件命名）
    
    Returns:
        tuple: (concatenated_audio, total_transcript, success)
    """
    try:
        concatenated_audio = np.array([])
        total_transcript = ""
        total_duration = 0.0
        segment_count = 0
        language_switches = 0
        last_language = None
        
        # 隨機決定起始語言
        current_language = random.choice(['zh', 'en'])
        
        while total_duration < MIN_TOTAL_LENGTH or (total_duration < MAX_TOTAL_LENGTH and random.random() < 0.7):
            # 剩余可用時間
            remaining_time = MAX_TOTAL_LENGTH - total_duration
            if remaining_time < MIN_SNIPPET_LENGTH:
                break
            
            # 選擇樣本
            if current_language == 'zh':
                sample = random.choice(zh_samples)
                samples_list = zh_samples
            else:
                sample = random.choice(en_samples)
                samples_list = en_samples
            
            # 計算這個片段的目標長度
            if remaining_time < TARGET_SNIPPET_LENGTH:
                snippet_target_length = remaining_time
            else:
                # 隨機選擇片段長度，但考慮剩余時間
                max_possible = min(MAX_SNIPPET_LENGTH, remaining_time)
                snippet_target_length = random.uniform(MIN_SNIPPET_LENGTH, max_possible)
            
            # 創建片段
            snippet_result = create_snippet(
                sample['audio_path'], 
                sample['text'], 
                snippet_target_length
            )
            
            if snippet_result is None:
                # 如果這個樣本失敗，嘗試其他樣本
                continue
            
            snippet_audio, snippet_duration, snippet_text = snippet_result
            
            # 檢查是否會超出總長度限制
            if total_duration + snippet_duration > MAX_TOTAL_LENGTH:
                # 截斷音頻以適應限制
                max_samples = int((MAX_TOTAL_LENGTH - total_duration) * TARGET_SR)
                if max_samples > 0:
                    snippet_audio = snippet_audio[:max_samples]
                    snippet_duration = len(snippet_audio) / TARGET_SR
                else:
                    break
            
            # 添加到總音頻
            concatenated_audio = np.concatenate([concatenated_audio, snippet_audio])
            total_duration += snippet_duration
            segment_count += 1
            
            # 更新文字稿
            if total_transcript:
                total_transcript += " " + snippet_text
            else:
                total_transcript = snippet_text
            
            # 語言切換邏輯
            if last_language is not None and last_language != current_language:
                language_switches += 1
            last_language = current_language
            
            # 決定下一個片段的語言（更高的切換概率以模擬code-switching）
            if random.random() < 0.4:  # 40%概率切換語言
                current_language = 'en' if current_language == 'zh' else 'zh'
            
            # 安全檢查：避免無限循環
            if segment_count >= 20:  # 最多20個片段
                break
        
        # 檢查是否滿足最小要求
        if total_duration < MIN_TOTAL_LENGTH or segment_count < 2 or language_switches < 1:
            return None, None, False
        
        print(f"樣本 {target_id}: {segment_count} 個片段, {language_switches} 次語言切換, 總長度 {total_duration:.2f}s")
        return concatenated_audio, total_transcript.strip(), True
        
    except Exception as e:
        print(f"遞迴拼接失敗 {target_id}: {e}")
        return None, None, False

class PaperStyleDatasetManager:
    """基於論文方法的資料集管理器"""
    
    def __init__(self, zh_dir, en_dir):
        print("建立音檔索引...")
        self.zh_audio_index = build_audio_index(zh_dir)
        self.en_audio_index = build_audio_index(en_dir)
        print(f"中文音檔: {len(self.zh_audio_index)}, 英文音檔: {len(self.en_audio_index)}")
        
        self.zh_samples = []
        self.en_samples = []

    def load_datasets(self, num_zh=1000, num_en=500):
        """載入並預處理資料集"""
        # 載入中文
        try:
            zh_df = pd.read_csv(ZH_TSV_PATH, sep="\t", low_memory=False)
            zh_df["audio_path"] = zh_df["path"].apply(lambda x: self.zh_audio_index.get(x, None))
            zh_valid = zh_df[zh_df["audio_path"].notnull()]
            
            if num_zh < len(zh_valid):
                zh_valid = zh_valid.sample(num_zh, random_state=42)
            
            self.zh_samples = [{"audio_path": r["audio_path"], "text": r["sentence"]}
                             for _, r in zh_valid.iterrows()]
            print(f"載入中文樣本: {len(self.zh_samples)}")
        except Exception as e:
            print(f"載入中文失敗: {e}")
            return False

        # 載入英文
        try:
            en_df = pd.read_csv(EN_TSV_PATH, sep="\t", low_memory=False)
            en_df["audio_path"] = en_df["path"].apply(lambda x: self.en_audio_index.get(x, None))
            en_valid = en_df[en_df["audio_path"].notnull()]
            
            if num_en < len(en_valid):
                en_valid = en_valid.sample(num_en, random_state=42)
            
            self.en_samples = [{"audio_path": r["audio_path"], "text": r["sentence"]}
                             for _, r in en_valid.iterrows()]
            print(f"載入英文樣本: {len(self.en_samples)}")
        except Exception as e:
            print(f"載入英文失敗: {e}")
            return False

        return True

def generate_paper_style_augmentation():
    """基於論文方法的資料增強生成"""
    print("=== 基於論文方法的資料增強 ===")
    
    # 建立輸出目錄
    os.makedirs(CONCAT_OUTPUT_PATH, exist_ok=True)
    os.makedirs(NOISY_OUTPUT_PATH, exist_ok=True)
    os.makedirs(PERTURBED_OUTPUT_PATH, exist_ok=True)
    
    # 初始化資料集管理器
    dataset_manager = PaperStyleDatasetManager(ZH_AUDIO_DIR, EN_AUDIO_DIR)
    if not dataset_manager.load_datasets(NUM_ZH_SAMPLES, NUM_EN_SAMPLES):
        print("資料集載入失敗")
        return
    
    metadata = []
    successful_samples = 0
    failed_samples = 0
    
    # 1. 生成遞迴拼接樣本
    print(f"\n開始生成 {NUM_SAMPLES_TO_GENERATE} 個遞迴拼接樣本...")
    for i in tqdm(range(NUM_SAMPLES_TO_GENERATE), desc="遞迴拼接"):
        try:
            # 遞迴拼接
            concat_audio, concat_transcript, success = recursive_concatenation(
                dataset_manager.zh_samples, 
                dataset_manager.en_samples, 
                i
            )
            
            if not success:
                failed_samples += 1
                continue
            
            # 儲存拼接結果
            concat_filename = f"concat_{i:04d}.wav"
            concat_path = os.path.join(CONCAT_OUTPUT_PATH, concat_filename)
            sf.write(concat_path, concat_audio, TARGET_SR)
            
            # 記錄metadata
            duration = len(concat_audio) / TARGET_SR
            metadata.append([
                os.path.join("test_concat", concat_filename),
                concat_transcript,
                "concat",
                f"duration={duration:.2f}s"
            ])
            successful_samples += 1
            
            # 2. 基於拼接樣本生成噪音版本
            if os.path.exists(NOISE_FILE_PATH):
                noisy_filename = f"noisy_{i:04d}.wav"
                noisy_path = os.path.join(NOISY_OUTPUT_PATH, noisy_filename)
                
                noisy_transcript, noisy_success = add_background_noise(
                    concat_path,
                    NOISE_FILE_PATH,
                    noisy_path,
                    snr_db=random.uniform(5, 15),
                    text=concat_transcript
                )
                
                if noisy_success:
                    metadata.append([
                        os.path.join("test_noisy", noisy_filename),
                        noisy_transcript,
                        "noisy",
                        f"base_duration={duration:.2f}s"
                    ])
            
            # 3. 基於拼接樣本生成擾動版本
            perturbed_filename = f"perturbed_{i:04d}.wav"
            perturbed_path = os.path.join(PERTURBED_OUTPUT_PATH, perturbed_filename)
            
            perturbed_transcript, perturbed_success = apply_perturbation(
                concat_path,
                perturbed_path,
                speed_factor=random.uniform(0.9, 1.1),
                pitch_shift_semitones=random.randint(-2, 2),
                text=concat_transcript
            )
            
            if perturbed_success:
                metadata.append([
                    os.path.join("test_perturbed", perturbed_filename),
                    perturbed_transcript,
                    "perturbed",
                    f"base_duration={duration:.2f}s"
                ])
                
        except Exception as e:
            print(f"樣本 {i} 生成失敗: {e}")
            failed_samples += 1
    
    # 儲存metadata
    csv_path = os.path.join(AUG_DATA_BASE_PATH, "metadata.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "transcript", "augmentation_type", "extra_info"])
        writer.writerows(metadata)
    
    print(f"\n=== 生成完成 ===")
    print(f"成功樣本: {successful_samples}")
    print(f"失敗樣本: {failed_samples}")
    print(f"總計metadata: {len(metadata)} 筆")
    print(f"平均每個拼接樣本產生: {len(metadata)/max(successful_samples, 1):.1f} 個增強變體")

def analyze_dataset_lengths():
    """分析資料集中音檔長度分布"""
    print("=== 分析音檔長度分布 ===")
    
    dataset_manager = PaperStyleDatasetManager(ZH_AUDIO_DIR, EN_AUDIO_DIR)
    dataset_manager.load_datasets(200, 200)  # 載入較少樣本用於分析
    
    zh_durations = []
    en_durations = []
    
    # 分析中文音檔
    print("分析中文音檔長度...")
    for sample in tqdm(dataset_manager.zh_samples[:100], desc="中文"):
        try:
            duration = librosa.get_duration(filename=sample['audio_path'])
            zh_durations.append(duration)
        except:
            continue
    
    # 分析英文音檔  
    print("分析英文音檔長度...")
    for sample in tqdm(dataset_manager.en_samples[:100], desc="英文"):
        try:
            duration = librosa.get_duration(filename=sample['audio_path'])
            en_durations.append(duration)
        except:
            continue
    
    # 統計結果
    if zh_durations:
        print(f"\n中文音檔長度統計 (n={len(zh_durations)}):")
        print(f"  平均: {np.mean(zh_durations):.2f}s")
        print(f"  中位數: {np.median(zh_durations):.2f}s")
        print(f"  範圍: {np.min(zh_durations):.2f}s - {np.max(zh_durations):.2f}s")
        print(f"  ≥{MIN_SNIPPET_LENGTH}s: {sum(d >= MIN_SNIPPET_LENGTH for d in zh_durations)} ({100*sum(d >= MIN_SNIPPET_LENGTH for d in zh_durations)/len(zh_durations):.1f}%)")
    
    if en_durations:
        print(f"\n英文音檔長度統計 (n={len(en_durations)}):")
        print(f"  平均: {np.mean(en_durations):.2f}s")
        print(f"  中位數: {np.median(en_durations):.2f}s")
        print(f"  範圍: {np.min(en_durations):.2f}s - {np.max(en_durations):.2f}s")
        print(f"  ≥{MIN_SNIPPET_LENGTH}s: {sum(d >= MIN_SNIPPET_LENGTH for d in en_durations)} ({100*sum(d >= MIN_SNIPPET_LENGTH for d in en_durations)/len(en_durations):.1f}%)")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_dataset_lengths()
    else:
        generate_paper_style_augmentation()
