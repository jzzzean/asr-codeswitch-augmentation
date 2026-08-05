# -*- coding: utf-8 -*-


import os
import random
import traceback
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path

# 配置參數
TARGET_SR = 16000
MIN_SEC = 3
MAX_SEC = 5
NUM_SAMPLES_TO_GENERATE = 200
NUM_ZH_SAMPLES = 1000
NUM_EN_SAMPLES = 500
AUG_DATA_BASE_PATH = "augmented_data"

def concatenate_utterances(zh_path, en_path, output_path, zh_text, en_text):
    """
    拼接中英文音檔，隨機決定順序並確保文字稿對齊
    
    Args:
        zh_path: 中文音檔路徑
        en_path: 英文音檔路徑
        output_path: 輸出音檔路徑
        zh_text: 中文文字稿
        en_text: 英文文字稿
    
    Returns:
        tuple: (transcript, success) - 拼接後的文字稿和成功標誌
    """
    try:
        # 讀取音檔並重取樣
        y_zh, sr_zh = librosa.load(zh_path, sr=TARGET_SR)
        y_en, sr_en = librosa.load(en_path, sr=TARGET_SR)
        
        # 確認音檔長度（轉換為樣本數）
        min_samples = int(MIN_SEC * TARGET_SR)
        max_samples = int(MAX_SEC * TARGET_SR)
        
        # 檢查音檔是否太短
        if len(y_zh) < min_samples or len(y_en) < min_samples:
            print(f"跳過過短音檔: ZH={len(y_zh)/TARGET_SR:.2f}s, EN={len(y_en)/TARGET_SR:.2f}s")
            return None, False
        
        # 隨機截取片段長度（3-5秒之間）
        zh_len = random.randint(min_samples, min(max_samples, len(y_zh)))
        en_len = random.randint(min_samples, min(max_samples, len(y_en)))
        
        # 隨機選擇起始位置進行截取
        zh_start = random.randint(0, max(0, len(y_zh) - zh_len))
        en_start = random.randint(0, max(0, len(y_en) - en_len))
        
        y_zh_segment = y_zh[zh_start:zh_start + zh_len]
        y_en_segment = y_en[en_start:en_start + en_len]
        
        # 隨機決定拼接順序
        if random.random() < 0.5:
            # ZH + EN
            y_concat = np.concatenate([y_zh_segment, y_en_segment])
            transcript = zh_text.strip() + " " + en_text.strip()
        else:
            # EN + ZH
            y_concat = np.concatenate([y_en_segment, y_zh_segment])
            transcript = en_text.strip() + " " + zh_text.strip()
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 儲存音檔
        sf.write(output_path, y_concat, TARGET_SR)
        
        return transcript, True
        
    except Exception as e:
        print(f"拼接失敗: {zh_path} + {en_path}")
        print(f"錯誤: {e}")
        return None, False


def add_background_noise(clip_path, noise_path, output_path, snr_db=10, text=""):
    """
    在音檔上疊加背景噪音
    
    Args:
        clip_path: 原始音檔路徑
        noise_path: 噪音檔路徑
        output_path: 輸出音檔路徑
        snr_db: 信噪比 (dB)
        text: 對應的文字稿
    
    Returns:
        tuple: (transcript, success) - 文字稿和成功標誌
    """
    try:
        speech, _ = librosa.load(clip_path, sr=TARGET_SR)
        noise, _ = librosa.load(noise_path, sr=TARGET_SR)
        
        # 如果噪音檔比語音檔短，重複噪音
        if len(noise) < len(speech):
            repeats = int(np.ceil(len(speech) / len(noise)))
            noise = np.tile(noise, repeats)
        
        # 截取與語音相同長度的噪音
        noise = noise[:len(speech)]
        
        # 計算RMS
        rms_speech = np.sqrt(np.mean(speech**2))
        rms_noise = np.sqrt(np.mean(noise**2))
        
        if rms_noise == 0:
            print(f"警告: 噪音檔案無聲: {noise_path}")
            return None, False
            
        # 根據SNR調整噪音強度
        target_rms_noise = rms_speech / (10**(snr_db / 20))
        scaling_factor = target_rms_noise / rms_noise
        
        scaled_noise = noise * scaling_factor
        mixed_audio = speech + scaled_noise
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 儲存音檔
        sf.write(output_path, mixed_audio, TARGET_SR)
        
        return text.strip(), True
        
    except Exception as e:
        print(f"加噪失敗: {clip_path}")
        print(f"錯誤: {e}")
        return None, False


def apply_perturbation(clip_path, output_path, speed_factor=1.0, pitch_shift_semitones=0, text=""):
    """
    對音檔進行速度或音高調整
    
    Args:
        clip_path: 原始音檔路徑
        output_path: 輸出音檔路徑
        speed_factor: 速度調整因子
        pitch_shift_semitones: 音高調整(半音)
        text: 對應的文字稿
    
    Returns:
        tuple: (transcript, success) - 文字稿和成功標誌
    """
    try:
        y, sr = librosa.load(clip_path, sr=TARGET_SR)
        
        y_perturbed = y
        
        # 速度調整
        if speed_factor != 1.0:
            y_perturbed = librosa.effects.time_stretch(y_perturbed, rate=speed_factor)
            
        # 音高調整
        if pitch_shift_semitones != 0:
            y_perturbed = librosa.effects.pitch_shift(y_perturbed, sr=sr, n_steps=pitch_shift_semitones)
        
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 儲存音檔
        sf.write(output_path, y_perturbed, sr)
        
        return text.strip(), True
        
    except Exception as e:
        print(f"擾動失敗: {clip_path}")
        print(f"錯誤: {e}")
        return None, False


def load_dataset_samples(zh_data_path, en_data_path, zh_metadata_file, en_metadata_file):
    """
    載入中英文資料集樣本
    
    Args:
        zh_data_path: 中文音檔資料夾路徑
        en_data_path: 英文音檔資料夾路徑
        zh_metadata_file: 中文metadata CSV檔路徑
        en_metadata_file: 英文metadata CSV檔路徑
    
    Returns:
        tuple: (zh_samples, en_samples) - 中英文樣本列表
    """
    try:
        # 讀取metadata
        zh_df = pd.read_csv(zh_metadata_file, encoding='utf-8')
        en_df = pd.read_csv(en_metadata_file, encoding='utf-8')
        
        # 隨機抽取樣本
        zh_samples = zh_df.sample(n=min(NUM_ZH_SAMPLES, len(zh_df))).to_dict('records')
        en_samples = en_df.sample(n=min(NUM_EN_SAMPLES, len(en_df))).to_dict('records')
        
        # 補充完整路徑
        for sample in zh_samples:
            sample['full_path'] = os.path.join(zh_data_path, sample['file_path'])
        
        for sample in en_samples:
            sample['full_path'] = os.path.join(en_data_path, sample['file_path'])
        
        print(f"載入 {len(zh_samples)} 個中文樣本, {len(en_samples)} 個英文樣本")
        return zh_samples, en_samples
        
    except Exception as e:
        print(f"載入資料集失敗: {e}")
        return [], []


def generate_augmented_data(zh_samples, en_samples, noise_files=None):
    """
    生成增強資料
    
    Args:
        zh_samples: 中文樣本列表
        en_samples: 英文樣本列表
        noise_files: 噪音檔案列表（可選）
    
    Returns:
        list: 增強資料的metadata記錄
    """
    augmented_records = []
    
    # 建立輸出資料夾
    concat_dir = os.path.join(AUG_DATA_BASE_PATH, "test_concat")
    noisy_dir = os.path.join(AUG_DATA_BASE_PATH, "test_noisy")
    perturbed_dir = os.path.join(AUG_DATA_BASE_PATH, "test_perturbed")
    
    for dir_path in [concat_dir, noisy_dir, perturbed_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    generated_count = 0
    attempts = 0
    max_attempts = NUM_SAMPLES_TO_GENERATE * 3  # 防止無限迴圈
    
    print(f"開始生成 {NUM_SAMPLES_TO_GENERATE} 個增強樣本...")
    
    while generated_count < NUM_SAMPLES_TO_GENERATE and attempts < max_attempts:
        attempts += 1
        
        # 隨機選擇中英文樣本
        zh_sample = random.choice(zh_samples)
        en_sample = random.choice(en_samples)
        
        # 1. 生成拼接樣本
        concat_filename = f"concat_{generated_count:04d}.wav"
        concat_path = os.path.join(concat_dir, concat_filename)
        
        transcript, success = concatenate_utterances(
            zh_sample['full_path'], 
            en_sample['full_path'],
            concat_path,
            zh_sample['transcript'],
            en_sample['transcript']
        )
        
        if success:
            augmented_records.append({
                'file_path': os.path.join("test_concat", concat_filename),
                'transcript': transcript,
                'augmentation_type': 'concat',
                'duration': librosa.get_duration(filename=concat_path),
                'sample_rate': TARGET_SR
            })
            
            # 2. 基於拼接樣本生成加噪樣本
            if noise_files:
                noise_file = random.choice(noise_files)
                noisy_filename = f"noisy_{generated_count:04d}.wav"
                noisy_path = os.path.join(noisy_dir, noisy_filename)
                
                noisy_transcript, noisy_success = add_background_noise(
                    concat_path, noise_file, noisy_path, 
                    snr_db=random.uniform(5, 15), text=transcript
                )
                
                if noisy_success:
                    augmented_records.append({
                        'file_path': os.path.join("test_noisy", noisy_filename),
                        'transcript': noisy_transcript,
                        'augmentation_type': 'noisy',
                        'duration': librosa.get_duration(filename=noisy_path),
                        'sample_rate': TARGET_SR
                    })
            
            # 3. 基於拼接樣本生成擾動樣本
            perturbed_filename = f"perturbed_{generated_count:04d}.wav"
            perturbed_path = os.path.join(perturbed_dir, perturbed_filename)
            
            # 隨機選擇擾動參數
            speed_factor = random.uniform(0.9, 1.1)
            pitch_shift = random.randint(-2, 2)
            
            perturbed_transcript, perturbed_success = apply_perturbation(
                concat_path, perturbed_path,
                speed_factor=speed_factor, 
                pitch_shift_semitones=pitch_shift,
                text=transcript
            )
            
            if perturbed_success:
                augmented_records.append({
                    'file_path': os.path.join("test_perturbed", perturbed_filename),
                    'transcript': perturbed_transcript,
                    'augmentation_type': 'perturbed',
                    'duration': librosa.get_duration(filename=perturbed_path),
                    'sample_rate': TARGET_SR
                })
            
            generated_count += 1
            if generated_count % 50 == 0:
                print(f"已生成 {generated_count}/{NUM_SAMPLES_TO_GENERATE} 個樣本")
    
    print(f"增強資料生成完成! 共生成 {len(augmented_records)} 個增強樣本")
    return augmented_records


def save_metadata(records, output_path):
    """
    儲存metadata到CSV檔案
    
    Args:
        records: 資料記錄列表
        output_path: 輸出CSV檔案路徑
    """
    try:
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Metadata已儲存至: {output_path}")
    except Exception as e:
        print(f"儲存metadata失敗: {e}")


# --- 主程式執行區塊 ---
if __name__ == '__main__':
    print("=== 音檔資料增強腳本 ===")
    
    # 配置路徑（請根據實際情況修改）
    ZH_DATA_PATH = "data/mandarin"  # 中文音檔資料夾
    EN_DATA_PATH = "data/english"   # 英文音檔資料夾
    ZH_METADATA_FILE = "metadata/zh_metadata.csv"  # 中文metadata
    EN_METADATA_FILE = "metadata/en_metadata.csv"  # 英文metadata
    NOISE_DIR = "noise"  # 噪音檔案資料夾（可選）
    
    # 檢查必要檔案是否存在
    required_paths = [ZH_DATA_PATH, EN_DATA_PATH, ZH_METADATA_FILE, EN_METADATA_FILE]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    
    if missing_paths:
        print("缺少以下必要檔案/資料夾:")
        for path in missing_paths:
            print(f"  - {path}")
        print("\n執行示範模式...")
        
        # 示範模式：建立假資料
        print("建立示範資料...")
        demo_dirs = [
            "demo_data/mandarin",
            "demo_data/english", 
            "demo_data/noise",
            AUG_DATA_BASE_PATH
        ]
        for d in demo_dirs:
            os.makedirs(d, exist_ok=True)
        
        # 建立假音檔和metadata
        sr_demo = TARGET_SR
        duration = 4  # 秒
        t = np.linspace(0., duration, int(sr_demo * duration), endpoint=False)
        
        # 建立示範音檔
        for i in range(5):
            # 中文音檔
            freq = 440 + i * 50
            y = 0.5 * np.sin(2. * np.pi * freq * t)
            zh_path = f"demo_data/mandarin/zh_{i:03d}.wav"
            sf.write(zh_path, y, sr_demo)
            
            # 英文音檔
            freq = 550 + i * 60
            y = 0.5 * np.sin(2. * np.pi * freq * t)
            en_path = f"demo_data/english/en_{i:03d}.wav"
            sf.write(en_path, y, sr_demo)
        
        # 建立噪音檔
        noise_y = 0.1 * np.random.normal(0, 1, len(t))
        sf.write("demo_data/noise/background.wav", noise_y, sr_demo)
        
        # 建立metadata檔案
        zh_metadata = pd.DataFrame({
            'file_path': [f'zh_{i:03d}.wav' for i in range(5)],
            'transcript': [f'這是中文樣本{i}' for i in range(5)],
            'duration': [duration] * 5,
            'sample_rate': [TARGET_SR] * 5
        })
        zh_metadata.to_csv("demo_data/zh_metadata.csv", index=False, encoding='utf-8')
        
        en_metadata = pd.DataFrame({
            'file_path': [f'en_{i:03d}.wav' for i in range(5)],
            'transcript': [f'This is English sample {i}' for i in range(5)],
            'duration': [duration] * 5,
            'sample_rate': [TARGET_SR] * 5
        })
        en_metadata.to_csv("demo_data/en_metadata.csv", index=False, encoding='utf-8')
        
        # 使用示範資料
        ZH_DATA_PATH = "demo_data/mandarin"
        EN_DATA_PATH = "demo_data/english"
        ZH_METADATA_FILE = "demo_data/zh_metadata.csv"
        EN_METADATA_FILE = "demo_data/en_metadata.csv"
        NOISE_DIR = "demo_data/noise"
    
    # 載入資料集
    zh_samples, en_samples = load_dataset_samples(
        ZH_DATA_PATH, EN_DATA_PATH, ZH_METADATA_FILE, EN_METADATA_FILE
    )
    
    if not zh_samples or not en_samples:
        print("無法載入資料集，請檢查路徑和檔案格式")
        exit(1)
    
    # 載入噪音檔案（如果存在）
    noise_files = []
    if os.path.exists(NOISE_DIR):
        for ext in ['.wav', '.mp3', '.flac']:
            noise_files.extend(Path(NOISE_DIR).glob(f'*{ext}'))
        noise_files = [str(f) for f in noise_files]
        print(f"找到 {len(noise_files)} 個噪音檔案")
    
    # 生成增強資料
    augmented_records = generate_augmented_data(zh_samples, en_samples, noise_files)
    
    # 儲存metadata
    metadata_path = os.path.join(AUG_DATA_BASE_PATH, "metadata.csv")
    save_metadata(augmented_records, metadata_path)
    
    print(f"\n=== 完成 ===")
    print(f"增強音檔儲存位置: {AUG_DATA_BASE_PATH}")
    print(f"Metadata檔案: {metadata_path}")
    print(f"總共生成: {len(augmented_records)} 個增強樣本")