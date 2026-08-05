# -*- coding: utf-8 -*-

import os
import glob
import io
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

# To ensure proper display of plot elements like minus signs
plt.rcParams['axes.unicode_minus'] = False

# Try to import the datasets library
try:
    from datasets import load_dataset, Audio
except ImportError:
    print("Error: 'datasets' library is missing. Please run: pip install datasets")
    load_dataset = None

def calculate_advanced_metrics(y, sr, top_db=30):
    """
    Calculates the silence ratio and an estimated Signal-to-Noise Ratio (SNR).
    """
    # 1. Calculate silence ratio
    non_silent_intervals = librosa.effects.split(y, top_db=top_db)
    non_silent_duration = sum(librosa.samples_to_time(interval[1] - interval[0], sr=sr) for interval in non_silent_intervals)
    total_duration = librosa.get_duration(y=y, sr=sr)
    
    if total_duration == 0:
        return {'silence_ratio': 0, 'snr_estimate': np.nan}
        
    silence_ratio = (1 - (non_silent_duration / total_duration)) * 100

    # 2. Estimate SNR
    signal_energy = 0
    noise_mask = np.ones_like(y, dtype=bool)
    
    for start, end in non_silent_intervals:
        noise_mask[start:end] = False
        signal_energy += np.sum(y[start:end]**2)

    noise_energy = np.sum(y[noise_mask]**2)
    
    num_signal_samples = len(y) - np.sum(noise_mask)
    num_noise_samples = np.sum(noise_mask)
    
    rms_signal = np.sqrt(signal_energy / num_signal_samples) if num_signal_samples > 0 else 0
    rms_noise = np.sqrt(noise_energy / num_noise_samples) if num_noise_samples > 0 else 0
    
    if rms_noise > 1e-9:
        snr_estimate = 20 * np.log10(rms_signal / rms_noise)
    else:
        snr_estimate = np.inf

    return {'silence_ratio': silence_ratio, 'snr_estimate': snr_estimate}


def analyze_from_metadata(metadata_csv_path, augmentation_type):
    """
    分析基於 metadata 的音檔，而不是直接掃資料夾。
    只分析有成功生成的音檔。
    """
    df_meta = pd.read_csv(metadata_csv_path)
    
    # 篩選指定增強類型
    df_meta_type = df_meta[df_meta['augmentation_type'] == augmentation_type]
    
    print(f"\nAnalyzing {augmentation_type} - {len(df_meta_type)} audio files from metadata")
    
    metrics_list = []
    
    for _, row in tqdm(df_meta_type.iterrows(), total=len(df_meta_type)):
        file_path = os.path.join(os.path.dirname(metadata_csv_path), row['file_path'])
        try:
            y, sr = librosa.load(file_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.mean(librosa.feature.rms(y=y))
            adv_metrics = calculate_advanced_metrics(y, sr)
            metrics_list.append({
                'duration': duration,
                'rms': rms,
                'silence_ratio': adv_metrics['silence_ratio'],
                'snr_estimate': adv_metrics['snr_estimate'],
                'file_path': file_path
            })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    df = pd.DataFrame(metrics_list)
    print(f"Final dataset size for {augmentation_type}: {len(df)} files")
    
    results = {
        "Dataset": f"Augmented ({augmentation_type})",
        "Num Audios": len(df),
        "Total Duration (h)": f"{df['duration'].sum() / 3600:.2f}",
        "Avg Duration (s)": f"{df['duration'].mean():.2f}",
        "Min Duration (s)": f"{df['duration'].min():.2f}",
        "Max Duration (s)": f"{df['duration'].max():.2f}",
        "Avg Volume (RMS)": f"{df['rms'].mean():.4f}",
        "Silence Ratio (%)": f"{df['silence_ratio'].mean():.2f}",
        "Avg SNR (dB)": f"{df['snr_estimate'].mean():.2f}"
    }
    
    return results, df['duration'].tolist()

    
    print(f"Files matching pattern '{pattern}': {len(base_files)}")
    
    if not base_files:
        print(f"Warning: No files matching pattern in {folder_path}")
        print("Available files (first 10):")
        for example in audio_files[:10]:
            print("  ", os.path.basename(example))
        return None, None

    # ✅ Debug：列印前 10 個篩選後的檔案名稱
    print(f"Processing {len(base_files)} files from {dataset_name}. Examples:")
    for example in base_files[:10]:
        print("  ", os.path.basename(example))

    metrics_list = []
    
    for file_path in tqdm(base_files, desc=f"Processing {dataset_name}"):
        try:
            y, sr = librosa.load(file_path, sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.mean(librosa.feature.rms(y=y))
            adv_metrics = calculate_advanced_metrics(y, sr)
            
            metrics_list.append({
                'duration': duration,
                'rms': rms,
                'silence_ratio': adv_metrics['silence_ratio'],
                'snr_estimate': adv_metrics['snr_estimate'],
                'file_path': file_path
            })
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            continue

    print(f"Total processed files: {len(metrics_list)}")
    
    if not metrics_list:
        return None, None

    # 完全不做任何資料過濾，只轉換成DataFrame
    df = pd.DataFrame(metrics_list)
    
    print(f"Final dataset size: {len(df)} files")

    
    results = {
        "Dataset": dataset_name,
        "Num Audios": len(df),
        "Total Duration (h)": f"{df['duration'].sum() / 3600:.2f}",
        "Avg Duration (s)": f"{df['duration'].mean():.2f}",
        "Min Duration (s)": f"{df['duration'].min():.2f}",
        "Max Duration (s)": f"{df['duration'].max():.2f}",
        "Avg Speech Rate (char/s)": "N/A",
        "Avg Volume (RMS)": f"{df['rms'].mean():.4f}",
        "Silence Ratio (%)": f"{df['silence_ratio'].mean():.2f}",
        "Avg SNR (dB)": f"{df['snr_estimate'].mean():.2f}"
    }
    
    return results, df['duration'].tolist()


def analyze_hf_dataset(dataset_id, name, split, dataset_name):
    """Analyzes a dataset from Hugging Face, manually decoding with librosa."""
    if not load_dataset:
        return None, None
        
    print(f"\n--- Analyzing Hugging Face Dataset: {dataset_name} ---")
    
    try:
        if name:
            dataset = load_dataset(dataset_id, name, split=split, trust_remote_code=True).cast_column("audio", Audio(decode=False))
        else:
            dataset = load_dataset(dataset_id, split=split, trust_remote_code=True).cast_column("audio", Audio(decode=False))
    except Exception as e:
        print(f"Error: Could not load {dataset_id}. Message: {e}")
        return None, None

    metrics_list = []
    failed_count = 0
    
    for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
        try:
            audio_bytes = item['audio']['bytes']
            if not audio_bytes:
                failed_count += 1
                continue
            
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
            duration = librosa.get_duration(y=y, sr=sr)
            transcription = item.get('transcription') or item.get('sentence') or item.get('text') or ""
            
            if duration > 0:
                rms = np.mean(librosa.feature.rms(y=y))
                adv_metrics = calculate_advanced_metrics(y, sr)
                metrics_list.append({
                    'duration': duration,
                    'speech_rate': len(transcription) / duration,
                    'rms': rms,
                    'silence_ratio': adv_metrics['silence_ratio'],
                    'snr_estimate': adv_metrics['snr_estimate']
                })
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
            continue
    
    print(f"Successfully processed: {len(metrics_list)} samples")
    print(f"Failed to process: {failed_count} samples")
    
    if not metrics_list:
        return None, None

    df = pd.DataFrame(metrics_list).replace([np.inf, -np.inf], np.nan).dropna()
    
    results = {
        "Dataset": dataset_name,
        "Num Audios": len(df),
        "Total Duration (h)": f"{df['duration'].sum() / 3600:.2f}",
        "Avg Duration (s)": f"{df['duration'].mean():.2f}",
        "Min Duration (s)": f"{df['duration'].min():.2f}",
        "Max Duration (s)": f"{df['duration'].max():.2f}",
        "Avg Speech Rate (char/s)": f"{df['speech_rate'].mean():.2f}",
        "Avg Volume (RMS)": f"{df['rms'].mean():.4f}",
        "Silence Ratio (%)": f"{df['silence_ratio'].mean():.2f}",
        "Avg SNR (dB)": f"{df['snr_estimate'].mean():.2f}"
    }
    return results, df['duration'].tolist()


def plot_durations(all_durations, output_filename="duration_distribution_optimized.png"):
    """Plots the duration distribution for all datasets (optimized version)."""
    plt.figure(figsize=(14, 8))
    
    all_flat_durations = [d for durations in all_durations.values() for d in durations]
    if not all_flat_durations:
        print("No valid duration data to plot.")
        return
        
    xlim_upper = min(np.percentile(all_flat_durations, 99.5), 35)

    colors = plt.cm.Set3(np.linspace(0, 1, len(all_durations)))
    
    for i, (name, durations) in enumerate(all_durations.items()):
        if durations:
            filtered_durations = [d for d in durations if d <= xlim_upper]
            if filtered_durations:
                sns.kdeplot(filtered_durations, label=f"{name} (n={len(durations)})", 
                           fill=True, alpha=0.6, cut=0, color=colors[i])
            
    plt.title("Audio Duration Distribution Across Datasets", fontsize=16, fontweight='bold')
    plt.xlabel("Duration (seconds)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.xlim(0, xlim_upper)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved optimized duration distribution plot to: {output_filename}")


def check_folder_contents(folder_path, folder_name):
    """檢查資料夾內容的詳細資訊"""
    print(f"\n=== 檢查 {folder_name} 資料夾內容 ===")
    print(f"路徑: {folder_path}")
    
    if not os.path.exists(folder_path):
        print("❌ 資料夾不存在！")
        return
    
    # 取得所有檔案
    all_files = os.listdir(folder_path)
    wav_files = [f for f in all_files if f.endswith('.wav')]
    txt_files = [f for f in all_files if f.endswith('.txt')]
    
    print(f"總檔案數: {len(all_files)}")
    print(f".wav 檔案: {len(wav_files)}")
    print(f".txt 檔案: {len(txt_files)}")
    
    if wav_files:
        print(f"\n前10個 .wav 檔案:")
        for f in wav_files[:10]:
            file_path = os.path.join(folder_path, f)
            try:
                size = os.path.getsize(file_path)
                duration = librosa.get_duration(filename=file_path)
                print(f"  {f} - {size} bytes - {duration:.2f}s")
            except Exception as e:
                print(f"  {f} - 檔案讀取錯誤: {e}")


if __name__ == '__main__':
    # --- Define paths for your augmented data ---
    data_dir = Path(os.environ.get("ASR_DATA_DIR", "data"))
    AUG_DATA_BASE_PATH = str(data_dir / "augmented_data")
    CONCAT_PATH = os.path.join(AUG_DATA_BASE_PATH, "test_concat")
    NOISY_PATH = os.path.join(AUG_DATA_BASE_PATH, "test_noisy")
    PERTURBED_PATH = os.path.join(AUG_DATA_BASE_PATH, "test_perturbed")

    # 先檢查資料夾內容
    check_folder_contents(CONCAT_PATH, "Concat")
    check_folder_contents(NOISY_PATH, "Noisy")
    check_folder_contents(PERTURBED_PATH, "Perturbed")

    # --- Execute Analysis ---
    all_results = []
    all_durations_data = {}

    # 1. Analyze Hugging Face Datasets
    print("\n" + "="*60)
    print("開始分析 Hugging Face 資料集")
    print("="*60)
    
    ascend_results, ascend_durations = analyze_hf_dataset("CAiRE/ASCEND", None, "test", "ASCEND (Conversational)")
    if ascend_results:
        all_results.append(ascend_results)
        all_durations_data["ASCEND (Conversational)"] = ascend_durations

    ml_lecture_results, ml_lecture_durations = analyze_hf_dataset("ky552/ML2021_ASR_ST", None, "test", "ML-Lecture (Lecture-style)")
    if ml_lecture_results:
        all_results.append(ml_lecture_results)
        all_durations_data["ML-Lecture (Lecture-style)"] = ml_lecture_durations
    
    # 2. Analyze Augmented Data
    print("\n" + "="*60)
    print("開始分析增強資料")
    print("="*60)
    
    metadata_csv_path = os.path.join(AUG_DATA_BASE_PATH, "metadata.csv")

    # Analyze concatenated data
    concat_results, concat_durations = analyze_from_metadata(metadata_csv_path, "concat")
    if concat_results:
        all_results.append(concat_results)
        all_durations_data["Augmented (Concat)"] = concat_durations

    # Analyze noisy data
    noisy_results, noisy_durations = analyze_from_metadata(metadata_csv_path, "noisy")
    if noisy_results:
        all_results.append(noisy_results)
        all_durations_data["Augmented (Noisy)"] = noisy_durations

    # Analyze perturbed data
    perturbed_results, perturbed_durations = analyze_from_metadata(metadata_csv_path, "perturbed")
    if perturbed_results:
        all_results.append(perturbed_results)
        all_durations_data["Augmented (Perturbed)"] = perturbed_durations

    # --- Display and Export Results ---
    if all_results:
        df = pd.DataFrame(all_results).set_index('Dataset')
        print("\n\n" + "="*140)
        print(" " * 50 + "Final Results of Dataset Analysis")
        print("="*140)
        print(df.to_string())
        print("="*140)
        
        # Export to CSV (utf-8-sig for Excel compatibility)
        output_csv_path = "dataset_analysis_results_detailed.csv"
        df.to_csv(output_csv_path, encoding='utf-8-sig')
        print(f"\nAnalysis results have been exported to: {output_csv_path}")

        if all_durations_data:
            plot_durations(all_durations_data, "duration_distribution_detailed.png")
    else:
        print("\n❌ 沒有成功分析任何資料集")
