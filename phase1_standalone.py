#!/usr/bin/env python3
"""
WeNet ASR Phase 1 - Standalone Script
Pre-trained Model Setup & Baseline Validation

This script implements Phase 1 of the WeNet ASR Transfer Learning project.
Run this in Google Colab with GPU enabled.

Usage:
    1. Upload to Colab
    2. Run with GPU runtime
    3. Results saved to /content/wenet_asr_project/results/
"""

import os
import sys
import json
import yaml
import time
import urllib.request
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

QUICK_TEST = True  # Set to False for full evaluation (295 speakers)

# Paths
BASE_DIR = Path("/content/wenet_asr_project")
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Model URL
MODEL_URL = "https://wenet-1256283475.cos.ap-shanghai.myqcloud.com/models/aishell/20210601_u2pp_conformer_exp.tar.gz"

# AISHELL-1 URL
AISHELL_URL = "https://www.openslr.org/resources/33/data_aishell.tgz"

# Test speakers
if QUICK_TEST:
    TEST_SPEAKERS = [f"S{i:04d}" for i in range(764, 769)]  # 5 speakers
else:
    TEST_SPEAKERS = [f"S{i:04d}" for i in range(764, 1059)]  # 295 speakers

# =============================================================================
# Utility Functions
# =============================================================================

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def setup_directories():
    """Create project directory structure"""
    for dir_path in [MODELS_DIR, DATA_DIR, RESULTS_DIR, LOGS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")


def install_wenet():
    """Install WeNet and dependencies"""
    print("Installing WeNet and dependencies...")
    os.system("pip install -q torchaudio transformers torchmetrics jiwer")
    
    # Clone WeNet
    os.chdir("/content")
    if not Path("/content/wenet").exists():
        os.system("git clone https://github.com/wenet-e2e/wenet.git")
    
    os.chdir("/content/wenet")
    os.system("pip install -q -e .")
    
    sys.path.insert(0, '/content/wenet')
    print("✅ WeNet installation complete!")


# =============================================================================
# CER Calculation
# =============================================================================

def levenshtein_distance(ref: str, hyp: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    m, n = len(ref), len(hyp)
    dp = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i, j] = dp[i-1, j-1]
            else:
                dp[i, j] = min(
                    dp[i-1, j] + 1,      # deletion
                    dp[i, j-1] + 1,      # insertion
                    dp[i-1, j-1] + 1     # substitution
                )
    
    return dp[m, n]


def calculate_cer(ref: str, hyp: str) -> float:
    """Calculate Character Error Rate"""
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    distance = levenshtein_distance(ref, hyp)
    return distance / len(ref)


# =============================================================================
# Feature Extraction & Decoding
# =============================================================================

def extract_fbank(audio_path: str, num_mel_bins: int = 80, 
                  frame_length: int = 25, frame_shift: int = 10, 
                  sample_rate: int = 16000) -> torch.Tensor:
    """Extract filterbank features from audio"""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Extract fbank features
    features = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate
    )
    
    return features


def decode_ctc(indices: np.ndarray, vocab: Dict[int, str]) -> str:
    """Decode CTC output indices to text"""
    result = []
    prev = -1
    for idx in indices:
        if idx != 0 and idx != prev:  # 0 is blank
            result.append(vocab.get(idx, ''))
        prev = idx
    return ''.join(result)


def greedy_decode(model, features: torch.Tensor, device: torch.device) -> np.ndarray:
    """Greedy CTC decoding"""
    with torch.no_grad():
        features = features.unsqueeze(0).to(device)
        features_lengths = torch.tensor([features.shape[1]], dtype=torch.long).to(device)
        
        encoder_out, encoder_mask = model.encoder(features, features_lengths)
        ctc_probs = model.ctc.log_softmax(encoder_out)
        
        top_probs, top_indices = ctc_probs.max(dim=-1)
        indices = top_indices[0].cpu().numpy()
    
    return indices


# =============================================================================
# Data Loading
# =============================================================================

def download_model():
    """Download pre-trained WeNet model"""
    model_tar = MODELS_DIR / "20210601_u2pp_conformer_exp.tar.gz"
    MODEL_PATH = MODELS_DIR / "pretrained_conformer"
    
    if not model_tar.exists():
        print("📥 Downloading pre-trained Conformer-CTC model...")
        download_with_progress(MODEL_URL, model_tar)
        print(f"✅ Model downloaded: {model_tar.stat().st_size / 1e6:.2f} MB")
    
    # Extract
    if not MODEL_PATH.exists():
        print("📦 Extracting model archive...")
        with tarfile.open(model_tar, 'r:gz') as tar:
            tar.extractall(MODELS_DIR)
        
        extracted_dir = MODELS_DIR / "20210601_u2pp_conformer_exp"
        if extracted_dir.exists():
            extracted_dir.rename(MODEL_PATH)
        
        print(f"✅ Model extracted to: {MODEL_PATH}")
    
    return MODEL_PATH


def download_aishell_data():
    """Download and extract AISHELL-1 test data"""
    aishell_tar = DATA_DIR / "data_aishell.tgz"
    AISHELL_DIR = DATA_DIR / "aishell1"
    WAV_DIR = AISHELL_DIR / "wav"
    TRANSCRIPT_DIR = AISHELL_DIR / "transcript"
    
    if not aishell_tar.exists():
        print("📥 Downloading AISHELL-1 dataset...")
        print("   This may take 10-15 minutes (15 GB)...")
        download_with_progress(AISHELL_URL, aishell_tar)
    
    print("📦 Extracting test data...")
    with tarfile.open(aishell_tar, 'r:gz') as tar:
        # Extract transcript
        transcript_members = [m for m in tar.getmembers() if 'transcript' in m.name]
        for member in transcript_members:
            tar.extract(member, AISHELL_DIR)
        
        transcript_src = AISHELL_DIR / "data_aishell/transcript"
        if transcript_src.exists() and not TRANSCRIPT_DIR.exists():
            shutil.move(str(transcript_src), str(TRANSCRIPT_DIR))
        
        # Extract test speakers
        for speaker in tqdm(TEST_SPEAKERS, desc="Extracting speakers"):
            speaker_tar = f"data_aishell/wav/{speaker}.tar.gz"
            try:
                member = tar.getmember(speaker_tar)
                tar.extract(member, AISHELL_DIR)
                
                inner_tar_path = AISHELL_DIR / speaker_tar
                if inner_tar_path.exists():
                    with tarfile.open(inner_tar_path, 'r:gz') as inner_tar:
                        inner_tar.extractall(WAV_DIR)
                    inner_tar_path.unlink()
            except KeyError:
                print(f"   ⚠️ Speaker {speaker} not found")
    
    # Cleanup
    data_aishell_dir = AISHELL_DIR / "data_aishell"
    if data_aishell_dir.exists():
        shutil.rmtree(data_aishell_dir)
    
    return AISHELL_DIR


def load_test_data(aishell_dir: Path) -> List[Dict]:
    """Load test audio files and transcripts"""
    transcript_file = aishell_dir / "transcript" / "aishell_transcript_v0.8.txt"
    WAV_DIR = aishell_dir / "wav"
    
    # Load transcripts
    transcripts = {}
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                audio_id = parts[0]
                text = ''.join(parts[1:])
                transcripts[audio_id] = text
    
    # Find test audio files
    test_audio_files = []
    for speaker in TEST_SPEAKERS:
        speaker_dir = WAV_DIR / speaker
        if speaker_dir.exists():
            for wav_file in speaker_dir.glob("*.wav"):
                audio_id = wav_file.stem
                if audio_id in transcripts:
                    test_audio_files.append({
                        'audio_path': str(wav_file),
                        'audio_id': audio_id,
                        'speaker': speaker,
                        'text': transcripts[audio_id]
                    })
    
    return test_audio_files


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    """Main Phase 1 pipeline"""
    print("=" * 70)
    print("          WENET ASR - PHASE 1: BASELINE VALIDATION")
    print("=" * 70)
    
    # Setup
    print("\n[1/7] Setting up directories...")
    setup_directories()
    
    # Install WeNet
    print("\n[2/7] Installing WeNet...")
    install_wenet()
    
    # Import WeNet modules
    from wenet.utils.init_model import init_model
    from wenet.utils.checkpoint import load_checkpoint
    
    # Download model
    print("\n[3/7] Downloading pre-trained model...")
    model_path = download_model()
    
    # Load config and vocab
    config_path = model_path / "train.yaml"
    vocab_path = model_path / "words.txt"
    checkpoint_path = model_path / "final.pt"
    
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                char, idx = parts
                vocab[int(idx)] = char
    
    # Count parameters
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    
    print(f"   Model parameters: {total_params:,}")
    print(f"   Vocabulary size: {len(vocab)}")
    
    # Download data
    print("\n[4/7] Downloading test data...")
    aishell_dir = download_aishell_data()
    
    # Load test data
    print("\n[5/7] Loading test data...")
    test_audio_files = load_test_data(aishell_dir)
    print(f"   Loaded {len(test_audio_files)} test samples")
    
    # Initialize model
    print("\n[6/7] Initializing model...")
    model = init_model(configs['model'], configs.get('tokenizer', None))
    load_checkpoint(model, str(checkpoint_path))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"   Model loaded on {device}")
    
    # Run inference
    print("\n[7/7] Running baseline inference...")
    results = []
    total_inference_time = 0
    total_audio_duration = 0
    
    for entry in tqdm(test_audio_files, desc="Inferencing"):
        # Load audio
        waveform, sr = torchaudio.load(entry['audio_path'])
        audio_duration = waveform.shape[1] / sr
        total_audio_duration += audio_duration
        
        # Extract features and infer
        features = extract_fbank(entry['audio_path'])
        
        start_time = time.time()
        indices = greedy_decode(model, features, device)
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Decode
        pred_text = decode_ctc(indices, vocab)
        
        results.append({
            'audio_id': entry['audio_id'],
            'speaker': entry['speaker'],
            'reference': entry['text'],
            'prediction': pred_text,
            'audio_duration': audio_duration,
            'inference_time': inference_time
        })
    
    # Calculate CER
    print("\nCalculating CER...")
    total_errors = 0
    total_ref_chars = 0
    
    for result in results:
        cer = calculate_cer(result['reference'], result['prediction'])
        errors = levenshtein_distance(result['reference'], result['prediction'])
        
        result['cer'] = cer
        result['errors'] = errors
        result['ref_length'] = len(result['reference'])
        
        total_errors += errors
        total_ref_chars += len(result['reference'])
    
    overall_cer = total_errors / total_ref_chars if total_ref_chars > 0 else 0
    rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print("                    BASELINE RESULTS (No LM)")
    print("=" * 70)
    print(f"\n🎯 OVERALL CER: {overall_cer * 100:.2f}%")
    print(f"   Total samples: {len(results)}")
    print(f"   Total reference characters: {total_ref_chars}")
    print(f"   Total errors: {total_errors}")
    print(f"\n⚡ Real-time factor (RTF): {rtf:.4f}")
    print(f"   Inference speed: {1/rtf:.2f}x real-time")
    
    # Save results
    print("\nSaving results...")
    
    # JSON results
    phase1_results = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 1 - Baseline Validation',
        'model': {
            'name': 'WeNet Conformer-CTC',
            'parameters': total_params,
            'vocab_size': len(vocab)
        },
        'data': {
            'dataset': 'AISHELL-1',
            'num_samples': len(results),
            'quick_test': QUICK_TEST
        },
        'metrics': {
            'cer': overall_cer * 100,
            'rtf': rtf,
            'speedup': 1/rtf if rtf > 0 else 0
        },
        'results': results
    }
    
    with open(RESULTS_DIR / 'phase1_results.json', 'w', encoding='utf-8') as f:
        json.dump(phase1_results, f, indent=2, ensure_ascii=False)
    
    # Text predictions
    with open(RESULTS_DIR / 'baseline_predictions.txt', 'w', encoding='utf-8') as f:
        f.write(f"# WeNet ASR Phase 1 - Baseline Predictions\n")
        f.write(f"# Overall CER: {overall_cer*100:.2f}%\n\n")
        for r in results:
            f.write(f"{r['audio_id']}\t{r['cer']*100:.2f}%\t{r['reference']}\t{r['prediction']}\n")
    
    print(f"✅ Results saved to: {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("                    PHASE 1 COMPLETE ✅")
    print("=" * 70)
    
    return overall_cer, rtf


if __name__ == "__main__":
    main()
