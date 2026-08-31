#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modality Contribution Analysis During Token Generation (usr2)

Ported from Dr-SHAP-AV's Compute_Generative_SHAP.py, scoped to a single
model (usr2) instead of a 3-method comparison. Computes windowed audio/video
contributions from a Shapley matrix saved by shap_evaluator.py and plots
Clean vs Noisy conditions.

Unlike the original 3-model script, usr2's num_audio_tokens already stores
the exact N_a used in the Shapley matrix (see shap_evaluator.py), so no
per-model downsample-factor guessing is needed here.

Usage:
    python Compute_Generative_SHAP.py \
        --clean-path path/to/usr2_clean.npz \
        --noisy-path path/to/usr2_noisy.npz \
        --num-samples 20 --num-windows 5
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

sns.set_context('paper')
rc('font', **{'family': 'cursive', 'cursive': ['Comic Sans MS']})

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(
    description='Modality Contribution Analysis During Token Generation (usr2)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--clean-path', required=True, metavar='PATH',
                    help='Path to usr2 clean .npz file.')
parser.add_argument('--noisy-path', required=True, metavar='PATH',
                    help='Path to usr2 noisy .npz file.')
parser.add_argument('--num-samples', required=True, type=int, default=20,
                    help='The number of samples to use to compute generative SHAP. By default, we use the 20 longest ones.')
parser.add_argument('--num-windows', required=True, type=int, default=5,
                    help='Number of windows.')

args = parser.parse_args()

num_samples = args.num_samples
num_windows = args.num_windows

color_clean = 'xkcd:teal'
color_noisy = 'xkcd:coral'


# =============================================================================
# FUNCTION TO COMPUTE WINDOWED CONTRIBUTIONS
# =============================================================================
def compute_windowed_contributions(npz_file, num_samples=10, num_windows=10):
    """
    Load SHAP values and compute windowed audio/video contributions.

    Returns:
    --------
    audio_mean, audio_std, video_mean, video_std : np.arrays of shape (num_windows,)
    """
    data = np.load(npz_file, allow_pickle=True)

    shap_values_all = data['shap_values']
    num_audio_tokens = data['num_audio_tokens']

    # Use the longest utterances for stable windowed estimates, as in Dr-SHAP-AV.
    sorted_indices_desc = sorted(range(len(num_audio_tokens)), key=lambda i: num_audio_tokens[i], reverse=True)

    all_audio = []
    all_video = []

    for sample_idx in range(num_samples):
        idx = sorted_indices_desc[sample_idx]
        shap_values = shap_values_all[idx]
        N_a = num_audio_tokens[idx]
        T_out = shap_values.shape[1]

        window_boundaries = np.linspace(0, T_out, num_windows + 1).astype(int)

        audio_windowed = []
        video_windowed = []

        for w in range(num_windows):
            start_idx = window_boundaries[w]
            end_idx = window_boundaries[w + 1]

            if end_idx <= start_idx:
                audio_windowed.append(audio_windowed[-1] if audio_windowed else 0.5)
                video_windowed.append(video_windowed[-1] if video_windowed else 0.5)
                continue

            audio_win = np.abs(shap_values[:N_a, start_idx:end_idx]).sum()
            video_win = np.abs(shap_values[N_a:, start_idx:end_idx]).sum()

            total_win = audio_win + video_win
            audio_windowed.append(audio_win / total_win)
            video_windowed.append(video_win / total_win)

        all_audio.append(audio_windowed)
        all_video.append(video_windowed)

        print(f"  Sample {sample_idx}: T_out={T_out}, N_audio={N_a}, N_video={shap_values.shape[0] - N_a}")

    all_audio = np.array(all_audio)
    all_video = np.array(all_video)

    return (np.mean(all_audio, axis=0), np.std(all_audio, axis=0),
            np.mean(all_video, axis=0), np.std(all_video, axis=0))


# =============================================================================
# COMPUTE CONTRIBUTIONS
# =============================================================================
print("=" * 60)
print("Computing windowed contributions...")
print("=" * 60)

print("\nusr2 - Clean:")
audio_clean_mean, audio_clean_std, _, _ = compute_windowed_contributions(
    args.clean_path, num_samples, num_windows)

print("\nusr2 - Noisy:")
audio_noisy_mean, audio_noisy_std, _, _ = compute_windowed_contributions(
    args.noisy_path, num_samples, num_windows)

# =============================================================================
# PLOT
# =============================================================================
print("\n" + "=" * 60)
print("Creating plot...")
print("=" * 60)

x_positions = np.linspace(5, 95, num_windows)

fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

ax.fill_between(x_positions,
                (audio_clean_mean - audio_clean_std) * 100,
                (audio_clean_mean + audio_clean_std) * 100,
                color=color_clean, alpha=0.15)
ax.fill_between(x_positions,
                (audio_noisy_mean - audio_noisy_std) * 100,
                (audio_noisy_mean + audio_noisy_std) * 100,
                color=color_noisy, alpha=0.15)

ax.plot(x_positions, audio_clean_mean * 100, color=color_clean,
        linewidth=3, marker='o', markersize=18,
        markeredgecolor='xkcd:charcoal grey', markeredgewidth=2., linestyle='-',
        label='Clean (∞ SNR)')
ax.plot(x_positions, audio_noisy_mean * 100, color=color_noisy,
        linewidth=3, marker='o', markersize=18,
        markeredgecolor='xkcd:charcoal grey', markeredgewidth=2., linestyle='-',
        label='Noisy')

ax.set_xlabel('Token Generation Progress (%)', fontsize=20)
ax.set_ylabel('Audio Contribution (%)', fontsize=20)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.tick_params(axis='both', labelsize=14)

ax2 = ax.twinx()
ax2.set_ylim(100, 0)
ax2.set_ylabel('Video Contribution (%)', fontsize=20)
ax2.tick_params(axis='y', labelsize=14)

ax.grid(color='#95a5a6', linestyle='--', linewidth=0.5, alpha=0.4)
ax.xaxis.set_minor_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(5))
ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='#cccccc', linewidth=0.5)

ax.legend(title='Condition', fontsize=15, title_fontsize=16, loc='best',
          fancybox=True, shadow=True, framealpha=0.95)

plt.tight_layout()
plt.savefig('usr2_modality_contribution_output_tokens.pdf', dpi=400, bbox_inches='tight')

print("\nPlot saved to 'usr2_modality_contribution_output_tokens.pdf'")
plt.show()
