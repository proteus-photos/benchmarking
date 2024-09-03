import matplotlib.pyplot as plt
import numpy as np

# Data
samples = [16, 32, 64, 128]
methods = ['IG', 'Optimized IG', 'BlurIG', 'Optimized BlurIG']

insertion_scores = {
    'IG': [0.36155823, 0.36369896, 0.36567894, 0.368],
    'Optimized IG': [0.4396924, 0.44105795, 0.44110486, 0.445],
    'BlurIG': [0.3026216, 0.32038304, 0.32449973, 0.32510218],
    'Optimized BlurIG': [0.32414642, 0.32445028, 0.32535937, 0.3402909],
}

# Colors and markers
colors = ['red', 'red', 'blue', 'blue']
markers = ['v', 's', 'v', 's']
linestyles = [':', '-', ':', '-']

# Create the figure and subplots
fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 5))

# Plot Insertion Score
for i, method in enumerate(methods):
    ax1.plot(samples, insertion_scores[method], color=colors[i], marker=markers[i],
             linestyle=linestyles[i], label=method)

ax1.set_xscale('log', base=2)
ax1.set_xlabel('Samples')
ax1.set_ylabel('Score')
ax1.set_xticks(samples)
ax1.set_xticklabels(['$16$', '$32$', '$64$', '$128$'])
ax1.grid(True, which='both', linestyle='--', alpha=0.7)

# # Plot Normalized Insertion Score
# for i, method in enumerate(methods):
#     ax2.plot(samples, normalized_insertion_scores[method], color=colors[i], marker=markers[i],
#              linestyle=linestyles[i], label=method)

# ax2.set_xscale('log', base=2)
# ax2.set_xlabel('Samples')
# ax2.set_ylabel('Normalized Insertion Score (↑)')
# ax2.set_title('Normalized Insertion Score (↑)')
# ax2.set_xticks(samples)
# ax2.set_xticklabels(['$2^4$', '$2^5$', '$2^6$', '$2^7$'])
# ax2.set_ylim(0.3, 0.55)
# ax2.grid(True, which='both', linestyle='--', alpha=0.7)

# # Add a single legend for both subplots
# handles, labels = ax2.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05),
#            ncol=3, fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig('insertion_score.png')