import matplotlib.pyplot as plt
import numpy as np

# Data from the user's table
T = np.array([4, 5, 6, 7, 8])
SNN_accuracy = np.array([92.21, 92.74, 93.28, 93.28, 93.45])
spike_rate = np.array([8.3390, 7.8707, 7.5748, 7.3187, 7.2995])
spikes_per_neuron = np.array([0.3336, 0.3935, 0.4545, 0.5123, 0.5840])

# Create the plot with annotations without background color
plt.figure(figsize=(10, 6))
ax1 = plt.gca()

# Plotting SNN Accuracy
color = 'darkblue'
ax1.set_xlabel('Time Steps (T)', fontsize=14, fontweight='bold')
ax1.set_ylabel('SNN Accuracy (%)', color=color, fontsize=14, fontweight='bold')
line1, = ax1.plot(T, SNN_accuracy, 'o-', color=color, label='SNN Accuracy', linewidth=2, markersize=8)
ax1.tick_params(axis='y', labelcolor=color, labelsize=14)
ax1.set_xticks(T)
ax1.set_xticklabels(T, fontsize=14)

# Annotations for SNN Accuracy above the line without background color
for i, txt in enumerate(SNN_accuracy):
    ax1.annotate(f'{txt:.2f}', (T[i], SNN_accuracy[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14, verticalalignment='top')

# Secondary Y-axis for Spike Rate
ax2 = ax1.twinx()
color = 'darkred'
ax2.set_ylabel('Spike Rate (%)', color=color, fontsize=14, fontweight='bold')
line2, = ax2.plot(T, spike_rate, 's-', color=color, label='Sparsity Rate', linewidth=2, markersize=8)
ax2.tick_params(axis='y', labelcolor=color, labelsize=14)

# Annotations for Spike Rate without background color
for i, txt in enumerate(spike_rate):
    offset = -15 if i in [0, 4] else 10
    ax2.annotate(f'{txt:.2f}', (T[i], spike_rate[i]), textcoords="offset points", xytext=(0,offset), ha='center', fontsize=14)

# Third Y-axis for Spikes per Neuron
ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color = 'darkgreen'
ax3.set_ylabel('#Spikes/neuron', color=color, fontsize=14, fontweight='bold')
line3, = ax3.plot(T, spikes_per_neuron, 'd-', color=color, label='#Spikes/neuron', linewidth=2, markersize=8)
ax3.tick_params(axis='y', labelcolor=color, labelsize=14)

# Annotations for Spikes per Neuron below the line without background color
for i, txt in enumerate(spikes_per_neuron):
    ax3.annotate(f'{txt:.4f}', (T[i], spikes_per_neuron[i]), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=14, verticalalignment='bottom')

# Adjust layout, add grid and enhance legend
ax1.grid(True, linestyle='--', linewidth=0.5)
fig = plt.gcf()
fig.legend([line1, line2, line3], ['SNN Accuracy', 'Spike Rate', '#Spikes/neuron'], loc='upper left', bbox_to_anchor=(0.1, 0.9), fontsize=14)
plt.tight_layout()

# Show the improved plot
plt.show()

# Show the improved plot
plt.savefig('./d_plot_dis_.pdf', format='pdf', bbox_inches='tight')

plt.show()
