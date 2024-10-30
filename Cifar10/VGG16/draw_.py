import matplotlib.pyplot as plt
import numpy as np

# Define the functions for the example data
def f1(x):
    if x == 0:
        return np.nan  # Avoid division by zero by returning NaN
    else:
        return 1 - (0.45 * 80.23+2020/216-4040*x/(216*(1+x)) - 0.026868 * x) / (x * 64.32+x*0.03/2395)

def f2(x):
    if x == 0:
        return np.nan  # Avoid division by zero by returning NaN
    else:
        return 1 - (0.45 * 20.23 - 40.06 / 2395 * x) / (x * 20.03)

# Generate x values from 0 to 16 (inclusive)
x = np.arange(0, 17)  # np.arange to include 16
y1 = np.vectorize(f1)(x)
y2 = np.vectorize(f2)(x)


# Create the plot with enhanced aesthetics
plt.figure(figsize=(12, 8))
plt.plot(x, y1, 'o-', label='Classical Architecture', linewidth=2, markersize=8, color='darkblue')
plt.plot(x, y2, 's-', label='Spatial-Dataflow Architecture', linewidth=2, markersize=8, color='darkred')

# Styling the plot
#plt.title('Comparison of Energy Equivalence Points for ANN Implementations on Different Architectures', fontsize=16, fontweight='bold')
plt.xlabel('T', fontsize=14)
plt.ylabel('s', fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Annotations for each point
for i, (txt1, txt2) in enumerate(zip(y1, y2)):
    if not np.isnan(txt1):
        plt.annotate(f'{txt1:.2f}', (x[i], y1[i]), textcoords="offset points", xytext=(0,-20), ha='center', fontsize=14, color='black')
    if not np.isnan(txt2):
        plt.annotate(f'{txt2:.2f}', (x[i], y2[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=14, color='black')

# Enhancing legend and overall look
plt.legend(fontsize=14, loc='lower right')
# Set the style to be visually appealing
#plt.style.use('seaborn-darkgrid')
plt.savefig('./d_plot_dis_2.pdf', format='pdf', bbox_inches='tight')

# Show the plot
plt.show()

