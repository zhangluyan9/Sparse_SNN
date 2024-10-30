import pandas as pd
import matplotlib.pyplot as plt

# Data
new_data = {
    "SNN_accuracy": [93.01, 92.36, 92.08, 91.86, 91.42, 91.37, 90.43, 90.38],
    "sparsity_rate": [92.7257, 94.4107, 94.8784, 95.1676, 95.5376, 95.8118, 96.1811, 96.2803]
}

new_df = pd.DataFrame(new_data)

# Plotting the new data
plt.figure(figsize=(10, 6))
plt.plot(new_df['SNN_accuracy'], new_df['sparsity_rate'], marker='o', linestyle='-', color='darkblue')

# Annotating with coordinates
for i, txt in enumerate(new_df['sparsity_rate']):
    if i == len(new_df) - 2:
        offset = (5, 10)  # Second to last point above
    elif i == len(new_df) - 1:
        offset = (20, -20) # Last point below
    elif i == 0:
        offset = (-10, 5) # Last point below
    else:
        offset = (0, 5)  # All other points above
    
    plt.annotate(f"({new_df['SNN_accuracy'][i]:.2f}, {txt:.2f})",
                 (new_df['SNN_accuracy'][i], txt),
                 textcoords="offset points",  # how to position the text
                 xytext=offset,  # dynamic distance from text to points (x,y)
                 ha='center',fontsize=14)  # horizontal alignment can be left, right or center

plt.xlabel('SNN Accuracy',fontsize=14)
plt.ylabel('Sparsity Rate (%)',fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
#plt.title('SNN Accuracy vs. Sparsity Rate')
plt.savefig('./d_acc_sparsity.pdf', format='pdf', bbox_inches='tight')
plt.show()
