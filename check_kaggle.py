
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df = pd.read_csv("train.csv")

# Print the first 5 rows
print(df.head())
#print(df.isnull())#empty values given as true
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()