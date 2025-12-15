import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")

# Load data
genai = pd.read_csv(BASE_DIR / "genai_features.csv")
meta = pd.read_csv(BASE_DIR / "daic_metadata.csv")
test_lbls = pd.read_csv(BASE_DIR / "full_test_split.csv")

# Clean IDs
genai['participant_id'] = genai['participant_id'].astype(str)
meta['participant_id'] = meta['participant_id'].astype(str)
test_lbls = test_lbls.rename(columns={'Participant_ID': 'participant_id', 'PHQ_Binary': 'phq8_binary'})
test_lbls['participant_id'] = test_lbls['participant_id'].astype(str)

# Update metadata with test labels
meta = meta.merge(test_lbls[['participant_id', 'phq8_binary']], on='participant_id', how='left', suffixes=('', '_new'))
meta['phq8_binary'] = meta['phq8_binary'].fillna(meta['phq8_binary_new'])

# Merge
df = meta.merge(genai, on='participant_id', how='inner')

# Plot
plt.figure(figsize=(12, 6))
features = ['cognitive_negativity', 'emotional_flatness', 'overall_risk']

# Melt for visualization
plot_df = df.melt(id_vars=['phq8_binary', 'participant_id'], 
                  value_vars=features, 
                  var_name='GenAI_Feature', value_name='Score')

# Boxplot
sns.boxplot(data=plot_df, x='GenAI_Feature', y='Score', hue='phq8_binary', palette={0: 'green', 1: 'red'})
plt.title("Distribution of GenAI Scores: Healthy (0) vs Depressed (1)")
plt.ylabel("Llama-2 Score (0-10)")
plt.grid(True, alpha=0.3)
plt.savefig("genai_distribution_check.png")

print("Distribution check saved to genai_distribution_check.png")

# Print numeric stats
print("\n--- Mean Scores by Group ---")
print(df.groupby('phq8_binary')[features].mean())

