from sklearn.datasets import make_classification
import pandas as pd

n_samples = 1000 
n_features = 64
n_classes = 3
n_informative = 50
n_redundant = 10
n_repeated = 4
random_state = 42

# Generating synthetic dataset
X, y = make_classification(
    n_samples = n_samples,
    n_features = n_features,
    n_classes = n_classes,
    n_informative = n_informative,
    n_redundant = n_redundant,
    n_repeated = n_repeated,
    random_state = random_state
)

# Convert to a DataFrame for better readability
df = pd.DataFrame(X, columns = [f"Feature_{i + 1}" for i in range(X.shape[1])])
df['Class'] = y

# Saving the dataset to a CSV file
df.to_csv('synthetic_dataset.csv', index = False)
