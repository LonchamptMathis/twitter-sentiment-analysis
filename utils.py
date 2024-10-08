import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_dataset(paths: str) -> pd.DataFrame:
    """Retrieve data and store it in a data frame"""
    if len(paths) != 2:
        raise ValueError("Please give path to training data and test data")
    cols = ["id", "topic", "label", "tweet"]
    data_1 = pd.read_csv(paths[0], names=cols)
    data_2 = pd.read_csv(paths[1], names=cols)
    dataset = pd.concat([data_1, data_2], axis=0).reset_index(drop=True)
    return dataset

def plot_label_distribution(df: pd.DataFrame):
    """Plot the label distribution"""
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=df, palette='Set2')
    plt.title('Distribution des classes (label)')
    plt.xlabel('Label')
    plt.ylabel('Nombre de tweets')
    plt.show()