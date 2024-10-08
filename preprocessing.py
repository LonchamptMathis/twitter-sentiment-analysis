import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

def data_preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a preprocessing on data : 
    text preprocessing, row deletion if there is no text, encodes the label column, etc.
    """
    # 1) Drops row with no text data
    dataset.dropna(inplace=True)

    # 2) Applies text preprocessing on the tweet column
    dataset["Preprocessed tweet"] = dataset["tweet"].apply(preprocessing)

    # 3) Encodes the label column (LabelEncoder)
    le = LabelEncoder()
    dataset['Encoded label'] = le.fit_transform(dataset['label'])

    return dataset
                       
def preprocessing(text: str, param: str = "lemmatization") -> str:
    """Text preprocessing : stop words deletion, stemming"""    
    # 1) Tokenization
    tokens = word_tokenize(text)

    # 2) Lower-case word conversion
    tokens = [word.lower() for word in tokens]

    # 3) Ponctuation deletion
    tokens = [word for word in tokens if word.isalpha()]

    # 4) Stopwords deletion
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5) Stemming or Lemmatization
    if param == "stemming":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]
    elif param == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    else:
        raise ValueError("Param must be either lemmatization or stemming.")
    
    return " ".join(tokens)

def get_X_y(dataset: pd.DataFrame):
    """Split the dataset into features (X) and the variable to be predicted (y)"""
    features = ["Preprocessed tweet"]
    variables = ["Encoded label"]

    X = dataset[features]
    y = dataset[variables]

    return X, y