from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd

from gensim.models import KeyedVectors
from tqdm.notebook import trange, tqdm


def ADA_Regression(train_df):
    """Trains an ADA boost model on the category code."""
    train_df = train_df.dropna(subset=["cat_code"])
    x = train_df["cat_code"].astype(int).to_numpy().reshape(-1, 1)

    classes = list(train_df["Category"].unique())
    classes_dict = {k: v for v, k in enumerate(classes)}
    y = train_df["Category"].apply(lambda x: classes_dict[x]).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = AdaBoostClassifier(n_estimators=100).fit(x_train, y_train)
    print(f"Trained ADA Boost Model with mean accuracy of {model.score(x_test, y_test)}")
    return model


def __extract_key_words__(kw_minimum, df):
    brand_name_list = df["brand"].tolist()
    brand_names = [[x for xs in brand_name_list for x in xs.split()]][0]
    brand_names_counted = dict(Counter(brand_names))
    brand_names_counted = {k: v for k, v in brand_names_counted.items() if v > kw_minimum}

    feature_df = df.copy()[["brand", "Category"]].reset_index()

    for v in brand_names_counted.keys():
        feature_df[f"Has_{v}"] = feature_df["brand"].apply(lambda row: 1 if v in row else 0)
    return feature_df.drop(columns=["id", "brand"])


def __load_W2V_model__(model_path):
    try:
        return KeyedVectors.load_word2vec_format(model_path, binary=True)
    except:
        print("Failed to load model!")


def generate_brand_vectors(df, model_path='D:\GoogleNews-vectors-negative300.bin\GoogleNews-vectors-negative300.bin'):
    model = __load_W2V_model__(model_path)

    df["vectors"] = df["brand"].apply(lambda row: __w2v_lambda__(row, model))
    df_exploded = pd.DataFrame(df["vectors"].to_list(), columns=[f"w2v_token_{x}" for x in range(17)], index=df.index)
    return df


def __w2v_lambda__(row, model):
    split_brand = row.split()
   
    vocab = model.key_to_index

    ls = []

    split_brand_length = len(split_brand)

    for x in range(17):
        if(x < split_brand_length and split_brand[x] in vocab): 
            ls.append(model[split_brand[x]])
        else:
            ls.append(0.0)
    return ls


def keyword_classifier(kw_minimum, df):
    feature_df = __extract_key_words__(kw_minimum, df)

    y = feature_df["Category"]

    encoder = LabelEncoder()
    encoder.fit(y)
    y = encoder.transform(y)

    y = np_utils.to_categorical(y)
    x = feature_df.drop(columns=["Category"])

    input_size = len(x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Input(shape=(input_size,)))
    model.add(Dense(input_size, activation="relu"))
    model.add(Dense(10, activation="softmax"))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=10, epochs=5, validation_data=(x_test, y_test))

    return model
