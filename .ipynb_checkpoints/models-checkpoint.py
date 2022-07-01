from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier

from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
def ADA_Regression(train_df):
	"""Trains an ADA boost model on the category code."""
	train_df= train_df.dropna(subset=["cat_code"])
	x = train_df["cat_code"].astype(int).to_numpy().reshape(-1, 1)
	
	classes = list(train_df["Category"].unique())
	classes_dict =  {k: v for v, k in enumerate(classes)}
	y = train_df["Category"].apply(lambda x: classes_dict[x]).to_numpy()
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
	
	

	model = AdaBoostClassifier(n_estimators=100).fit(X_train, y_train)
	print(f"Trained ADA Boost Model with mean accuracy of {model.score(X_test, y_test)}")
	return model
def __extract_key_words__(kw_minimum, df):
	brand_name_list = df["brand"].tolist()
	brand_names = [[x for xs in brand_name_list for x in xs.split()]][0]
	brand_names_counted = dict(Counter(brand_names))
	brand_names_counted = {k:v for k,v in brand_names_counted.items() if v > kw_minimum}
	
	feature_df = df.copy()[["brand", "Category"]].reset_index()
	
	for v in brand_names_counted.keys():
		feature_df[f"Has_{v}"] = feature_df["brand"].apply(lambda row: 1 if v in row else 0)
	return feature_df.drop(columns=["id", "brand"])

def keyword_classifier(kw_minimum, df):
	feature_df = __extract_key_words__(kw_minimum, df)
	
	
	y = to_categorical(pd.get_dummies(feature_df["Category"])).reshape(-1, 1)
	
	
	x = feature_df.drop(columns=["Category"])
	
	input_size = len(x.columns)
	
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)	
	
	
	model = Sequential()
	model.add(Input(shape=(input_size,)))
	model.add(Dense(round(input_size/2, 0), activation="relu"))
	model.add(Dense(10, activation="softmax"))
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
	
	model.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_data=[X_test, y_test])
	
	return model
	