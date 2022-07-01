import pandas as pd
import numpy as np



def get_dataframes():
	"""Converts CSV's into pandas dataframes. Output: Train, Test"""
	return pd.read_csv("train_data.csv"), pd.read_csv("test_data.csv")

def clean_data(df, test=False):
	"""Renames columns, and sets ID as index. Set test to true if cleaning test data."""
	
	df = df.rename(columns={"sor":"Record_source", "cdf_seq_no":"id", "amt":"purchase_amount", "merchant_cat_code":"cat_code", "defualt_location":"location", "coalesced_brand":"brand"})
	
	if(test==False):
		return df[["id", "Record_source", "trans_desc", "cat_code", "purchase_amount", "payment_category", "is_international", "brand", "Category"]].set_index("id")
	
	return df[["id", "Record_source", "trans_desc", "cat_code", "purchase_amount", "payment_category", "is_international", "brand"]].set_index("id")


def extract_probabilities_CADA(df, model):
	"""Get exploded DF contaning regression output from CADA(Catergory Code ADA Boost) model."""
	df = df.dropna(subset=["cat_code"]).copy()
	series = df["cat_code"].apply(lambda row: __get_cat_model_probabilties__(model, row))
	series_exploded =  pd.DataFrame(series.apply(lambda x: x[0]))
	return pd.DataFrame(series_exploded["cat_code"].to_list(), columns=["cat_code_prob_class_" + str(x) for x in range(1, 11)], index=series_exploded.index)

	
def __get_cat_model_probabilties__(model, row):
	"""UDF to take a category code and spit out probabilities for each of category."""
	return model.predict_proba(np.asarray(row).reshape(-1, 1))


class features:

	@staticmethod
	def recuring_column(df):
		df["recuring"] = df["trans_desc"].apply(lambda row: 1 if "RECUR" in row else 0)
		return df
	
	@staticmethod
	def __normalize_nonbinary_columns__(column_index_dict, uv_length, row):
		return round(column_index_dict[row] / uv_length, 2)
	
	@staticmethod
	def __get_index_dict__(column):
		unique_values = list(column.unique())
		
		uv_length = len(unique_values) - 1
		
		column_index_dict = {k: v for v, k in enumerate(unique_values)}
		
		return column_index_dict, uv_length
	
	
	@staticmethod
	def normalize_columns(df):
		df["Record_source_normalized"] = df["Record_source"].apply(lambda row: 1 if row == "HH" else 0)
		
		df["is_international"] = df["is_international"].apply(lambda row: 1 if row == "TRUE" else 0)
		
		column_index_dict, uv_length = features.__get_index_dict__(df["payment_category"])
		
		df["payment_category_normalized"] = df["payment_category"].apply(lambda row: features.__normalize_nonbinary_columns__(column_index_dict, uv_length, row))
		
		max_amt_dict = df["purchase_amount"].max()
		
												
		
		df["purchase_amount_normalized"] = df.apply(lambda row: round(row["purchase_amount"] / max_amt_dict, 20), axis=1)
		
		
		return df
	
		