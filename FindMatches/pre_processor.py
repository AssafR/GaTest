import pandas as pd
import re

def preprocess_data(df: pd.DataFrame, prefix):
    for (idx, row) in df.iterrows():
        # Handle Hotel Name
        field_name = prefix + '.hotel_name'
        strip_and_lower(df, field_name, idx, row)
        val = row.loc[field_name]
        if val:
            if val is not str:
                val = str(val)
            val.replace('@','')
        df.at[idx, field_name] = val


        field_name = prefix + '.city_name'
        strip_and_lower(df, field_name, idx, row)

        field_name = prefix + '.hotel_address'
        strip_and_lower(df, field_name, idx, row)

        field_name = prefix + '.postal_code'
        val = row.loc[field_name]
        if val:
            if val is not str:
                val = str(val)
            strip_and_lower(df, field_name, idx, row)
            val = re.sub(r"\s+", "", val) # Remove whitespaces
            val = re.sub(r"\:-", "", val) # Remove ":"
        df.at[idx, field_name] = val


def strip_and_lower(df, field_name, idx, row):
    name = str(row.loc[field_name])
    name = name.strip().lower()
    df.at[idx, field_name] = name

