import pandas as pd
import numpy as np
import os, sys, ipdb

if __name__ == "__main__":

    COMP = 'spaceship_titanic'
    PATH = f'./submissions/{COMP}'
    os.makedirs('./submissions_kaggle_50/', exist_ok=True)
    dataframes = {}
    for f in os.listdir(PATH):
        if(f.endswith('.csv') and ('lats_outcome_rewards_agent' in f)):
            dataframes[f] = pd.read_csv(os.path.join(PATH, f))

    test_df = pd.read_csv(f'./data/{COMP}/test.csv')
    test_data_ids = test_df['PassengerId'].to_frame()
    # ipdb.set_trace()

    for df_name, df in dataframes.items():
        # Drop the first column if there are 2 columns
        if len(df.columns) == 2:
            df = df.drop(columns=[df.columns[0]])

        if len(df.columns) == 1:
            # Rename the single column to 'Transported'
            old_column = df.columns[0]
            df = df.rename(columns={old_column: 'Transported'})
            
            # Convert values to True/False (handles 1/0, True/False, and other truthy/falsy values)
            df['Transported'] = df['Transported'].astype(bool)

            df_new = pd.concat([test_data_ids, df], axis=1)
            
            # Update the dataframe in the dictionary
            dataframes[df_name] = df_new

            df_new.to_csv(os.path.join('./submissions_kaggle_50/', df_name), index=False)
        elif len(df.columns) == 2:
            cols = list(df.columns)
            
            # Try to identify prediction column by checking unique values
            prediction_col_idx = None
            for i, col in enumerate(cols):
                unique_vals = set(df[col].dropna().unique())
                # Check if column contains only boolean-like values
                if unique_vals.issubset({True, False, 1, 0, 1.0, 0.0}):
                    prediction_col_idx = i
                    break
            
            # If no clear prediction column, assume second column is prediction
            if prediction_col_idx is None:
                prediction_col_idx = 1
            
            id_col_idx = 1 - prediction_col_idx  # The other column
            
            # Create new DataFrame with proper column names
            df_new = pd.DataFrame()
            df_new['PassengerId'] = df.iloc[:, id_col_idx]
            df_new['Transported'] = df.iloc[:, prediction_col_idx].astype(bool)
            
            # Update the dataframe in the dictionary
            dataframes[df_name] = df_new

            df_new.to_csv(os.path.join('./submissions_kaggle_50/', df_name), index=False)
            # print(f"Processed {df_name}: renamed to 'PassengerId' and 'Transported', converted to bool")
        else:
            raise ValueError(f"Unexpected number of columns in {df_name}: {len(df.columns)}")

    # ipdb.set_trace()
