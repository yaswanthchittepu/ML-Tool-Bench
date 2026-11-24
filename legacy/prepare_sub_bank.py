import pandas as pd
import numpy as np
import os, sys, ipdb

if __name__ == "__main__":

    COMP = 'customer_churn_prediction'
    TARGET_COL = 'Churn'
    PATH = f'./submissions/{COMP}'
    os.makedirs('./submissions_kaggle_50/', exist_ok=True)
    dataframes = {}
    for f in os.listdir(PATH):
        if(f.endswith('.csv') and ('lats_hierarchical_verifiable_rewards' in f)):
            dataframes[f] = pd.read_csv(os.path.join(PATH, f))
            print(dataframes[f])

    # sys.exit()
    test_df = pd.read_csv(f'./data/{COMP}/test_with_labels.csv')
    test_df_labels = test_df[TARGET_COL]

    acc = {}
    for df_name, df in dataframes.items():
        col_name = df.columns[0]
        # df['deposit'] is either 1/0 or True/False. Convert this to yes/no
        # Convert 1/0 or True/False to yes/no
        # df[col_name] = df[col_name].map({1: 'yes', 0: 'no', True: 'yes', False: 'no', 'True': 'yes', 'False': 'no', 'yes': 'yes', 'no': 'no'})

        df[col_name] = df[col_name].map({1: 'Yes', 0: 'No', True: 'Yes', False: 'No', 'True': 'Yes', 'False': 'No', 'yes': 'Yes', 'no': 'No', 'Yes': 'Yes', 'No': 'No'})
        # print(f"Converted {col_name} in {df_name} to yes/no format")
        # ipdb.set_trace()
        if(df[col_name].shape[0] != test_df_labels.shape[0]):
            continue
        acc[df_name] = (df[col_name] == test_df_labels).mean()*100


    acc_arr = np.array(list(acc.values())+[0]*(10-len(acc)))
    print(f"Consistency: {len(acc)/10.0}")
    print(f"Mean accuracy: {acc_arr.mean()}, Stdev: {acc_arr.std()}")
    print(f"Max accuracy: {acc_arr.max()}, Min accuracy: {acc_arr.min()}")
    print(f"Median accuracy: {np.median(acc_arr)}")
    pcts = np.percentile(acc_arr, [25, 75])
    print(f"25th percentile accuracy: {pcts[0]}, 75th percentile accuracy: {pcts[1]}, IQR : {pcts[1] - pcts[0]}")

        
        