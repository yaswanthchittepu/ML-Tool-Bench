import pandas as pd
import numpy as np
import os, sys, ipdb
import pickle
from dotenv import load_dotenv
load_dotenv()


if __name__ == "__main__":

    COMP = 'bank_marketing_campaign'
    TARGET_COL = 'deposit'
    ALGO = 'lats_reflection'
    POS_LABEL = 'yes'
    NEG_LABEL = 'no'
    SUBMISSION_PATH = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], 'submissions', COMP, ALGO)
    LOG_PATH = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], 'logs', COMP, ALGO)
    NUM_TRAJECTORIES = 30
    MATCHING_KEY = 'manual' if('hierarchical' in ALGO or 'reward' in ALGO) else ''
    # os.makedirs('./submissions_kaggle_50/', exist_ok=True)
    dataframes = {}
    for f in os.listdir(SUBMISSION_PATH):
        if(f.endswith('.csv') and (MATCHING_KEY in f)):
            dataframes[f] = pd.read_csv(os.path.join(SUBMISSION_PATH, f))
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

        df[col_name] = df[col_name].map({1: POS_LABEL, 0: NEG_LABEL, True: POS_LABEL, False: NEG_LABEL, 'True': POS_LABEL, 'False': NEG_LABEL, 'yes': POS_LABEL, 'no': NEG_LABEL, 'Yes': POS_LABEL, 'No': NEG_LABEL})
        # print(f"Converted {col_name} in {df_name} to yes/no format")
        # ipdb.set_trace()
        if(df[col_name].shape[0] != test_df_labels.shape[0]):
            continue
        acc[df_name] = (df[col_name] == test_df_labels).mean()*100


    acc_arr = np.array(list(acc.values())+[0]*(10-len(acc)))
    print(f"Consistency: {len(acc)/NUM_TRAJECTORIES}")
    print(f"Mean accuracy: {acc_arr.mean()}, Stdev: {acc_arr.std()}")
    print(f"Max accuracy: {acc_arr.max()}, Min accuracy: {acc_arr.min()}")
    print(f"Median accuracy: {np.median(acc_arr)}")
    pcts = np.percentile(acc_arr, [25, 75])
    print(f"25th percentile accuracy: {pcts[0]}, 75th percentile accuracy: {pcts[1]}, IQR : {pcts[1] - pcts[0]}")

    # Latency
    # Find all pickle files in the log directory
    pickle_files = [f for f in os.listdir(LOG_PATH) if f.endswith(".pkl") or f.endswith(".pickle")]

    if len(pickle_files) != 1:
        raise ValueError(f"Expected exactly 1 pickle file, but found {len(pickle_files)}: {pickle_files}")

    # Build full path
    pickle_path = os.path.join(LOG_PATH, pickle_files[0])

    # Load pickle file
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    # ipdb.set_trace()
    latency_list = []
    for dct in data:
        if('execution_time' in dct.keys()):
            latency_list.append(dct['execution_time'])
    latency_arr = np.array(latency_list)
    print(f"Mean Latency: {latency_arr.mean()}, Stdev: {latency_arr.std()}")
    print(f"Max Latency: {latency_arr.max()}, Min Latency: {latency_arr.min()}")
    print(f"Median Latency: {np.median(latency_arr)}")
    pcts = np.percentile(latency_arr, [25, 75])
    print(f"25th percentile Latency: {pcts[0]}, 75th percentile Latency: {pcts[1]}, IQR : {pcts[1] - pcts[0]}")



        
        