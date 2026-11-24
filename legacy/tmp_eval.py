import pickle
import xgboost as xgb
import os, sys
import pandas as pd

def evaluate_model(model_path: str):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

if __name__ == "__main__":
    # model_path = "model_saves/best_model.pkl"
    # model = evaluate_model(model_path)
    # print(model)
    df = pd.read_csv("data/train.csv")
    df['TotalExpenses'] = df['Spa'] + df['RoomService'] + df['VRDeck'] + df['FoodCourt'] + df['ShoppingMall']
    print(df[df['Age'] <= 12]['TotalExpenses'].isna().sum())