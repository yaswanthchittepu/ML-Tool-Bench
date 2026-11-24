from pandas.io.sql import com
import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
import pandas as pd
import numpy as np
import sys, ipdb, os, ipdb

def logistic_regression_model():

    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int'})

    train_df, test_df = fe.split_combined_into_train_test(combined_df)

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    best_model = hpt.tune_logistic_regressor(X_train, Y_train)

    ut.save_model(best_model, './models/logistic_regressor_model.pkl')

    Y_test_pred = ev.predict_target(best_model, X_test, 'logistic_regressor')
    Y_test_pred = fe.cast_columns(Y_test_pred, {'predicted_target': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    results_df = fe.rename_feature(results_df, ['predicted_target'], ['Transported'])
    ut.save_dataframe_to_csv(results_df, './submissions/logistic_regressor_submission.csv')

def random_forest_classifier_model():

    # Best parameters: {'max_depth': 10, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}, Best CV score: 0.7993
    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int'})

    train_df, test_df = fe.split_combined_into_train_test(combined_df)

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    best_model = hpt.tune_random_forest_classifier(X_train, Y_train)

    ut.save_model(best_model, './models/random_forest_classifier_model.pkl')

    Y_test_pred = ev.predict_target(best_model, X_test, 'random_forest_classifier')
    Y_test_pred = fe.cast_columns(Y_test_pred, {'predicted_target': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    results_df = fe.rename_feature(results_df, ['predicted_target'], ['Transported'])
    ut.save_dataframe_to_csv(results_df, './submissions/random_forest_classifier_submission.csv')

def xgboost_classifier_model():

    # Best parameters: {'max_depth': 20, 'n_estimators': 100, 'subsample': 1.0}, Best CV score: 0.7782
    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int'})

    train_df, test_df = fe.split_combined_into_train_test(combined_df)

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    best_model = hpt.tune_xgboost_classifier(X_train, Y_train)

    ut.save_model(best_model, './models/xgboost_classifier_model.pkl')

    Y_test_pred = ev.predict_target(best_model, X_test, 'xgboost_classifier')
    Y_test_pred = fe.cast_columns(Y_test_pred, {'predicted_target': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    results_df = fe.rename_feature(results_df, ['predicted_target'], ['Transported'])
    ut.save_dataframe_to_csv(results_df, './submissions/xgboost_classifier_submission.csv')


def lightgbm_classifier_model():

    # Best parameters: {'learning_rate': 0.01, 'max_depth': 20, 'n_estimators': 100, 'subsample': 0.8}, Best CV score: 0.7886
    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int'})

    train_df, test_df = fe.split_combined_into_train_test(combined_df)

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    best_model = hpt.tune_lightgbm_classifier(X_train, Y_train)

    ut.save_model(best_model, './models/lightgbm_classifier_model.pkl')

    Y_test_pred = ev.predict_target(best_model, X_test, 'lightgbm_classifier')
    Y_test_pred = fe.cast_columns(Y_test_pred, {'predicted_target': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    results_df = fe.rename_feature(results_df, ['predicted_target'], ['Transported'])
    ut.save_dataframe_to_csv(results_df, './submissions/lightgbm_classifier_submission.csv')

def catboost_classifier_model():

    # Best parameters: {'depth': 6, 'iterations': 50, 'l2_leaf_reg': 1, 'learning_rate': 0.1}, Best CV score: 0.7997
    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    # Add more complexfunctionality to fillna's
    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int'})

    train_df, test_df = fe.split_combined_into_train_test(combined_df)
    
    print(train_df.dtypes, '\n\n########################\n\n', test_df.dtypes)
    sys.exit()

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    best_model = hpt.tune_catboost_classifier(X_train, Y_train)

    ut.save_model(best_model, './models/catboost_classifier_model.pkl')

    Y_test_pred = ev.predict_target(best_model, X_test, 'catboost_classifier')
    Y_test_pred = fe.cast_columns(Y_test_pred, {'predicted_target': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    results_df = fe.rename_feature(results_df, ['predicted_target'], ['Transported'])
    ut.save_dataframe_to_csv(results_df, './submissions/catboost_classifier_submission.csv')

def ensemble_model():
    
    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int'})

    train_df, test_df = fe.split_combined_into_train_test(combined_df)

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    logistic_regressor_model = ut.load_model('./models/logistic_regressor_model.pkl')
    random_forest_classifier_model = ut.load_model('./models/random_forest_classifier_model.pkl')
    xgboost_classifier_model = ut.load_model('./models/xgboost_classifier_model.pkl')
    lightgbm_classifier_model = ut.load_model('./models/lightgbm_classifier_model.pkl')
    catboost_classifier_model = ut.load_model('./models/catboost_classifier_model.pkl')

    Y_test_pred =md.ensemble_classification_predictions([logistic_regressor_model, random_forest_classifier_model, xgboost_classifier_model, lightgbm_classifier_model, catboost_classifier_model], X_test, method='weighted')
    Y_test_pred = fe.convert_to_dataframe(Y_test_pred)
    Y_test_pred = fe.rename_feature(Y_test_pred, [0], ['Transported'])
    Y_test_pred = fe.cast_columns(Y_test_pred, {'Transported': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    ut.save_dataframe_to_csv(results_df, './submissions/ensemble_classifier_submission.csv')


# This underperforms compared to catboost, irrespective of filling in Nan's logically
def catboost_classifier_nans_model():

    # Best parameters: {'depth': 6, 'iterations': 50, 'l2_leaf_reg': 1, 'learning_rate': 0.1}, Best CV score: 0.7997
    train_df_raw = ld.read_data("data/train.csv")
    test_df_raw = ld.read_data("data/test.csv")

    # Set the target column to True for the test data (dummy data)
    test_df_raw = fe.create_conditional_feature(test_df_raw, 'Transported', 'True', True, False)

    combined_df = fe.concatenate_train_test(train_df_raw, test_df_raw)

    # Add more complexfunctionality to fillna's
    # combined_df = fe.fillna_with_median(combined_df)
    # combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.create_numeric_feature(combined_df, 'TotalExpenditure', 'Spa + ShoppingMall + RoomService + FoodCourt + VRDeck')
    # combined_df = fe.drop_feature(combined_df, ['Spa', 'ShoppingMall', 'RoomService', 'FoodCourt', 'VRDeck'])

    # combined_df = fe.fillna_with_condition(combined_df, 'CryoSleep', 'TotalExpenditure == 0', True)
    # combined_df = fe.fillna_with_condition(combined_df, 'CryoSleep', 'TotalExpenditure > 0', False)
    # combined_df = fe.fillna_with_condition(combined_df, 'CryoSleep', 'Age <= 12', True)

    # combined_df = fe.create_conditional_feature(combined_df, 'IsChild', 'Age <= 12', True, False)
    # agg_child = fe.get_group_aggregation(combined_df, 'IsChild', 'Age', 'median')
    # combined_df = fe.fillna_with_condition(combined_df, 'Age', 'TotalExpenditure > 0 or CryoSleep == False', agg_child[False])
    # combined_df = fe.drop_feature(combined_df, ['IsChild'])
    # combined_df = fe.create_conditional_feature(combined_df, 'IsChild', 'Age <= 12', True, False)
    # combined_df = fe.fillna_with_condition(combined_df, 'TotalExpenditure', 'IsChild or CryoSleep', 0)

    combined_df = fe.extract_string_pattern(combined_df, 'LastName', 'Name', r'\b(\w+)$')
    combined_df = fe.extract_string_pattern(combined_df, 'GroupId', 'PassengerId', r'^(\w+)_\w+$')
    combined_df = fe.extract_string_pattern(combined_df, 'Deck', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=1)
    combined_df = fe.extract_string_pattern(combined_df, 'CabinNum', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=2)
    combined_df = fe.extract_string_pattern(combined_df, 'Side', 'Cabin', r'^(\w+)/(\d+)/(P|S)$', group=3)

    combined_df = fe.create_group_aggregation(combined_df, 'FamilySize', 'LastName', 'LastName', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'GroupSize', 'GroupId', 'GroupId', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'CabinSize', 'Cabin', 'Cabin', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'DeckSize', 'Deck', 'Deck', 'count')
    combined_df = fe.create_group_aggregation(combined_df, 'SideSize', 'Side', 'Side', 'count')

    combined_df = fe.drop_feature(combined_df, ['Name', 'LastName', 'PassengerId', 'GroupId', 'Cabin'])

    combined_df = fe.fillna_with_median(combined_df)
    combined_df = fe.fillna_with_mode(combined_df)

    combined_df = fe.cast_columns(combined_df, {'CabinNum': 'int'})

    combined_df = fe.normalize_all_numerical_columns(combined_df, method='standard')
    combined_df = fe.encode_all_categorical_columns(combined_df, method='one_hot')

    combined_df = fe.cast_columns(combined_df, {'CryoSleep': 'int', 'VIP': 'int', 'Transported': 'int', 
                                    # 'IsChild': 'int'
                                    })

    train_df, test_df = fe.split_combined_into_train_test(combined_df)

    X_train = fe.drop_feature(train_df, ['Transported'])
    X_test = fe.drop_feature(test_df, ['Transported'])
    Y_train = fe.get_features(train_df, ['Transported'])

    best_model = hpt.tune_catboost_classifier(X_train, Y_train)

    ut.save_model(best_model, './models/catboost_classifier_model.pkl')

    Y_test_pred = ev.predict_target(best_model, X_test, 'catboost_classifier')
    Y_test_pred = fe.cast_columns(Y_test_pred, {'predicted_target': 'bool'})
    test_passenger_ids = fe.get_features(test_df_raw, ['PassengerId'])
    results_df = fe.concatenate_dataframes([test_passenger_ids, Y_test_pred], axis=1)
    results_df = fe.rename_feature(results_df, ['predicted_target'], ['Transported'])
    ut.save_dataframe_to_csv(results_df, './submissions/catboost_classifier_submission.csv')


if __name__ == "__main__":

    catboost_classifier_model()

    








