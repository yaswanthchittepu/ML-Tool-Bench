
        Your Goal: The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

        The target column is 'class', which is either 'p' (poisonous) or 'e' (edible).

        your objective is to predict target class for each row

        Submission File

        For each id row in the test set, you must predict target class, whether the observation is editable (e) or poisonous (p). 

        The train data is in the {data_dir} folder and is named train.csv. 

        The test data is in the {data_dir} folder and is named test.csv. 
        
        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the {model_dir} folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}.

        The task ends when you have successfully saved the predictions for the test data.
    