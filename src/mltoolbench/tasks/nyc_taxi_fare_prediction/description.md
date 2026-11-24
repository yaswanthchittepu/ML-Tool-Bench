

        In this playground competition, hosted in partnership with Google Cloud and Coursera, you are tasked with predicting the fare amount (inclusive of tolls) for a taxi ride in New York City given the pickup and dropoff locations. While you can get a basic estimate based on just the distance between the two points, this will result in an RMSE of $5-$8, depending on the model used (see the starter code for an example of this approach in Kernels). Your challenge is to do better than this using Machine Learning techniques!

        Data fields
        ID
        key - Unique string identifying each row in both the training and test sets. Comprised of pickup_datetime plus a unique integer, but this doesn't matter, it should just be used as a unique ID field.
        Required in your submission CSV. Not necessarily needed in the training set, but could be useful to simulate a 'submission file' while doing cross-validation within the training set.
        Features
        pickup_datetime - timestamp value indicating when the taxi ride started.
        pickup_longitude - float for longitude coordinate of where the taxi ride started.
        pickup_latitude - float for latitude coordinate of where the taxi ride started.
        dropoff_longitude - float for longitude coordinate of where the taxi ride ended.
        dropoff_latitude - float for latitude coordinate of where the taxi ride ended.
        passenger_count - integer indicating the number of passengers in the taxi ride.
        Target
        fare_amount - float dollar amount of the cost of the taxi ride. This value is only in the training set; this is what you are predicting in the test set and it is required in your submission CSV.

        Submission File
        
        For each key in the test set, you must predict a value for the fare_amount variable.

        The train data is in the {data_dir} folder and is named train.csv. 

        The test data is in the {data_dir} folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the {model_dir} folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    