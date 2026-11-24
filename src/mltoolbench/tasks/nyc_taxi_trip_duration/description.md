
        In this competition, Kaggle is challenging you to build a model that predicts the total ride duration of taxi trips in New York City. Your primary dataset is one released by the NYC Taxi and Limousine Commission, which includes pickup time, geo-coordinates, number of passengers, and several other variables.

        Data fields
        id - a unique identifier for each trip
        vendor_id - a code indicating the provider associated with the trip record
        pickup_datetime - date and time when the meter was engaged
        dropoff_datetime - date and time when the meter was disengaged
        passenger_count - the number of passengers in the vehicle (driver entered value)
        pickup_longitude - the longitude where the meter was engaged
        pickup_latitude - the latitude where the meter was engaged
        dropoff_longitude - the longitude where the meter was disengaged
        dropoff_latitude - the latitude where the meter was disengaged
        store_and_fwd_flag - This flag indicates whether the trip record was held in vehicle memory before sending to the vendor because the vehicle did not have a connection to the server - Y=store and forward; N=not a store and forward trip
        trip_duration - duration of the trip in seconds

        Submission File
        
        For every row in the dataset, submission files should contain the trip_duration. 

        The train data is in the {data_dir} folder and is named train.csv. 

        The test data is in the {data_dir} folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the {model_dir} folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    