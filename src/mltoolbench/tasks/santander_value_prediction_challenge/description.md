

        According to Epsilon research, 80% of customers are more likely to do business with you if you provide personalized service. Banking is no exception.

        The digitalization of everyday lives means that customers expect services to be delivered in a personalized and timely manner… and often before they´ve even realized they need the service. In their 3rd Kaggle competition, Santander Group aims to go a step beyond recognizing that there is a need to provide a customer a financial service and intends to determine the amount or value of the customer's transaction. This means anticipating customer needs in a more concrete, but also simple and personal way. With so many choices for financial services, this need is greater now than ever before.

        In this competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.

        You are provided with an anonymized dataset containing numeric feature variables, the numeric target column, and a string ID column.

        The task is to predict the value of target column in the test set.

        Submission File
        
        For every row in the test.csv, submission files should contain the target. 

        The train data is in the {data_dir} folder and is named train.csv. 

        The test data is in the {data_dir} folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the {model_dir} folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    