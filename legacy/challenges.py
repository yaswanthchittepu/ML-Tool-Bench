import os

CHALLENGES = {
    "spaceship_titanic": """
        Welcome to the year 2912, where your data science skills are needed to solve a cosmic mystery. We've received a transmission from four lightyears away and things aren't looking good.

        The Spaceship Titanic was an interstellar passenger liner launched a month ago. With almost 13,000 passengers on board, the vessel set out on its maiden voyage transporting emigrants from our solar system to three newly habitable exoplanets orbiting nearby stars.

        While rounding Alpha Centauri en route to its first destination—the torrid 55 Cancri E—the unwary Spaceship Titanic collided with a spacetime anomaly hidden within a dust cloud. Sadly, it met a similar fate as its namesake from 1000 years before. Though the ship stayed intact, almost half of the passengers were transported to an alternate dimension!
        Read the data from the train.csv file in the data folder.

        To help rescue crews and retrieve the lost passengers, you are challenged to predict which passengers were transported by the anomaly using records recovered from the spaceship's damaged computer system.

        Help save them and change history!

        In this competition your task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. To help you make these predictions, you're given a set of personal records recovered from the ship's damaged computer system.

        File and Data Field Descriptions:
        
        PassengerId - A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.
        HomePlanet - The planet the passenger departed from, typically their planet of permanent residence.
        CryoSleep - Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.
        Cabin - The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
        Destination - The planet the passenger will be debarking to.
        Age - The age of the passenger.
        VIP - Whether the passenger has paid for special VIP service during the voyage.
        RoomService, FoodCourt, ShoppingMall, Spa, VRDeck - Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.
        Name - The first and last names of the passenger.
        Transported - Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

        A submission file in the correct format.
            PassengerId - Id for each passenger in the test set.
            Transported - The target. For each passenger, predict either True or False.

        The train data is in the data/spaceship_titanic folder and is named train.csv. 

        The test data is in the data/spaceship_titanic folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/spaceship_titanic folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all passengers in the test set. Name the file as {save_file_name}.

        The task ends when you have successfully saved the predictions for the test data.

        """,

    # "bank_marketing_campaign": """
    #     In this competetion your job is to predict whether a potential client will suscribe to a term deposit or not.

    #     Feature Descriptions:
    #     Ai. bank client data:
    #         1 - age: (numeric)
    #         2 - job: type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
    #         3 - marital: marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
    #         4 - education: (categorical: primary, secondary, tertiary and unknown)
    #         5 - default: has credit in default? (categorical: 'no','yes','unknown')
    #         6 - housing: has housing loan? (categorical: 'no','yes','unknown')
    #         7 - loan: has personal loan? (categorical: 'no','yes','unknown')
    #         8 - balance: Balance of the individual.

    #     Aii. Related with the last contact of the current campaign:
    #         9 - contact: contact communication type (categorical: 'cellular','telephone')
    #         10 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    #         11 - day: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
    #         12 - duration: last contact duration, in seconds (numeric). 

    #     Aiii. other attributes:
    #         13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    #         14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    #         15 - previous: number of contacts performed before this campaign and for this client (numeric)
    #         16 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

    #     Output variable (desired target):
    #         17 - deposit: has the client subscribed a term deposit? (binary: 'yes','no')

    #     The train data is in the data/bank_marketing_campaign folder and is named train.csv. 

    #     The test data is in the data/bank_marketing_campaign folder and is named test.csv. 

    #     You have to provide me the best possible model you can for the task, using the tools provided to you. 

    #     You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

    #     In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
    #     You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

    #     Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/bank_marketing_campaign folder.

    #     Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for datapoints in the test set. Name the file as {save_file_name}.

    #     """,

    # "customer_churn_prediction": """

    #     Customer churn, the phenomenon where customers stop using a company's products or services, is a critical metric for businesses across industries. High churn rates directly impact revenue, increase acquisition costs, and reduce long-term profitability.

    #     Accurately predicting which customers are likely to leave enables companies to implement targeted retention strategies, optimize marketing efforts, and maintain a loyal customer base.

    #     In this competetion your job is to predict whether a customer will churn or not.

    #     Each row represents a customer, each column contains customer's attributes described on the column Metadata.

    #     The data set includes information about:

    #     Customers who left within the last month: the column is called Churn. This is the target variable.

    #     Services that each customer has signed up for: phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies.

    #     Customer account information: how long they've been a customer, contract, payment method, paperless billing, monthly charges, and total charges.

    #     Demographic info about customers: gender, age range, and if they have partners and dependents.

    #     The train data is in the data/customer_churn_prediction folder and is named train.csv. 

    #     The test data is in the data/customer_churn_prediction folder and is named test.csv. 

    #     You have to provide me the best possible model you can for the task, using the tools provided to you. 

    #     You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

    #     In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
    #     You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

    #     Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/customer_churn_prediction folder.

    #     Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}.

    #     """,
    "santander_value_prediction_challenge": """

        According to Epsilon research, 80% of customers are more likely to do business with you if you provide personalized service. Banking is no exception.

        The digitalization of everyday lives means that customers expect services to be delivered in a personalized and timely manner… and often before they´ve even realized they need the service. In their 3rd Kaggle competition, Santander Group aims to go a step beyond recognizing that there is a need to provide a customer a financial service and intends to determine the amount or value of the customer's transaction. This means anticipating customer needs in a more concrete, but also simple and personal way. With so many choices for financial services, this need is greater now than ever before.

        In this competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.

        You are provided with an anonymized dataset containing numeric feature variables, the numeric target column, and a string ID column.

        The task is to predict the value of target column in the test set.

        Submission File
        
        For every row in the test.csv, submission files should contain the target. 

        The train data is in the data/santander_value_prediction_challenge folder and is named train.csv. 

        The test data is in the data/santander_value_prediction_challenge folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/santander_value_prediction_challenge folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "nyc_taxi_fare_prediction": """

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

        The train data is in the data/nyc_taxi_fare_prediction folder and is named train.csv. 

        The test data is in the data/nyc_taxi_fare_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/nyc_taxi_fare_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "nyc_taxi_trip_duration": """
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

        The train data is in the data/nyc_taxi_trip_duration folder and is named train.csv. 

        The test data is in the data/nyc_taxi_trip_duration folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/nyc_taxi_trip_duration folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "bpm_prediction": """
        Your Goal: The goal of this competition is to predict a song's beats-per-minute.

        BeatsPerMinute is the continuous target ground truth

        Submission File
        
        For each id in the test set, you must predict a continuous-valued BeatsPerMinute. 

        The train data is in the data/bpm_prediction folder and is named train.csv. 

        The test data is in the data/bpm_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/bpm_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "calorie_expenditure_prediction": """

        Your Goal: Your goal is to predict how many calories were burned during a workout.

        Calories is the continuous target

        Submission File
        
        For each id row in the test set, you must predict the continuous target, Calories.

        The train data is in the data/calorie_expenditure_prediction folder and is named train.csv. 

        The test data is in the data/calorie_expenditure_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/calorie_expenditure_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "california_housing_regression": """

        Your Goal: Your objective is to predict MedHouseVal, the house price in the California housing dataset.

        MedHouseVal is the continuous target

        Submission File

        For each id in the test set, you must predict the value for the target MedHouseVal.

        The train data is in the data/california_housing_regression folder and is named train.csv. 

        The test data is in the data/california_housing_regression folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/california_housing_regression folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 
        
        The task ends when you have successfully saved the predictions for the test data.
    """,
    "used_car_prices_regression": """
        Your Goal: The goal of this competition is to predict the price of used cars based on various attributes.

        price is the continuous target

        Submission File
        
        For each id in the test set, you must predict the price of the car. 

        The train data is in the data/used_car_prices_regression folder and is named train.csv. 

        The test data is in the data/used_car_prices_regression folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/used_car_prices_regression folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "porto_seguro_safe_driver_prediction": """
        Nothing ruins the thrill of buying a brand new car more quickly than seeing your new insurance bill. The sting's even more painful when you know you're a good driver. It doesn't seem fair that you have to pay so much if you've been cautious on the road for years.

        Porto Seguro, one of Brazil's largest auto and homeowner insurance companies, completely agrees. Inaccuracies in car insurance company's claim predictions raise the cost of insurance for good drivers and reduce the price for bad ones.

        In this competition, you're challenged to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. While Porto Seguro has used machine learning for the past 20 years, they're looking to Kaggle's machine learning community to explore new, more powerful methods. A more accurate prediction will allow them to further tailor their prices, and hopefully make auto insurance coverage more accessible to more drivers.

        In this competition, you will predict the probability that an auto insurance policy holder files a claim.

        In the train and test data, features that belong to similar groupings are tagged as such in the feature names (e.g., ind, reg, car, calc). In addition, feature names include the postfix bin to indicate binary features and cat to indicate categorical features. Features without these designations are either continuous or ordinal. Values of -1 indicate that the feature was missing from the observation. The target columns signifies whether or not a claim was filed for that policy holder.

        train.csv contains the training data, where each row corresponds to a policy holder, and the target columns signifies that a claim was filed.

        Submission File

        For each id in the test set, you must predict a probability of an insurance claim in the target column.
        
        The train data is in the data/porto_seguro_safe_driver_prediction folder and is named train.csv. 

        The test data is in the data/porto_seguro_safe_driver_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/porto_seguro_safe_driver_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "costa_rican_household_poverty_level_prediction": """

        The Inter-American Development Bank is asking the Kaggle community for help with income qualification for some of the world's poorest families. Are you up for the challenge?

        Here's the backstory: Many social programs have a hard time making sure the right people are given enough aid. It’s especially tricky when a program focuses on the poorest segment of the population. The world’s poorest typically can’t provide the necessary income and expense records to prove that they qualify.

        In Latin America, one popular method uses an algorithm to verify income qualification. It’s called the Proxy Means Test (or PMT). With PMT, agencies use a model that considers a family’s observable household attributes like the material of their walls and ceiling, or the assets found in the home to classify them and predict their level of need.

        While this is an improvement, accuracy remains a problem as the region’s population grows and poverty declines.

        To improve on PMT, the IDB (the largest source of development financing for Latin America and the Caribbean) has turned to the Kaggle community. They believe that new methods beyond traditional econometrics, based on a dataset of Costa Rican household characteristics, might help improve PMT’s performance.

        Beyond Costa Rica, many countries face this same problem of inaccurately assessing social need. If Kagglers can generate an improvement, the new algorithm could be implemented in other countries around the world.
    
        train.csv - the training set
        test.csv - the test set

        This is the main table, broken into two files for Train (with a Target column) and Test (without the Target column).
        One row represents one person in our data sample.
        Multiple people can be part of a single household. Only predictions for heads of household are scored.

        Core Data fields
            Id - a unique identifier for each row.
            Target - the target is an ordinal variable indicating groups of income levels.
            0 = extreme poverty
            1 = moderate poverty
            2 = vulnerable households
            3 = non vulnerable households
            idhogar - this is a unique identifier for each household. This can be used to create household-wide features, etc. All rows in a given household will have a matching value for this identifier.
            parentesco1 - indicates if this person is the head of the household.
            This data contains 142 total columns.
        
        All Data fields
        Variable name, Variable description

            v2a1, Monthly rent payment
            hacdor, =1 Overcrowding by bedrooms
            rooms,  number of all rooms in the house
            hacapo, =1 Overcrowding by rooms
            v14a, =1 has bathroom in the household
            refrig, =1 if the household has refrigerator
            v18q, owns a tablet
            v18q1, number of tablets household owns
            r4h1, Males younger than 12 years of age
            r4h2, Males 12 years of age and older
            r4h3, Total males in the household
            r4m1, Females younger than 12 years of age
            r4m2, Females 12 years of age and older
            r4m3, Total females in the household
            r4t1, persons younger than 12 years of age
            r4t2, persons 12 years of age and older
            r4t3, Total persons in the household
            tamhog, size of the household
            tamviv, number of persons living in the household
            escolari, years of schooling
            rez_esc, Years behind in school
            hhsize, household size
            paredblolad, =1 if predominant material on the outside wall is block or brick
            paredzocalo, "=1 if predominant material on the outside wall is socket (wood,  zinc or absbesto"
            paredpreb, =1 if predominant material on the outside wall is prefabricated or cement
            pareddes, =1 if predominant material on the outside wall is waste material
            paredmad, =1 if predominant material on the outside wall is wood
            paredzinc, =1 if predominant material on the outside wall is zink
            paredfibras, =1 if predominant material on the outside wall is natural fibers
            paredother, =1 if predominant material on the outside wall is other
            pisomoscer, "=1 if predominant material on the floor is mosaic,  ceramic,  terrazo"
            pisocemento, =1 if predominant material on the floor is cement
            pisoother, =1 if predominant material on the floor is other
            pisonatur, =1 if predominant material on the floor is  natural material
            pisonotiene, =1 if no floor at the household
            pisomadera, =1 if predominant material on the floor is wood
            techozinc, =1 if predominant material on the roof is metal foil or zink
            techoentrepiso, "=1 if predominant material on the roof is fiber cement,  mezzanine "
            techocane, =1 if predominant material on the roof is natural fibers
            techootro, =1 if predominant material on the roof is other
            cielorazo, =1 if the house has ceiling
            abastaguadentro, =1 if water provision inside the dwelling
            abastaguafuera, =1 if water provision outside the dwelling
            abastaguano, =1 if no water provision
            public, "=1 electricity from CNFL,  ICE,  ESPH/JASEC"
            planpri, =1 electricity from private plant
            noelec, =1 no electricity in the dwelling
            coopele, =1 electricity from cooperative
            sanitario1, =1 no toilet in the dwelling
            sanitario2, =1 toilet connected to sewer or cesspool
            sanitario3, =1 toilet connected to  septic tank
            sanitario5, =1 toilet connected to black hole or letrine
            sanitario6, =1 toilet connected to other system
            energcocinar1, =1 no main source of energy used for cooking (no kitchen)
            energcocinar2, =1 main source of energy used for cooking electricity
            energcocinar3, =1 main source of energy used for cooking gas
            energcocinar4, =1 main source of energy used for cooking wood charcoal
            elimbasu1, =1 if rubbish disposal mainly by tanker truck
            elimbasu2, =1 if rubbish disposal mainly by botan hollow or buried
            elimbasu3, =1 if rubbish disposal mainly by burning
            elimbasu4, =1 if rubbish disposal mainly by throwing in an unoccupied space
            elimbasu5, "=1 if rubbish disposal mainly by throwing in river,  creek or sea"
            elimbasu6, =1 if rubbish disposal mainly other
            epared1, =1 if walls are bad
            epared2, =1 if walls are regular
            epared3, =1 if walls are good
            etecho1, =1 if roof are bad
            etecho2, =1 if roof are regular
            etecho3, =1 if roof are good
            eviv1, =1 if floor are bad
            eviv2, =1 if floor are regular
            eviv3, =1 if floor are good
            dis, =1 if disable person
            male, =1 if male
            female, =1 if female
            estadocivil1, =1 if less than 10 years old
            estadocivil2, =1 if free or coupled uunion
            estadocivil3, =1 if married
            estadocivil4, =1 if divorced
            estadocivil5, =1 if separated
            estadocivil6, =1 if widow/er
            estadocivil7, =1 if single
            parentesco1, =1 if household head
            parentesco2, =1 if spouse/partner
            parentesco3, =1 if son/doughter
            parentesco4, =1 if stepson/doughter
            parentesco5, =1 if son/doughter in law
            parentesco6, =1 if grandson/doughter
            parentesco7, =1 if mother/father
            parentesco8, =1 if father/mother in law
            parentesco9, =1 if brother/sister
            parentesco10, =1 if brother/sister in law
            parentesco11, =1 if other family member
            parentesco12, =1 if other non family member
            idhogar, Household level identifier
            hogar_nin, Number of children 0 to 19 in household
            hogar_adul, Number of adults in household
            hogar_mayor, # of individuals 65+ in the household
            hogar_total, # of total individuals in the household
            dependency, Dependency rate, calculated = (number of members of the household younger than 19 or older than 64)/(number of member of household between 19 and 64)
            edjefe, years of education of male head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
            edjefa, years of education of female head of household, based on the interaction of escolari (years of education), head of household and gender, yes=1 and no=0
            meaneduc,average years of education for adults (18+)
            instlevel1, =1 no level of education
            instlevel2, =1 incomplete primary
            instlevel3, =1 complete primary
            instlevel4, =1 incomplete academic secondary level
            instlevel5, =1 complete academic secondary level
            instlevel6, =1 incomplete technical secondary level
            instlevel7, =1 complete technical secondary level
            instlevel8, =1 undergraduate and higher education
            instlevel9, =1 postgraduate higher education
            bedrooms, number of bedrooms
            overcrowding, # persons per room
            tipovivi1, =1 own and fully paid house
            tipovivi2, "=1 own,  paying in installments"
            tipovivi3, =1 rented
            tipovivi4, =1 precarious
            tipovivi5, "=1 other(assigned,  borrowed)"
            computer, =1 if the household has notebook or desktop computer
            television, =1 if the household has TV
            mobilephone, =1 if mobile phone
            qmobilephone, # of mobile phones
            lugar1, =1 region Central
            lugar2, =1 region Chorotega
            lugar3, =1 region PacÃƒÂ­fico central
            lugar4, =1 region Brunca
            lugar5, =1 region Huetar AtlÃƒÂ¡ntica
            lugar6, =1 region Huetar Norte
            area1, =1 zona urbana
            area2, =2 zona rural
            age, Age in years
            SQBescolari, escolari squared
            SQBage, age squared
            SQBhogar_total, hogar_total squared
            SQBedjefe, edjefe squared
            SQBhogar_nin, hogar_nin squared
            SQBovercrowding, overcrowding squared
            SQBdependency, dependency squared
            SQBmeaned, square of the mean years of education of adults (>=18) in the household
            agesq, Age squared

        Submission File

        For each Id in the test set, you must predict a class for the Target variable as described in the data.

        The train data is in the data/costa_rican_household_poverty_level_prediction folder and is named train.csv. 

        The test data is in the data/costa_rican_household_poverty_level_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/costa_rican_household_poverty_level_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "forest_cover_type_prediction": """

        Random forests? Cover trees? Not so fast, computer nerds. We're talking about the real thing.

        In this competition you are asked to predict the forest cover type (the predominant kind of tree cover) from cartographic variables. The actual forest cover type for a given 30 x 30 meter cell was determined from US Forest Service (USFS) Region 2 Resource Information System data. Independent variables were then derived from data obtained from the US Geological Survey and USFS. The data is in raw form and contains binary columns of data for qualitative independent variables such as wilderness areas and soil type.

        This study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. These areas represent forests with minimal human-caused disturbances, so that existing forest cover types are more a result of ecological processes rather than forest management practices.

        The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:

        0 - Spruce/Fir
        1 - Lodgepole Pine
        2 - Ponderosa Pine
        3 - Cottonwood/Willow
        4 - Aspen
        5 - Douglas-fir
        6 - Krummholz

        The training set contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set.

        Data Fields

        Elevation - Elevation in meters
        Aspect - Aspect in degrees azimuth
        Slope - Slope in degrees
        Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
        Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
        Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
        Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
        Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
        Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
        Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
        Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
        Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
        Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

        The wilderness areas are:

        1 - Rawah Wilderness Area
        2 - Neota Wilderness Area
        3 - Comanche Peak Wilderness Area
        4 - Cache la Poudre Wilderness Area

        The soil types are:

        1 Cathedral family - Rock outcrop complex, extremely stony.
        2 Vanet - Ratake families complex, very stony.
        3 Haploborolis - Rock outcrop complex, rubbly.
        4 Ratake family - Rock outcrop complex, rubbly.
        5 Vanet family - Rock outcrop complex complex, rubbly.
        6 Vanet - Wetmore families - Rock outcrop complex, stony.
        7 Gothic family.
        8 Supervisor - Limber families complex.
        9 Troutville family, very stony.
        10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
        11 Bullwark - Catamount families - Rock land complex, rubbly.
        12 Legault family - Rock land complex, stony.
        13 Catamount family - Rock land - Bullwark family complex, rubbly.
        14 Pachic Argiborolis - Aquolis complex.
        15 unspecified in the USFS Soil and ELU Survey.
        16 Cryaquolis - Cryoborolis complex.
        17 Gateview family - Cryaquolis complex.
        18 Rogert family, very stony.
        19 Typic Cryaquolis - Borohemists complex.
        20 Typic Cryaquepts - Typic Cryaquolls complex.
        21 Typic Cryaquolls - Leighcan family, till substratum complex.
        22 Leighcan family, till substratum, extremely bouldery.
        23 Leighcan family, till substratum - Typic Cryaquolls complex.
        24 Leighcan family, extremely stony.
        25 Leighcan family, warm, extremely stony.
        26 Granile - Catamount families complex, very stony.
        27 Leighcan family, warm - Rock outcrop complex, extremely stony.
        28 Leighcan family - Rock outcrop complex, extremely stony.
        29 Como - Legault families complex, extremely stony.
        30 Como family - Rock land - Legault family complex, extremely stony.
        31 Leighcan - Catamount families complex, extremely stony.
        32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
        33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
        34 Cryorthents - Rock land complex, extremely stony.
        35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
        36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
        37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
        38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
        39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
        40 Moran family - Cryorthents - Rock land complex, extremely stony.

        Submission File
        
        Your submission file should have the observation Id and a predicted cover type (an integer between 1 and 7, inclusive). 

        The train data is in the data/forest_cover_type_prediction folder and is named train.csv. 

        The test data is in the data/forest_cover_type_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/forest_cover_type_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "santander_customer_transaction_prediction": """
        At Santander our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.

        Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

        In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.
    
        You are provided with an anonymized dataset containing numeric feature variables, the binary target column, and a string ID_code column.

        The task is to predict the value of target column in the test set.

        Submission File
        
        For each Id in the test set, you must make a binary prediction of the target variable. 

        The train data is in the data/santander_customer_transaction_prediction folder and is named train.csv. 

        The test data is in the data/santander_customer_transaction_prediction folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/santander_customer_transaction_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "poisonous_mushroom_prediction": """
        Your Goal: The goal of this competition is to predict whether a mushroom is edible or poisonous based on its physical characteristics.

        The target column is 'class', which is either 'p' (poisonous) or 'e' (edible).

        your objective is to predict target class for each row

        Submission File

        For each id row in the test set, you must predict target class, whether the observation is editable (e) or poisonous (p). 

        The train data is in the data/poisonous_mushroom_prediction folder and is named train.csv. 

        The test data is in the data/poisonous_mushroom_prediction folder and is named test.csv. 
        
        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/poisonous_mushroom_prediction folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}.

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "bank_deposit_classification": """

        Your Goal: Your goal is to predict whether a client will subscribe to a bank term deposit.

        y is the binary target

        your objective is to predict the probability y for each row.

        Submission File

        For each id in the test set, you must predict the probability of the binary target y.

        The train data is in the data/bank_deposit_classification folder and is named train.csv. 

        The test data is in the data/bank_deposit_classification folder and is named test.csv. 
        
        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/bank_deposit_classification folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}.

        The task ends when you have successfully saved the predictions for the test data.
    """,
    "bank_churn_classification": """
        Your Goal: For this Episode of the Series, your task is to predict whether a customer continues with their account or closes it (e.g., churns). Good luck!

        Exited is the binary target

        your objective is to predict the probability of Exited

        Submission File

        For each id in the test set, you must predict the probability for the target variable Exited.

        The train data is in the data/bank_churn_classification folder and is named train.csv. 

        The test data is in the data/bank_churn_classification folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the model_saves/bank_churn_classification folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}.
        
        The task ends when you have successfully saved the predictions for the test data.
    """,
}