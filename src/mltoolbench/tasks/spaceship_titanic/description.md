
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

        The train data is in the {data_dir} folder and is named train.csv. 

        The test data is in the {data_dir} folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the {model_dir} folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all passengers in the test set. Name the file as {save_file_name}.

        The task ends when you have successfully saved the predictions for the test data.

        