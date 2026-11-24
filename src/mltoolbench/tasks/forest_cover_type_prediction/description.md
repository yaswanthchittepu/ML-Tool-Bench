

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

        The train data is in the {data_dir} folder and is named train.csv. 

        The test data is in the {data_dir} folder and is named test.csv. 

        You have to provide me the best possible model you can for the task, using the tools provided to you. 

        You need to load the data, clean the data, and perform fearture engineering first, before fitting a model, evaluating the model and saving the predictions on the test data.

        In order to perform consistent data manipulation for both train and test data, you are recommended to concatenate the two datasets first, perform the requisite operations, and then split them back into train and test.
        
        You might need to experiment with different models, hyperparameter tuning, and creative feature engineering to find the best model.

        Give me the evaluation resuls for the best model you can find, that show the goodness of the model fit to the data. Also save the best model you find to the {model_dir} folder.

        Most Importantly, save your predictions for the test data, as a csv, to the {save_directory} folder. The csv should have predictions for all the datapoints in the test set. Name the file as {save_file_name}. 

        The task ends when you have successfully saved the predictions for the test data.
    