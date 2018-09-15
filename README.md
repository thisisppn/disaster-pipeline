# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Files:
1. `process_data.py`
    - Load the CSV files.
    - Merge the messages & categories.
    - Process the categories to a format which is better suitable for processing.
    - Clean the data ( Remove Duplicates ).
    - Save the DataFrame into SQLite db.
    
2. `train_classifier.py`
    - Load and split the data from the SQLite DB into test and train sets.
    - The script uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in the machine learning pipeline to vectorize and then apply TF-IDF to the text. 
    - Use GridSearch to find the best parameters of a `RandomForestClassifier`.
    - Use the best parameters found above to train the model.
    - Measure & display the performance of the trained model on the test set. 
    - Save the model as a Pickle file. 
