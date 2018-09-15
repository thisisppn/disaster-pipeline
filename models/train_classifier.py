import sys
import pandas as pd
import numpy as np
import nltk

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import pickle

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load the database as Pandas DataFrame
    :param database_filepath: Path to the database file.
    :return:
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset', con=engine)
    X = df.loc[:, 'message']
    Y = df.loc[:, 'genre']
    return X, Y, Y.unique()


def tokenize(text):
    """
    With reference to this tutorial https://www.nltk.org/book/ch03.html

    This function pre processed the text that comes in.
    :param text:
    :return:
    """
    tokens = word_tokenize(text)

    wnl = WordNetLemmatizer()
    
    # case normalise and lemmatize in the same loop.
    tokens = [wnl.lemmatize(t).lower().strip() for t in tokens]
    return tokens


def build_model(params={}):
    """
    Builds the model Pipeline.
    :param params:
    :return:
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('classifier', MultiOutputClassifier(RandomForestClassifier(**params)))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model's performance on the test set.
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, labels=category_names))


def save_model(model, model_filepath):
    """
    Saves the model into a Pickle file.
    :param model:
    :param model_filepath:
    :return:
    """
    pickle.dump(model, open(model_filepath, "wb"))


def get_best_params(model, X_train, Y_train):
    # print(pipeline.get_params().keys()) # Use this to find out which keys to add in the parameter

    parameters = {
        'classifier__estimator__n_estimators': [100, 150],
        'classifier__estimator__max_features': ['sqrt',],
        'classifier__estimator__criterion': ['gini', 'entropy']
    }

    cv = GridSearchCV(model, param_grid = parameters)
    cv.fit(X_train, Y_train)

    return cv.best_params_


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        Y_train = np.reshape(Y_train.values, (Y_train.shape[0], 1))
        Y_test = np.reshape(Y_test.values, (Y_test.shape[0], 1))

        print("Finding best parameters...")
        model = build_model()
        best_params = get_best_params(model, X_train, Y_train)

        rf_params = {
            'n_estimators': best_params['classifier__estimator__n_estimators'],
            'max_features': best_params['classifier__estimator__max_features'],
            'criterion': best_params['classifier__estimator__criterion'],
        }

        print("Best parameters are... ")
        print(rf_params)

        print('Building model with best parameters...')
        model = build_model(rf_params)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
