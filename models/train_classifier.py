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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('dataset', con=engine)
    X = df.loc[:, 'message']
    Y = df.loc[:, 'genre']
    return X, Y, Y.unique()


def tokenize(text):
    """
    With reference to this tutorial https://www.nltk.org/book/ch03.html
    """
    tokens = word_tokenize(text)
    
    # case normalise
#     tokens = [word.lower() for word in tokens]
    
    wnl = WordNetLemmatizer()
    
    # case normalise and lemmatize in the same loop.
    tokens = [wnl.lemmatize(t).lower().strip() for t in tokens]
    return tokens

def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=150, criterion='entropy', max_features='sqrt')))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, labels=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        Y_train = np.reshape(Y_train.values, (Y_train.shape[0], 1))
        Y_test = np.reshape(Y_test.values, (Y_test.shape[0], 1))
        
        print('Building model...')
        model = build_model()
        
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
