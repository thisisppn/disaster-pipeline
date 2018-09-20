import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function loads the data from the CSV files and setups up the categories column as required.
    :param messages_filepath: path to the messages CSV file
    :param categories_filepath: path to the categories CSV file
    :return: dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the two dataframes on the common column 'id'
    df = messages.merge(categories, on='id', how='inner')
    categories = categories.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [cat_name[:-2] for cat_name in row]
    categories.columns = category_colnames
    
    for column in categories:

        # set each value to be the last character of the string
        # categories[column] = categories[column].str[-1]

        # After looking at the dataset, we see that one of the columns have three values 0, 1 and 2.
        # So, we need to remove the value 2, so that we only have 0 and 1
        categories[column] = categories[column].apply(lambda x: x[-1] if int(x[-1]) < 2 else 1)

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    df.drop('categories', inplace = True, axis = 1)
    df = pd.concat([df, categories], axis = 1)
    
    return df


def clean_data(df):
    """
    Remove the duplicate columns from the data
    :param df: The dataframe object to be cleaned
    :return: Cleaned dataframe object
    """
    # drop duplicates
    df = df[~df.duplicated()]
    return df


def save_data(df, database_filename):
    """
    Save the dataframe into a SQLite database.
    :param df: dataframe object to be saved in the db
    :param database_filename: path to the destination database file
    :return:
    """
    engine = create_engine('sqlite:///'+database_filename)

    # We need the chunksize=999 because that's the default limit, and it gives errors without it in my local system.
    df.to_sql('dataset', engine, if_exists='replace', index=False, chunksize=999)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
