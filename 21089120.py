# as usual, let us load all the necessary libraries
import numpy as np  # numerical computation with arrays
import pandas as pd # library to manipulate datasets using dataframes
import seaborn as sns

# plot 
import matplotlib.pyplot as plt
import random

#ignore wanring 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def read_csv_file(file_path):
    """
    Reads in a CSV file using pandas and returns the resulting DataFrame.
    
    Args:
    - file_path (str): The path to the CSV file to read in.
    
    Returns:
    - pandas.DataFrame: The resulting DataFrame.
    """
    df = pd.read_csv(file_path)
    return df
    
df_data = read_csv_file('african_crises.csv')

def analyze_data(df):
    """
    This function takes a DataFrame as input and performs various operations to analyze the data.

    Args:
    - df (pandas DataFrame): The DataFrame to be analyzed.

    Returns:
    - df (pandas DataFrame): The DataFrame after performing the analysis.
    """
    # Print the first few rows of the DataFrame
    print("First few rows of the DataFrame:")
    print(df.head())
    
    # Print summary statistics for the DataFrame
    print("----------------------------------------------------------------")
    print("\nSummary statistics for the DataFrame:")
    print(df.describe())
    
    # Print the mean, max, and min values for each column
    print("----------------------------------------------------------------")
    print("\nMean, max, and min values for each column:")
    stat = df.describe().T
    print(stat[['mean', 'max', 'min']])
    
    # Print information about the DataFrame, including data types and missing values
    print("----------------------------------------------------------------")
    print("\nInformation about the DataFrame:")
    print(df.info())
    print("----------------------------------------------------------------")
    print("\nNumber of missing values in each column:")
    print(df.isna().sum())
    
    # Convert the banking crisis values into categorical 0 and 1's
    df['banking_crisis'] = df['banking_crisis'].replace({'crisis': 1, 'no_crisis':0})
    
    # Drop unimportant covariates
    df.drop(columns=['case'], axis=1, inplace=True)
    
    # Drop ALL redundant covariates cc3 as it represents country
    df.drop(columns=['cc3'], axis=1, inplace=True)
    
    # Print the remaining columns after dropping some
    print("----------------------------------------------------------------")
    print("\nRemaining columns after dropping some:")
    print(df.columns)
    
    return df
    
analyze_data(df_data)


def get_unique_countries(df):
    """
    Extracts the unique countries from the input DataFrame.
    
    Args:
    df (pandas.DataFrame): The input DataFrame containing the country column.
    
    Returns:
    numpy.ndarray: A numpy array containing the unique country names.
    """
    # Extract the unique countries
    unique_countries = df['country'].unique()
    
    return unique_countries
    
unique_countries = get_unique_countries(df_data)
print(unique_countries) 


def plot_heatmap(df):
    """
    Creates a correlation matrix of the columns in the input DataFrame, and plots a heatmap of the correlations.

    Parameters:
    df (pandas.DataFrame): Input DataFrame

    Returns:
    None
    """
    # Create a correlation matrix
    corr_matrix = df.corr()
    
    # Create a heatmap of the correlations
    plt.figure(figsize=(15,10))
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Heatmap')
    #plt.show()
    plt.savefig('Correlation Heatmap.png', dpi=300, transparent=False)
    
# Call the function to plot the heatmap
plot_heatmap(df_data)


def plot_country_inflation_crises(df):
    """
    This function takes in a DataFrame and creates a countplot of the number of inflation crises by country.
    
    Args:
    df: Pandas DataFrame
    
    Returns:
    None
    """
    # Create a countplot of the number of inflation crises by country
    fig,ax = plt.subplots(figsize=(15,10))
    sns.countplot(df['country'],hue=df['inflation_crises'],ax=ax)
    plt.xlabel('Countries')
    plt.ylabel('Counts')
    plt.xticks(rotation=50)
    plt.title('Count of Inflation Crises by Country')
    #plt.show()
    plt.savefig('Count of Inflation Crises by Country.png', dpi=300,transparent=False)
    
# Call the function to plot the countplot
plot_country_inflation_crises(df_data)


def plot_exchange_rates(data):
    """
    Creates scatter and line plots to visualize the USD exchange rates per countries after and before independence.
    Also includes a countplot of inflation crises by country and a correlation heatmap.

    Args:
    df (pandas.DataFrame): DataFrame containing the data to be plotted.

    Returns:
    None
    """
    unique_countries = data['country'].unique()
    sns.set(style='whitegrid')
    plt.figure(figsize=(20,35))
    plt.suptitle('USD exchange rates per countries after and before independence', fontsize=20, y=1)
    plot_number=1

    # Create scatter and line plots for each country
    for country in unique_countries:
        plt.subplot(7,3,plot_number)
        plot_number+=1
        color ="#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

        # Scatter plot of exchange rates over time for each country
        plt.scatter(data[data.country==country]['year'],
                    data[data.country==country]['exch_usd'],
                    color=color,
                    s=20)

        # Line plot of exchange rates over time for each country
        sns.lineplot(data[data.country==country]['year'],
                     data[data.country==country]['exch_usd'],
                     label=country,
                     color=color)
        
        # Vertical line to indicate year of independence for each country
        plt.plot([np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),
                  np.min(data[np.logical_and(data.country==country,data.independence==1)]['year'])],
                 [0, np.max(data[data.country==country]['exch_usd'])],
                 color='black',
                 linestyle='dotted',
                 alpha=0.8)
        
        # Text label to indicate year of independence for each country
        plt.text(np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),
                 np.max(data[data.country==country]['exch_usd'])/2,
                 'Independence',
                 rotation=-90)
        
        # Point to indicate year of independence for each country
        plt.scatter(x=np.min(data[np.logical_and(data.country==country,data.independence==1)]['year']),
                    y=0,
                    s=50)
        # Set the title of the plot to the name of the country
        plt.title(country)
    print(plot_number)
    
    # Add a countplot of inflation crises by country
    plt.subplot(7,3,plot_number)
    sns.countplot(df_data['country'],hue=df_data['inflation_crises'])
    plt.xlabel('Countries')
    plt.ylabel('Counts')
    plt.xticks(rotation=50)
    plt.title('Count of Inflation Crises by Country')
    
    # Add a correlation heatmap
    plot_number+=1
    plt.subplot(7,3,plot_number)
    corr_matrix = df_data.corr()
    sns.heatmap(corr_matrix, annot=True)
    plt.title('Correlation Heatmap')
    
    # Save the figure to a file
    plt.tight_layout()
    
    plt.savefig('USD exchange rates per countries after and before independence.png', dpi=300, transparent=False)
    
    plt.show()
    
plot_exchange_rates(df_data)