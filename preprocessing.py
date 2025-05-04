# data sources: https://pastebin.com/rFq663H7 and https://www.kaggle.com/datasets/rzgiza/pokdex-for-all-1025-pokemon-w-text-description?
 
import pandas as pd
import io
import streamlit as st
 
@st.cache_data
def load_votes_data() -> pd.DataFrame:
    # Read from a file instead of the string
    with open('data/pokemon_votes.txt', 'r') as file:
        data_string = file.read()
 
    # Convert the tab-separated data to a pandas DataFrame
    df = pd.read_csv(io.StringIO(data_string), sep='\t')
 
    # Display the first few rows of the DataFrame
    print(df.head())
 
    # Basic statistics
    print("\nBasic statistics for #1 votes:")
    print(df['Number of #1 Votes'].describe())
 
    print("\nBasic statistics for Top 6 votes:")
    print(df['Number of Top 6 Votes'].describe())
 
    # Find Pokémon with the most #1 votes
    top_first_votes = df.sort_values('Number of #1 Votes', ascending=False).head(10)
    print("\nTop 10 Pokémon by #1 votes:")
    print(top_first_votes[['Pokémon', 'Number of #1 Votes']])
 
    # Find Pokémon with the most Top 6 votes
    top_six_votes = df.sort_values('Number of Top 6 Votes', ascending=False).head(10)
    print("\nTop 10 Pokémon by Top 6 votes:")
    print(top_six_votes[['Pokémon', 'Number of Top 6 Votes']])
 
    return df
 
@st.cache_data
def load_pokedex_data() -> pd.DataFrame:
    df = pd.read_csv("data/pokedex.csv")
   
    # Remove curly braces and convert to list
    df["type"] = df["type"].str.replace(r"[{}]", "", regex=True).str.split(",") 
    return df
 
@st.cache_data
def preprocess_data() -> pd.DataFrame:
    pokedex_df = load_pokedex_data()
    votes_df = load_votes_data()
   
    # Normalize Pokemon names for matching
    pokedex_df['name_lower'] = pokedex_df['name'].str.lower()
    votes_df['Pokemon_lower'] = votes_df['Pokémon'].str.lower()
   
    # Perform a left join to keep all Pokemon from pokedex
    joint_df = pokedex_df.merge(
        votes_df,
        left_on='name_lower',
        right_on='Pokemon_lower',
        how='left'
    )
   
    # Fill NaN values for Pokemon without votes
    joint_df['Number of #1 Votes'] = joint_df['Number of #1 Votes'].fillna(0).astype(int)
    joint_df['Number of Top 6 Votes'] = joint_df['Number of Top 6 Votes'].fillna(0).astype(int)
   
    # Add popularity column based on vote counts
    joint_df['popularity'] = joint_df.apply(categorize_popularity, axis=1)
   
    # Drop the temporary columns used for joining
    joint_df = joint_df.drop(columns=['name_lower', 'Pokemon_lower'])
   
    return joint_df
 
def categorize_popularity(row):
    """
    Categorize Pokémon popularity based on vote counts:
    - "high": Either has ≥15 #1 votes OR ≥70 Top 6 votes
    - "medium": Either has ≥5 #1 votes OR ≥30 Top 6 votes (but not high)
    - "low": All others
   
    These thresholds were selected based on analysis of the vote distribution
    to create balanced categories.
    """
    votes_1 = row['Number of #1 Votes']
    votes_6 = row['Number of Top 6 Votes']
   
    if votes_1 >= 15 or votes_6 >= 70:
        return "high"
    elif votes_1 >= 5 or votes_6 >= 30:
        return "medium"
    else:
        return "low"