import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast  # To parse the type column
from preprocessing import load_pokedex_data, load_votes_data, preprocess_data

# Load the combined dataset
joint_df = preprocess_data()

# ---- Sidebar Filters ----
st.sidebar.header("Filter Pokémon")
selected_types = st.sidebar.multiselect("Select Type:", 
                                        sorted(set(t for types in joint_df["type"] for t in types)))

# Filter Data by Type
filtered_df = joint_df[joint_df["type"].apply(lambda x: any(t in x for t in selected_types))] if selected_types else joint_df

# ---- Main Page ----
st.title("Pokédex Explorer")
st.write("Explore Pokémon stats and popularity interactively!")

# Display filtered data with votes
display_columns = ["name", "type", "hp", "attack", "defense", "s_attack", "s_defense", "speed", 
                   "Number of #1 Votes", "Number of Top 6 Votes"]
st.dataframe(filtered_df[display_columns])

# ---- Stats Visualization Tab ----
tab1, tab2, tab3 = st.tabs(["Stats Distribution", "Popularity Analysis", "Pokémon Details"])

with tab1:
    st.subheader("Pokémon Stats Distribution")
    stat = st.selectbox("Select a stat to visualize:", 
                      ["hp", "attack", "defense", "s_attack", "s_defense", "speed"])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(filtered_df[stat], bins=20, kde=True, ax=ax)
    plt.xlabel(stat.capitalize())
    plt.ylabel("Count")
    plt.title(f"Distribution of {stat.capitalize()} Stat")
    st.pyplot(fig)

# ---- Popularity Analysis Tab ----
with tab2:
    st.subheader("Pokémon Popularity Analysis")
    
    # Top 10 Most Popular Pokémon by #1 Votes
    st.write("### Top 10 Pokémon by #1 Votes")
    top_first_votes = filtered_df.sort_values('Number of #1 Votes', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Number of #1 Votes', y='name', data=top_first_votes, ax=ax)
    plt.title("Top 10 Pokémon by #1 Votes")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Top 10 Most Popular Pokémon by Top 6 Votes
    st.write("### Top 10 Pokémon by Top 6 Votes")
    top_six_votes = filtered_df.sort_values('Number of Top 6 Votes', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Number of Top 6 Votes', y='name', data=top_six_votes, ax=ax)
    plt.title("Top 10 Pokémon by Top 6 Votes")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Average votes by type
    st.write("### Popularity by Type")
    vote_type = st.radio("Choose vote type:", ["Number of #1 Votes", "Number of Top 6 Votes"])
    
    # Explode the type list to calculate averages by type
    type_votes = pd.DataFrame({
        'type': [t for types in filtered_df['type'] for t in types],
        'votes': [votes for types, votes in zip(filtered_df['type'], filtered_df[vote_type]) for _ in types]
    })
    
    type_avg_votes = type_votes.groupby('type')['votes'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=type_avg_votes.values, y=type_avg_votes.index, ax=ax)
    plt.title(f"Average {vote_type} by Pokémon Type")
    plt.tight_layout()
    st.pyplot(fig)

# ---- Pokémon Info Tab ----
with tab3:
    st.subheader("Pokémon Details")
    pokemon_name = st.text_input("Enter Pokémon Name:")
    
    if pokemon_name:
        pokemon_info = joint_df[joint_df["name"].str.contains(pokemon_name, case=False, na=False)]
        
        if not pokemon_info.empty:
            selected_pokemon = pokemon_info.iloc[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Basic Info")
                st.write(f"**Name:** {selected_pokemon['name']}")
                st.write(f"**Type:** {', '.join(selected_pokemon['type'])}")
                st.write(f"**Height:** {selected_pokemon['height']}")
                st.write(f"**Weight:** {selected_pokemon['weight']}")
                
                if 'info' in selected_pokemon:
                    st.write(f"**Description:** {selected_pokemon['info']}")
            
            with col2:
                st.write("### Stats")
                
                # Create a radar chart of the Pokémon's stats
                stats = ['hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
                stats_values = [selected_pokemon[stat] for stat in stats]
                stats_labels = ['HP', 'Attack', 'Defense', 'Sp. Attack', 'Sp. Defense', 'Speed']
                
                # Create angles for each stat
                angles = [n / len(stats) * 2 * 3.14159 for n in range(len(stats))]
                angles += angles[:1]  # Close the polygon
                
                # Values need to be closed as well
                stats_values += stats_values[:1]
                
                # Create plot
                fig = plt.figure(figsize=(6, 6))
                ax = fig.add_subplot(111, polar=True)
                
                # Draw the polygon and fill it
                ax.plot(angles, stats_values)
                ax.fill(angles, stats_values, alpha=0.25)
                
                # Set labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(stats_labels)
                
                plt.title(f"{selected_pokemon['name']} Stats")
                st.pyplot(fig)
                
                # Display popularity info
                st.write("### Popularity")
                st.write(f"**#1 Votes:** {int(selected_pokemon['Number of #1 Votes'])}")
                st.write(f"**Top 6 Votes:** {int(selected_pokemon['Number of Top 6 Votes'])}")
                
        else:
            st.write("No Pokémon found!")
    else:
        st.write("Enter a Pokémon name to see details.")