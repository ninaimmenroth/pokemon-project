import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast  # To parse the type column
from preprocessing import load_pokedex_data, load_votes_data, preprocess_data
from model_integration import integrate_model_tab
 
# Load the combined dataset
joint_df = preprocess_data()
 
# ---- Sidebar with Pokemon logo ----
st.sidebar.header("Filter")
#st.sidebar.image("img/pokemon_logo.png", width=200)  # Add the local Pokemon logo image
selected_types = st.sidebar.multiselect("Select Type:",
                                        sorted(set(t for types in joint_df["type"] for t in types)))
 
# Add popularity filter
selected_popularity = st.sidebar.multiselect("Filter by Popularity:",
                                            ["high", "medium", "low"])
 
# Filter Data by Type and Popularity
filtered_df = joint_df.copy()
 
# Filter by type if types are selected
if selected_types:
    filtered_df = filtered_df[filtered_df["type"].apply(lambda x: any(t in x for t in selected_types))]
 
# Filter by popularity if popularity options are selected
if selected_popularity:
    filtered_df = filtered_df[filtered_df["popularity"].isin(selected_popularity)]
 
# ---- Main Page ----
st.title("Pokédex Explorer")
st.write("Explore Pokémon stats and popularity interactively!")
 
# Display filtered data with votes and popularity
display_columns = ["name", "type", "hp", "attack", "defense", "s_attack", "s_defense", "speed",
                  "Number of #1 Votes", "Number of Top 6 Votes", "popularity"]
st.dataframe(filtered_df[display_columns])
 
# ---- Stats Visualization Tab ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stats Distribution", "Popularity Analysis", "Popularity by Type", "Pokémon Details", "Prediction"])
 
with tab1:
    st.subheader("Pokémon Stats Distribution")
    stat = st.selectbox("Select a stat to visualize:",
                      ["hp", "attack", "defense", "s_attack", "s_defense", "speed"])
 
    # Add option to color by popularity
    color_by_popularity = st.checkbox("Color by popularity")
   
    fig, ax = plt.subplots(figsize=(8, 5))
    if color_by_popularity:
        # Use different colors for each popularity category
        for pop_cat, color in zip(["high", "medium", "low"], ["red", "orange", "blue"]):
            pop_data = filtered_df[filtered_df["popularity"] == pop_cat]
            sns.histplot(pop_data[stat], bins=20, kde=True, ax=ax, color=color, alpha=0.5, label=pop_cat)
        plt.legend()
    else:
        sns.histplot(filtered_df[stat], bins=20, kde=True, ax=ax)
   
    plt.xlabel(stat.capitalize())
    plt.ylabel("Count")
    plt.title(f"Distribution of {stat.capitalize()} Stat")
    st.pyplot(fig)
 
# ---- Popularity Analysis Tab ----
with tab2:
    st.subheader("Pokémon Popularity Analysis")
   
    # Distribution of popularity categories
    st.write("### Distribution of Popularity Categories")
    pop_counts = filtered_df["popularity"].value_counts().reset_index()
    pop_counts.columns = ["Popularity", "Count"]
   
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Popularity", y="Count", data=pop_counts, ax=ax, order=["high", "medium", "low"])
    plt.title("Number of Pokémon in Each Popularity Category")
    st.pyplot(fig)
   
    # Top 10 Most Popular Pokémon by #1 Votes
    st.write("### Top 10 Pokémon by #1 Votes")
    top_first_votes = filtered_df.sort_values('Number of #1 Votes', ascending=False).head(10)
   
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='Number of #1 Votes', y='name', data=top_first_votes, ax=ax)
   
    # Add popularity category color coding
    for i, bar in enumerate(bars.patches):
        pop_cat = top_first_votes.iloc[i]["popularity"]
        if pop_cat == "high":
            bar.set_facecolor("red")
        elif pop_cat == "medium":
            bar.set_facecolor("orange")
        else:
            bar.set_facecolor("blue")
   
    plt.title("Top 10 Pokémon by #1 Votes")
    plt.tight_layout()
    st.pyplot(fig)
   
    # Top 10 Most Popular Pokémon by Top 6 Votes
    st.write("### Top 10 Pokémon by Top 6 Votes")
    top_six_votes = filtered_df.sort_values('Number of Top 6 Votes', ascending=False).head(10)
   
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x='Number of Top 6 Votes', y='name', data=top_six_votes, ax=ax)
   
    # Add popularity category color coding
    for i, bar in enumerate(bars.patches):
        pop_cat = top_six_votes.iloc[i]["popularity"]
        if pop_cat == "high":
            bar.set_facecolor("red")
        elif pop_cat == "medium":
            bar.set_facecolor("orange")
        else:
            bar.set_facecolor("blue")
   
    plt.title("Top 10 Pokémon by Top 6 Votes")
    plt.tight_layout()
    st.pyplot(fig)
 
# ---- Popularity by Type Tab ----
with tab3:
    st.subheader("Popularity Analysis by Type")
   
    # Explode the type list to calculate popularity distribution by type
    type_pop_data = []
    for _, row in filtered_df.iterrows():
        for t in row['type']:
            type_pop_data.append({
                'type': t,
                'popularity': row['popularity'],
                'votes_1': row['Number of #1 Votes'],
                'votes_6': row['Number of Top 6 Votes']
            })
   
    type_pop_df = pd.DataFrame(type_pop_data)
   
    # Popularity distribution by type
    st.write("### Popularity Distribution by Type")
   
    # Count number of Pokémon of each popularity category per type
    type_pop_counts = type_pop_df.groupby(['type', 'popularity']).size().reset_index(name='count')
   
    # Get types sorted by total count
    type_totals = type_pop_df.groupby('type').size().sort_values(ascending=False)
    top_types = type_totals.head(15).index.tolist()
   
    # Filter to top 15 types
    type_pop_filtered = type_pop_counts[type_pop_counts['type'].isin(top_types)]
   
    # Pivot for stacked bar chart
    pivot_df = type_pop_filtered.pivot(index='type', columns='popularity', values='count').fillna(0)
   
    # Ensure all columns exist
    for col in ['high', 'medium', 'low']:
        if col not in pivot_df.columns:
            pivot_df[col] = 0
   
    # Sort by number of high popularity Pokémon
    pivot_df = pivot_df.sort_values('high', ascending=False)
   
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_df[['high', 'medium', 'low']].plot(kind='bar', stacked=True, ax=ax,
                                             color=['red', 'orange', 'blue'])
    plt.title("Distribution of Popularity Categories by Type")
    plt.xlabel("Type")
    plt.ylabel("Number of Pokémon")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Popularity")
    plt.tight_layout()
    st.pyplot(fig)
   
    # Average votes by type
    st.write("### Average Votes by Type")
    vote_type = st.radio("Choose vote type:", ["Number of #1 Votes", "Number of Top 6 Votes"])
   
    vote_col = 'votes_1' if vote_type == "Number of #1 Votes" else 'votes_6'
    type_avg_votes = type_pop_df.groupby('type')[vote_col].mean().sort_values(ascending=False)
   
    # Only show top 15 types
    type_avg_votes = type_avg_votes.head(15)
   
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=type_avg_votes.values, y=type_avg_votes.index, ax=ax)
    plt.title(f"Average {vote_type} by Pokémon Type")
    plt.tight_layout()
    st.pyplot(fig)
 
# ---- Pokémon Info Tab ----
with tab4:
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
                st.write(f"**Popularity:** {selected_pokemon['popularity'].upper()}")
               
                # Show vote counts with progress bars
                st.write("### Popularity Metrics")
                st.write(f"**#1 Votes:** {int(selected_pokemon['Number of #1 Votes'])}")
                max_votes_1 = joint_df['Number of #1 Votes'].max()
                st.progress(int(selected_pokemon['Number of #1 Votes']) / max_votes_1)
                
                st.write(f"**Top 6 Votes:** {int(selected_pokemon['Number of Top 6 Votes'])}")
                max_votes_6 = joint_df['Number of Top 6 Votes'].max()
                st.progress(int(selected_pokemon['Number of Top 6 Votes']) / max_votes_6)
               
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
               
                # Draw the polygon and fill it - color by popularity
                pop_color = {
                    'high': 'red',
                    'medium': 'orange',
                    'low': 'blue'
                }
                color = pop_color[selected_pokemon['popularity']]
               
                ax.plot(angles, stats_values, color=color)
                ax.fill(angles, stats_values, alpha=0.25, color=color)
               
                # Set labels
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(stats_labels)
               
                plt.title(f"{selected_pokemon['name']} Stats")
                st.pyplot(fig)
               
                # Compare to type averages
                st.write("### Stats vs. Type Average")
               
                # Get type averages for the Pokémon's types
                pokemon_types = selected_pokemon['type']
                type_filtered = joint_df[joint_df['type'].apply(lambda x: any(t in pokemon_types for t in x))]
                type_avgs = type_filtered[stats].mean()
               
                # Create comparison data
                comparison_data = pd.DataFrame({
                    'Stat': stats_labels,
                    'Value': [selected_pokemon[stat] for stat in stats],
                    'Type Average': [type_avgs[stat] for stat in stats]
                })
               
                fig, ax = plt.subplots(figsize=(8, 5))
                comparison_data.plot(kind='bar', x='Stat', y=['Value', 'Type Average'], ax=ax)
                plt.title(f"{selected_pokemon['name']} Stats vs. Type Average")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
               
        else:
            st.write("No Pokémon found!")
    else:
        st.write("Enter a Pokémon name to see details.")
with tab5:
    st.subheader("Prediction")
    integrate_model_tab()

# Add a footer with disclaimer
st.markdown("---")  # Horizontal line to separate content from footer
st.markdown("""
<div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-size: 12px; color: #6c757d; text-align: center; margin-top: 30px;">
    Pokedex Explorer by Nina Immenroth is not affiliated with "The Pokémon Company" and does not own or claim any rights to any Nintendo trademark or the Pokémon trademark, and all references to such are used for commentary and informational purposes only.
</div>
""", unsafe_allow_html=True)