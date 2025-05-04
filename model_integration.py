import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from preprocessing import load_pokedex_data, load_votes_data, preprocess_data

def load_model():
    """
    Load the trained model. Handles model not found scenarios gracefully.
    """
    try:
        return joblib.load('models/pokemon_rf_classifier.joblib')
    except FileNotFoundError:
        st.error("Model file not found. Please run train_model.py first to train the model.")
        return None

def extract_types(X):
    """
    Extract type features from the 'type' column.
    Similar to the function in train_model.py.
    """
    # Get unique Pokémon types
    all_types = set()
    for types in X['type']:
        if isinstance(types, list):
            all_types.update(types)
    
    # Create binary columns for each type
    type_matrix = np.zeros((len(X), len(all_types)))
    for i, types in enumerate(X['type']):
        if isinstance(types, list):
            for t in types:
                if t in all_types:
                    type_matrix[i, list(all_types).index(t)] = 1
    
    return pd.DataFrame(type_matrix, columns=list(all_types))

def prepare_pokemon_data(pokemon_data):
    """
    Prepare Pokémon data for prediction.
    """
    # Extract type features
    type_features = extract_types(pokemon_data)
    
    # Drop the type column and combine with extracted features
    X = pokemon_data.drop('type', axis=1) if 'type' in pokemon_data.columns else pokemon_data
    X_combined = pd.concat([X.reset_index(drop=True), type_features.reset_index(drop=True)], axis=1)
    
    return X_combined

def predict_pokemon_popularity(model, pokemon_data):
    """
    Make popularity predictions for given Pokémon data.
    
    Args:
        model: Trained model
        pokemon_data: DataFrame with Pokemon features
    
    Returns:
        DataFrame with added predicted_popularity column
    """
    if model is None:
        # Return the original data if model is not loaded
        return pokemon_data
    
    # Extract features for prediction
    X = pokemon_data[['name', 'height', 'weight', 'hp', 'attack', 'defense', 
                     's_attack', 's_defense', 'speed', 'type']].copy()
    
    # Prepare data for prediction
    X_processed = prepare_pokemon_data(X)
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    # Add predictions to the original data
    result_df = pokemon_data.copy()
    result_df['predicted_popularity'] = predictions
    
    return result_df

def evaluate_model_performance(df):
    """
    Evaluate model performance by comparing predicted_popularity with actual popularity.
    
    Args:
        df: DataFrame with actual and predicted popularity
    
    Returns:
        Accuracy score and confusion matrix
    """
    actual = df['popularity']
    predicted = df['predicted_popularity']
    
    # Calculate accuracy
    accuracy = sum(actual == predicted) / len(actual)
    
    # Create confusion matrix
    cm = confusion_matrix(actual, predicted, labels=['high', 'medium', 'low'])
    
    return accuracy, cm

def add_model_tab(joint_df):
    """
    Add a Model Analysis tab to the Streamlit app.
    This function should be called from app.py.
    
    Args:
        joint_df: Combined dataframe with Pokemon and votes data
    """
    st.header("Random Forest Model Analysis")
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.warning("Model not loaded. Some features will be unavailable.")
        return
    
    # Make predictions
    prediction_df = predict_pokemon_popularity(model, joint_df)
    
    # Add a dropdown to compare actual vs predicted
    comparison_option = st.selectbox(
        "Select analysis type:",
        ["Model Accuracy", "Feature Importance", "Misclassified Pokémon", "Predict Custom Pokémon"]
    )
    
    if comparison_option == "Model Accuracy":
        # Calculate and display accuracy
        accuracy, cm = evaluate_model_performance(prediction_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Accuracy", f"{accuracy:.2%}")
            
            # Distribution of actual vs predicted
            st.write("### Distribution of Popularity Classes")
            
            # Create a comparative bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get counts
            actual_counts = prediction_df['popularity'].value_counts().sort_index()
            predicted_counts = prediction_df['predicted_popularity'].value_counts().sort_index()
            
            # Plot bars
            x = np.arange(len(actual_counts.index))
            width = 0.35
            
            ax.bar(x - width/2, actual_counts.values, width, label='Actual')
            ax.bar(x + width/2, predicted_counts.values, width, label='Predicted')
            
            # Add labels and legend
            ax.set_xlabel('Popularity')
            ax.set_ylabel('Count')
            ax.set_title('Actual vs Predicted Popularity Distribution')
            ax.set_xticks(x)
            ax.set_xticklabels(actual_counts.index)
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            st.write("### Confusion Matrix")
            
            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['high', 'medium', 'low'],
                        yticklabels=['high', 'medium', 'low'],
                        ax=ax)
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            st.pyplot(fig)
            
            # Add explanation
            st.write("""
            **Interpreting the Confusion Matrix:**
            - The diagonal represents correctly classified Pokémon
            - Off-diagonal elements are misclassifications
            - Rows represent the actual popularity class
            - Columns represent the predicted popularity class
            """)
    
    elif comparison_option == "Feature Importance":
        try:
            # Load feature importance if available
            feature_importance = pd.read_csv('models/feature_importance.csv')
            
            st.write("### Feature Importance")
            st.write("These are the most influential features in determining Pokémon popularity:")
            
            # Plot top 15 features
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = feature_importance.head(15)
            
            # Create horizontal bar chart
            sns.barplot(x='Importance', y='Feature', data=top_features, ax=ax)
            plt.title('Top 15 Most Important Features')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add explanation about feature importance
            st.write("""
            **Understanding Feature Importance:**
            - Higher values indicate features that have a stronger influence on the model's predictions
            - The model considers these features most significant when determining a Pokémon's popularity
            - Type features (prefixed with 'type_') show the importance of specific Pokémon types
            """)
            
        except FileNotFoundError:
            st.error("Feature importance file not found. Please run train_model.py to generate it.")
    
    elif comparison_option == "Misclassified Pokémon":
        # Find misclassified Pokémon
        misclassified = prediction_df[prediction_df['popularity'] != prediction_df['predicted_popularity']]
        
        st.write(f"### Misclassified Pokémon ({len(misclassified)} total)")
        
        # Add filter for misclassification type
        misclass_filter = st.multiselect(
            "Filter by actual vs predicted:",
            ["high → medium", "high → low", "medium → high", "medium → low", "low → high", "low → medium"],
            default=["high → low", "low → high"]  # Most severe misclassifications by default
        )
        
        filtered_misclass = misclassified.copy()
        if misclass_filter:
            # Apply filters
            filter_conditions = []
            for filter_option in misclass_filter:
                actual, predicted = filter_option.split(" → ")
                filter_conditions.append(
                    (filtered_misclass['popularity'] == actual) & 
                    (filtered_misclass['predicted_popularity'] == predicted)
                )
            
            # Combine filters with OR
            filtered_misclass = filtered_misclass[np.logical_or.reduce(filter_conditions)]
        
        # Display misclassified Pokémon
        if len(filtered_misclass) > 0:
            # Add columns for display
            display_columns = ['name', 'type', 'hp', 'attack', 'defense', 's_attack', 's_defense', 'speed',
                              'Number of #1 Votes', 'Number of Top 6 Votes', 'popularity', 'predicted_popularity']
            
            st.dataframe(filtered_misclass[display_columns])
            
            # Show some analysis of misclassifications
            st.write("### Analysis of Misclassifications")
            
            # Compare average stats between correctly and incorrectly classified Pokémon
            stat_cols = ['hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
            
            correct = prediction_df[prediction_df['popularity'] == prediction_df['predicted_popularity']]
            
            st.write("Average stats for correctly vs incorrectly classified Pokémon:")
            
            # Calculate averages
            correct_avg = correct[stat_cols].mean()
            incorrect_avg = misclassified[stat_cols].mean()
            
            # Create comparison dataframe
            comparison_df = pd.DataFrame({
                'Correctly Classified': correct_avg,
                'Misclassified': incorrect_avg
            })
            
            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            comparison_df.plot(kind='bar', ax=ax)
            plt.title('Average Stats: Correctly vs Incorrectly Classified')
            plt.ylabel('Stat Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No Pokémon match the selected filters.")
    
    elif comparison_option == "Predict Custom Pokémon":
        st.write("### Predict Popularity for a Custom Pokémon")
        st.write("Enter the stats for a new Pokémon to predict its popularity:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name", "CustomMon")
            height = st.number_input("Height", min_value=1, max_value=100, value=10)
            weight = st.number_input("Weight", min_value=1, max_value=1000, value=100)
            hp = st.number_input("HP", min_value=1, max_value=255, value=50)
            attack = st.number_input("Attack", min_value=1, max_value=255, value=50)
        
        with col2:
            defense = st.number_input("Defense", min_value=1, max_value=255, value=50)
            s_attack = st.number_input("Special Attack", min_value=1, max_value=255, value=50)
            s_defense = st.number_input("Special Defense", min_value=1, max_value=255, value=50)
            speed = st.number_input("Speed", min_value=1, max_value=255, value=50)
            
            # Get all possible types from the dataset
            all_types = set()
            for types_list in joint_df['type']:
                all_types.update(types_list)
            
            selected_types = st.multiselect("Types", sorted(all_types), ["normal"])
        
        if st.button("Predict Popularity"):
            # Create a DataFrame for the custom Pokémon
            custom_pokemon = pd.DataFrame([{
                'name': name,
                'height': height,
                'weight': weight,
                'hp': hp,
                'attack': attack,
                'defense': defense,
                's_attack': s_attack,
                's_defense': s_defense,
                'speed': speed,
                'type': selected_types
            }])
            
            # Prepare data and make prediction
            custom_processed = prepare_pokemon_data(custom_pokemon)
            prediction = model.predict(custom_processed)[0]
            
            # Display prediction with styling
            st.write(f"### Prediction Result for {name}")
            
            # Create colored box based on prediction
            if prediction == 'high':
                st.success(f"Predicted Popularity: HIGH")
            elif prediction == 'medium':
                st.info(f"Predicted Popularity: MEDIUM")
            else:
                st.warning(f"Predicted Popularity: LOW")
            
            # Find similar Pokémon for comparison
            st.write("### Similar Pokémon for Comparison")
            
            # Calculate Euclidean distance for stats
            stat_cols = ['hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
            custom_stats = custom_pokemon[stat_cols].values[0]
            
            # Calculate distances
            joint_df['distance'] = joint_df.apply(
                lambda row: np.sqrt(sum((row[stat_cols].values - custom_stats) ** 2)),
                axis=1
            )
            
            # Get 5 most similar Pokémon
            similar_pokemon = joint_df.sort_values('distance').head(5)
            
            # Display similar Pokémon
            for _, pokemon in similar_pokemon.iterrows():
                st.write(f"**{pokemon['name'].title()}** - "
                         f"Popularity: {pokemon['popularity'].upper()} - "
                         f"Types: {', '.join(pokemon['type'])}")
                
                # Create two columns for stats
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"HP: {pokemon['hp']}")
                    st.write(f"Attack: {pokemon['attack']}")
                    st.write(f"Defense: {pokemon['defense']}")
                with col2:
                    st.write(f"Sp. Attack: {pokemon['s_attack']}")
                    st.write(f"Sp. Defense: {pokemon['s_defense']}")
                    st.write(f"Speed: {pokemon['speed']}")
                
                st.write("---")

# This function can be imported and called from app.py
def integrate_model_tab():
    # Load the data
    joint_df = preprocess_data()
    
    # Add the model tab
    add_model_tab(joint_df)