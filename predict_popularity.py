import pandas as pd
import numpy as np
import joblib
import argparse
import os

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    return joblib.load(model_path)

def extract_types(X):
    """
    Extract type features from the 'type' column.
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
    
    Args:
        pokemon_data: DataFrame containing Pokémon features
    
    Returns:
        Processed DataFrame ready for prediction
    """
    # Ensure 'type' is in the correct format (list)
    if 'type' in pokemon_data.columns:
        pokemon_data['type'] = pokemon_data['type'].apply(
            lambda x: x.strip("{}").split(",") if isinstance(x, str) else x
        )
    
    # Extract type features
    type_features = extract_types(pokemon_data)
    
    # Drop the type column and combine with extracted features
    X = pokemon_data.drop('type', axis=1) if 'type' in pokemon_data.columns else pokemon_data
    X_combined = pd.concat([X.reset_index(drop=True), type_features.reset_index(drop=True)], axis=1)
    
    return X_combined

def predict_pokemon_popularity(model, pokedex_file, output_file=None):
    """
    Predict popularity for Pokémon in the given pokedex file.
    
    Args:
        model: Trained model
        pokedex_file: Path to CSV file with Pokémon data
        output_file: Optional path to save predictions
    
    Returns:
        DataFrame with Pokémon data and predicted popularity
    """
    # Load Pokémon data
    pokedex_df = pd.read_csv(pokedex_file)
    
    # Ensure we have the required columns
    required_columns = ['name', 'height', 'weight', 'hp', 'attack', 'defense', 
                        's_attack', 's_defense', 'speed', 'type']
    
    missing_columns = [col for col in required_columns if col not in pokedex_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Extract features
    X = pokedex_df[required_columns].copy()
    
    # Prepare data for prediction
    X_processed = prepare_pokemon_data(X)
    
    # Make predictions
    predictions = model.predict(X_processed)
    
    # Add predictions to the original data
    result_df = pokedex_df.copy()
    result_df['predicted_popularity'] = predictions
    
    # Save to file if specified
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    
    return result_df

def predict_single_pokemon(model, pokemon_features):
    """
    Predict popularity for a single Pokémon based on provided features.
    
    Args:
        model: Trained model
        pokemon_features: Dict with Pokémon features
    
    Returns:
        Predicted popularity category
    """
    # Convert to DataFrame
    pokemon_df = pd.DataFrame([pokemon_features])
    
    # Ensure types are in list format
    if 'type' in pokemon_df.columns:
        pokemon_df['type'] = [pokemon_features['type']]
    
    # Process data
    pokemon_processed = prepare_pokemon_data(pokemon_df)
    
    # Make prediction
    prediction = model.predict(pokemon_processed)[0]
    
    return prediction

def main():
    parser = argparse.ArgumentParser(description='Predict Pokémon popularity using trained model')
    parser.add_argument('--model', default='models/pokemon_rf_classifier.joblib',
                        help='Path to the trained model')
    parser.add_argument('--input', default='data/pokedex.csv',
                        help='Path to the input pokedex CSV file')
    parser.add_argument('--output', default='models/predicted_popularity.csv',
                        help='Path to save prediction results')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    # Make predictions
    print(f"Making predictions for Pokémon in {args.input}...")
    predictions_df = predict_pokemon_popularity(model, args.input, args.output)
    
    # Print summary
    print("\nPopularity Prediction Summary:")
    print(predictions_df['predicted_popularity'].value_counts())
    
    # Examples of high-popularity Pokémon
    high_pop = predictions_df[predictions_df['predicted_popularity'] == 'high']['name'].head(10).tolist()
    print(f"\nSample high-popularity Pokémon: {', '.join(high_pop)}")
    
    # Examples of low-popularity Pokémon
    low_pop = predictions_df[predictions_df['predicted_popularity'] == 'low']['name'].head(10).tolist()
    print(f"Sample low-popularity Pokémon: {', '.join(low_pop)}")

if __name__ == "__main__":
    main()