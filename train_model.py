import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from preprocessing import preprocess_data
import ast  # For parsing the type field

def prepare_data(df):
    """
    Prepare data for model training by:
    1. Selecting relevant features
    2. Handling type lists (creating binary columns for type)
    3. Splitting into training and test sets (stratified)
    """
    # Extract features and target
    X = df[['height', 'weight', 'hp', 'attack', 'defense', 's_attack', 's_defense', 'speed', 'type']]
    y = df['popularity']
    
    # Create stratified train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Class distribution in training set: {y_train.value_counts().to_dict()}")
    print(f"Class distribution in test set: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Define, train and evaluate a Random Forest model with preprocessing pipeline.
    Includes handling for class imbalance. Excludes 'name' from feature set.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained model pipeline, extract_types function, and processed training data
    """
    # Create preprocessor for different feature types
    numeric_features = ['height', 'weight', 'hp', 'attack', 'defense', 's_attack', 's_defense', 'speed']
    
    # Special handling for the 'type' column which contains lists
    def extract_types(X):
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
    
    # Preprocessing for numerical features
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine preprocessing steps (only for numeric features now)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )
    
    # Create the pipeline with the preprocessor and random forest classifier
    # Use class_weight='balanced' to handle class imbalance
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            n_estimators=200,         # More trees for better performance
            max_samples=0.7,          # Use bootstrapping to further help with imbalance
            max_features=0.7,         # Use subset of features at each split to reduce overfitting
        ))
    ])
    
    # Define hyperparameters for grid search
    param_grid = {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2, 4],  # Require more samples per leaf to prevent overfitting
    }
    
    # Perform grid search with stratified k-fold to preserve class ratios
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1
    )
    
    # Extract type features and combine with other features
    types_train = extract_types(X_train)
    
    # Drop 'type' column from the training set
    X_train_without_type = X_train.drop(['type', 'name'], axis=1, errors='ignore')
    
    # Combine numerical features with type features
    X_train_combined = pd.concat([X_train_without_type.reset_index(drop=True), types_train.reset_index(drop=True)], axis=1)
    
    # Fit the model
    print("Training model with grid search...")
    print(f"Training with features: {X_train_combined.columns.tolist()}")
    grid_search.fit(X_train_combined, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_, extract_types, X_train_combined, X_train_without_type

def evaluate_model(model, extract_types_func, X_test, y_test):
    """
    Evaluate the model on test data using metrics appropriate for imbalanced classification.
    
    Args:
        model: Trained model
        extract_types_func: Function to extract type features
        X_test: Test features
        y_test: Test target
    """
    from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, confusion_matrix
    
    # Process test data the same way as training data
    types_test = extract_types_func(X_test)
    
    # Drop both 'name' and 'type' columns
    X_test_without_type = X_test.drop(['type', 'name'], axis=1, errors='ignore')
    X_test_combined = pd.concat([X_test_without_type.reset_index(drop=True), types_test.reset_index(drop=True)], axis=1)
    
    print(f"Evaluating with features: {X_test_combined.columns.tolist()}")
    
    # Make predictions
    y_pred = model.predict(X_test_combined)
    
    # Get prediction probabilities
    y_proba = model.predict_proba(X_test_combined)
    
    # Evaluate results
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    
    # Count predictions by class
    pred_counts = pd.Series(y_pred).value_counts()
    print("\nPredictions by class:")
    print(pred_counts)
    
    # Show confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return y_pred

def main():
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Load and prepare data with balanced classes
    print("Loading data with balanced popularity classes...")
    df = preprocess_data(use_balanced_classes=True)
    print(f"Total number of Pokémon: {len(df)}")
    
    # Get class distribution
    print("Popularity class distribution:")
    print(df['popularity'].value_counts())
    
    # Prepare data for model training
    print("\nPreparing data for model training...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Train model
    print("\nTraining model...")
    model, extract_types_func, X_train_combined, X_train_without_type = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    y_pred = evaluate_model(model, extract_types_func, X_test, y_test)
    
    # Save model
    print("\nSaving model...")
    joblib.dump(model, 'models/pokemon_rf_classifier.joblib')
    
    print("Model saved to models/pokemon_rf_classifier.joblib")
    
    # Create and save a feature importance DataFrame
    if hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
        # Get feature names from the preprocessor - now only numeric features
        preprocessor_features = list(model.named_steps['preprocessor'].get_feature_names_out())
        
        # Get type features
        type_features = [col for col in X_train_combined.columns if col not in X_train_without_type.columns]
        
        # Combine all feature names
        all_features = preprocessor_features + type_features
        
        # Get feature importances
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create a DataFrame of feature importances
        feature_importance = pd.DataFrame({
            'Feature': all_features[:len(importances)],
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Save feature importances
        feature_importance.to_csv('models/feature_importance.csv', index=False)
        print("Feature importances saved to models/feature_importance.csv")
        
        # Print top 10 most important features
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))

if __name__ == "__main__":
    main()