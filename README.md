# Pokémon Popularity Prediction
 
This project builds upon the "pokemon-project" to create an interactive web application that allows users to explore and analyze data about Pokémon, with a focus on predicting popularity.
 
## Overview
 
The application includes a machine learning model that predicts Pokémon popularity based on their stats, types, and other features. The prediction is categorized into three levels:
 
- **High**: Pokémon with ≥15 #1 votes OR ≥70 Top 6 votes
- **Medium**: Pokémon with ≥5 #1 votes OR ≥30 Top 6 votes (but not high)
- **Low**: All others
 
## Features
 
The application includes:
 
- **Data Exploration**: Analyze Pokémon stats distributions and popularity metrics
- **Type Analysis**: Explore how different Pokémon types relate to popularity
- **Individual Pokémon Details**: Look up specific Pokémon and view their stats
- **Machine Learning Model**: Train and use a Random Forest classifier to predict popularity
- **Custom Prediction**: Create custom Pokémon and predict their popularity
 
## Files
 
The project consists of the following files:
 
- `app.py`: Main Streamlit application
- `preprocessing.py`: Data loading and preprocessing functions
- `train_model.py`: Trains the Random Forest model on Pokémon features
- `predict_popularity.py`: Script to make predictions on new data
- `model_integration.py`: Integrates the model into the Streamlit app
 
## Model Implementation
 
The prediction model uses a Random Forest classifier with the following features:
- height
- weight
- hp
- attack
- defense
- s_attack
- s_defense
- speed
- type (converted to one-hot encoded features)
 
### Training Process
 
The model is trained using the following process:
 
1. **Data Loading**: Combines the Pokédex data with popularity votes
2. **Feature Engineering**: Converts Pokémon types into binary features
3. **Stratified Sampling**: Uses 80% of the data for training while maintaining class balance
4. **Hyperparameter Tuning**: Uses GridSearchCV to find optimal parameters
5. **Model Evaluation**: Assesses performance using accuracy and classification report
6. **Feature Importance**: Calculates and saves feature importance scores
 
### Model Usage
 
The model can be used in two ways:
 
1. **Integrated in the Streamlit App**: A dedicated tab for model analysis and prediction
2. **Standalone Script**: Use `predict_popularity.py` to make predictions on new data
 
## How to Use
 
### Setting Up
 
1. Clone the repository
2. Install the required packages: `pip install -r requirements.txt`
3. Create the necessary directories: `mkdir -p data models`
4. Place the required data files in the `data` directory
 
### Training the Model
 
Train the model using:
 
```bash
python train_model.py
```
 
This will create the model file at `models/pokemon_rf_classifier.joblib` and save feature importance information.
 
### Running the Application
 
Start the Streamlit application with:
 
```bash
streamlit run app.py
```
 
### Making Predictions
 
To make predictions on the entire Pokédex:
 
```bash
python predict_popularity.py --input data/pokedex.csv --output models/predicted_popularity.csv
```
 
## Model Performance
 
The Random Forest model typically achieves around 70-75% accuracy on the test set. Performance metrics include:
 
- **Accuracy**: Percentage of correctly classified Pokémon
- **Confusion Matrix**: Shows the distribution of correct and incorrect predictions
- **Feature Importance**: Indicates which features are most predictive of popularity
 
## Future Improvements
 
Potential enhancements to the model:
 
- Include evolutionary stage information
- Add generation/release date features
- Incorporate design features (colors, shapes)
- Experiment with more complex models (XGBoost, Neural Networks)