import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
data = pd.read_csv('/Users/bhoomikajethwani/Downloads/college_data.csv')

# Encode categorical variables (e.g., college names)
le_college = LabelEncoder()
data['college_name_encoded'] = le_college.fit_transform(data['college_name'])

# Automatically detect unique colleges from the dataset
unique_colleges = data['college_name'].unique()

# Define a function to filter data for a specific college and prepare it
def filter_and_prepare_data(college_of_interest, data, essay_weight=0.5):
    # Filter data for the selected college
    college_data = data[data['college_name'] == college_of_interest].copy()

    # Drop rows with missing values in important columns
    college_data.dropna(subset=['test_score_12th', 'sat_score', 'extracurriculars', 'essay', 'accepted', 'scholarship_amount'], inplace=True)

    # Map 'accepted' column to numerical weights: Yes = 1, No = 0, Waitlisted = 0.75
    acceptance_mapping = {'Yes': 1, 'No': 0, 'Waitlisted': 0.75}
    college_data['accepted_weighted'] = college_data['accepted'].map(acceptance_mapping)

    # Define basic features excluding extracurriculars and essays for now
    X_basic = college_data[['test_score_12th', 'sat_score', 'college_name_encoded']]
    y_acceptance = college_data['accepted_weighted'].astype(int)
    y_scholarship = college_data['scholarship_amount']

    # TF-IDF Vectorization for extracurriculars and essays
    tfidf_extra = TfidfVectorizer(max_features=50)
    extra_vectors = tfidf_extra.fit_transform(college_data['extracurriculars'])

    tfidf_essay = TfidfVectorizer(max_features=50)
    essay_vectors = tfidf_essay.fit_transform(college_data['essay'])

    # Convert to DataFrame
    extra_df = pd.DataFrame(extra_vectors.toarray(), columns=tfidf_extra.get_feature_names_out())
    essay_df = pd.DataFrame(essay_vectors.toarray(), columns=tfidf_essay.get_feature_names_out())

    # Apply weight to essay features
    essay_df = essay_df * essay_weight

    # Reset index of all DataFrames to ensure alignment
    college_data.reset_index(drop=True, inplace=True)
    X_basic.reset_index(drop=True, inplace=True)
    extra_df.reset_index(drop=True, inplace=True)
    essay_df.reset_index(drop=True, inplace=True)
    y_acceptance.reset_index(drop=True, inplace=True)
    y_scholarship.reset_index(drop=True, inplace=True)

    # Combine basic features, extracurriculars, and essay keywords
    X = pd.concat([X_basic, extra_df, essay_df], axis=1)

    # Replace NaNs with 0 if any
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Standardize the numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y_acceptance, y_scholarship, scaler, tfidf_extra, tfidf_essay

# Train models for each college and store them in dictionaries
classifiers = {}
regressors = {}
scalers = {}
tfidf_extras = {}
tfidf_essays = {}

essay_weight = 0.5  # Example weight to give less influence to essays

for college in unique_colleges:
    X, y_acceptance, y_scholarship, scaler, tfidf_extra, tfidf_essay = filter_and_prepare_data(college, data, essay_weight=essay_weight)
    
    # Random Forest Classifier for Acceptance Prediction
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_acceptance)
    classifiers[college] = clf
    
    # Random Forest Regressor for Scholarship Prediction
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X, y_scholarship)
    regressors[college] = reg
    
    # Store the scaler and vectorizers
    scalers[college] = scaler
    tfidf_extras[college] = tfidf_extra
    tfidf_essays[college] = tfidf_essay

# Function to make predictions for new data
def predict_for_new_data(test_score, sat_score, extracurriculars, essay_text, college_of_interest):
    # Use the appropriate model, scaler, and vectorizers for the specified college
    clf = classifiers[college_of_interest]
    reg = regressors[college_of_interest]
    scaler = scalers[college_of_interest]
    tfidf_extra = tfidf_extras[college_of_interest]
    tfidf_essay = tfidf_essays[college_of_interest]
    
    # Transform new input for the specific college
    new_data_basic = pd.DataFrame([[test_score, sat_score, le_college.transform([college_of_interest])[0]]], 
                                  columns=['test_score_12th', 'sat_score', 'college_name_encoded'])

    # Transform extracurriculars and essays using the existing vectorizers
    new_extra_keywords = pd.DataFrame(tfidf_extra.transform([extracurriculars]).toarray(), 
                                      columns=tfidf_extra.get_feature_names_out())
    new_essay_keywords = pd.DataFrame(tfidf_essay.transform([essay_text]).toarray(), 
                                      columns=tfidf_essay.get_feature_names_out())

    # Apply the same weight to the essay features
    new_essay_keywords = new_essay_keywords * essay_weight

    # Combine input features
    new_data = pd.concat([new_data_basic, new_extra_keywords, new_essay_keywords], axis=1)

    # Standardize new input
    new_data = scaler.transform(new_data)

    # Predict acceptance
    acceptance_prob = clf.predict_proba(new_data)[:, 1]

    # Predict scholarship
    scholarship_amount = reg.predict(new_data)

    print(f'Probability of Acceptance for {college_of_interest}: {acceptance_prob[0]:.2f}')
    print(f'Predicted Scholarship Amount for {college_of_interest}: ${scholarship_amount[0]:.2f}')

# Example input for prediction
test_score = 1000  # Example 12th-grade score
sat_score = 1600  # Example SAT score
extracurriculars = "Debate club, volunteering, football, debate, theater, robotics, community service"  # Example extracurricular activities
essay_text = "essay on challenges faced in life"  # Example essay
college_of_interest = 'Stanford'

# Make prediction
predict_for_new_data(test_score, sat_score, extracurriculars, essay_text, college_of_interest)

