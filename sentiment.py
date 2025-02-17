import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
import time  # For timing
import traceback  # Import traceback module
from imblearn.over_sampling import SMOTE  # ADD THIS LINE

# Download VADER lexicon (only need to do this once)
nltk.download('vader_lexicon', quiet=True)  # Added quiet to stop printing out in loop

# Initialize sentiment analyzer (do this once)
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()


sia = load_sentiment_analyzer()


# Optimized Sentiment Analysis (vectorized)
def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    return score['compound']


# @st.cache_data  # Cache the training process
def train_model(df):
    start_time = time.time()

    # Data Preprocessing
    le = LabelEncoder()
    df['Department'] = le.fit_transform(df['Department'])

    df['Feedback Length'] = df['Feedback'].apply(len)
    # Create Interaction feature department score
    df['Sentiment_Department'] = df['Sentiment Score'] * df['Department']

    # Prepare data
    X = df.drop(columns=['Employee ID', 'Name', 'Feedback', 'Attrition'])
    y = df['Attrition']

    # Feature Selection
    selector = SelectKBest(score_func=f_classif, k=5)
    X_selected = selector.fit_transform(X, y)
    selected_features_indices = selector.get_support(indices=True)  # Get indices of selected features

    #This has been replaced to handle to use of numpy instead of df
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[selected_features_indices]  # Assuming X is a DataFrame
    else:
        selected_features = np.arange(X.shape[1])[selected_features_indices]  # Generate feature names
        X = pd.DataFrame(X, columns = df.drop(columns=['Employee ID', 'Name', 'Feedback', 'Attrition']).columns)
        selected_features = X.columns[selected_features_indices]


    X = X[selected_features]

    # Data Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    # Oversampling
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Model and Hyperparameter Tuning
    model = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [150, 250],
        'learning_rate': [0.005, 0.05],
        'max_depth': [2, 4]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")  # Can remove
    return best_model, scaler, le, selector, selected_features, accuracy, auc_roc, conf_matrix


def predict_attrition(df, model, scaler, le, selector, selected_features):
    df['Feedback Length'] = df['Feedback'].apply(len)
    # Vectorized sentiment analysis
    df['Sentiment Score'] = df['Feedback'].apply(analyze_sentiment)

    df['Department'] = le.transform(df['Department'])  # Use fitted LabelEncoder
    df['Sentiment_Department'] = df['Sentiment Score'] * df['Department']
    # Prepare data
    X = df.drop(columns=['Employee ID', 'Name', 'Feedback'])

    # Handle the X being passed if not in DF Format

    X = X[selected_features]


    X_scaled = scaler.transform(X)
    df['Predicted Attrition Risk'] = model.predict(X_scaled)
    return df


# Streamlit App
st.title("Employee Sentiment & Attrition Analysis")
st.write("Upload a CSV file with employee feedback.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        start_time = time.time()
        df = pd.read_csv(uploaded_file)  # Can be enhanced with dtypes

        # Check for required columns
        if 'Feedback' not in df.columns or 'Department' not in df.columns:
            st.error("The CSV must contain 'Feedback' and 'Department' columns.")
        else:
            with st.spinner("Training model..."):  # Show spinner during training
                best_model, scaler, le, selector, selected_features, accuracy, auc_roc, conf_matrix = train_model(
                    df.copy())  # Train model

            with st.spinner("Predicting attrition risk..."):
                df = predict_attrition(df.copy(), best_model, scaler, le, selector,
                                        selected_features)  # Predict attrition
            prediction_time = time.time() - start_time

            # Display Results
            st.write("### Analysis Results")
            st.dataframe(df[['Employee ID', 'Name', 'Feedback', 'Sentiment Score', 'Predicted Attrition Risk']])

            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.write(f"AUC-ROC: {auc_roc:.2f}")
            st.write("Confusion Matrix:")
            st.dataframe(pd.DataFrame(conf_matrix, columns=['Predicted 0', 'Predicted 1'], index=['Actual 0',
                                                                                                       'Actual 1']))

            high_risk = df[df['Predicted Attrition Risk'] == 1]
            st.write(f"High Attrition Risk Employees: {len(high_risk)}")
            if not high_risk.empty:
                st.write("**Recommended Engagement Strategies:**")
                st.write("- Conduct 1-on-1 meetings to understand concerns.")
                st.write("- Offer career development programs.")
                st.write("- Improve work-life balance policies.")
                st.write("- Increase recognition and incentives.")

            # Save the model (consider only doing this periodically)
            joblib.dump(best_model, 'attrition_model.pkl')
            joblib.dump(scaler, 'scaler.pkl')
            joblib.dump(le, 'label_encoder.pkl')
            st.success("Model saved successfully!")
            print(f"Process done in: {prediction_time:.2f} seconds")  # Removable

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.write("Traceback:")
        st.text(traceback.format_exc())
