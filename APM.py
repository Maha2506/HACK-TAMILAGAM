import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.decomposition import TruncatedSVD


# -------------------------- Health Condition Prediction, Personalized Recommendations, Emotion Classification --------------------------

# Function to load and process health condition dataset
def load_and_process_health_data(file_path):
    df = pd.read_csv(file_path)
    df['condition'] = np.where(df['hydration_level'] < 30, 'Dehydration',
                               np.where(df['glucose_level'] > 150, 'Fatigue',
                                        np.where(df['temperature'] > 38, 'Cold',
                                                 np.where(df['stress_level'] > 7, 'Stress',
                                                          np.where(df['infection_biomarker'] == 1, 'Infection',
                                                                   np.where(df['fatigue_level'] > 6, 'Headache',
                                                                            'Healthy'))))))  # Corrected unmatched parentheses

    label_encoder = LabelEncoder()
    df['condition'] = label_encoder.fit_transform(df['condition'])

    X = df.drop(columns=['condition'])
    y = df['condition']
    return df, X, y, label_encoder


# Emotion Classification Web App
def emotion_classification_app():
    # Set a custom heading with a smaller font size
    st.markdown("<h2 style='font-size: 24px;'>Emotion Classification Web App</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())

        if 'emotion' not in df.columns:
            st.error("Dataset must contain an 'emotion' column.")
        else:
            X_data = df.drop(columns=['emotion']).values
            y_data = df['emotion'].values
            label_encoder = LabelEncoder()
            y_data_encoded = label_encoder.fit_transform(y_data)

            X_data = X_data / np.max(X_data, axis=1, keepdims=True)

            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data_encoded, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_probs = model.predict_proba(X_test)

            st.subheader('Precision-Recall Curve for Each Class (One-vs-Rest)')
            fig, ax = plt.subplots(figsize=(8, 6))
            for i in range(len(label_encoder.classes_)):
                y_true_binary = (y_test == i).astype(int)
                y_pred_probs_class = y_pred_probs[:, i]
                precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_probs_class)
                ax.plot(recall, precision, label=f'Class {label_encoder.classes_[i]}')

            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curves for Each Class (One-vs-Rest)')
            ax.legend()
            ax.grid()
            st.pyplot(fig)

            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp.plot(cmap='Blues', ax=ax)
            st.pyplot(fig)


# -------------------------- Anomaly Detection --------------------------

# Function to load and preprocess sensor data
def load_data():
    uploaded_file = st.file_uploader("Upload Sensor Data CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview", df.head())
        return df
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None


# Anomaly Detection using Isolation Forest
def detect_anomalies_isolation_forest(df_scaled):
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    anomalies = iso_forest.fit_predict(df_scaled)
    return anomalies


# Anomaly Detection using KMeans Clustering
def detect_anomalies_kmeans(df_scaled):
    kmeans = KMeans(n_clusters=2, random_state=42)
    anomalies = kmeans.fit_predict(df_scaled)
    return anomalies


# Visualizations for anomaly detection
def plot_visualizations(df):
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    sns.scatterplot(x=df['heart_rate'], y=df['temperature'], hue=df['iso_forest_anomaly'], palette='coolwarm',
                    ax=axs[0, 0])
    axs[0, 0].set_title('Anomaly Detection (Isolation Forest)')
    axs[0, 0].set_xlabel('Heart Rate')
    axs[0, 0].set_ylabel('Temperature')

    sns.scatterplot(x=df['heart_rate'], y=df['temperature'], hue=df['kmeans_anomaly'], palette='coolwarm', ax=axs[0, 1])
    axs[0, 1].set_title('Anomaly Detection (KMeans)')
    axs[0, 1].set_xlabel('Heart Rate')
    axs[0, 1].set_ylabel('Temperature')

    sns.histplot(df['heart_rate'], kde=True, bins=30, color='blue', ax=axs[1, 0])
    axs[1, 0].set_title('Distribution of Heart Rate')
    axs[1, 0].set_xlabel('Heart Rate')
    axs[1, 0].set_ylabel('Frequency')

    sns.histplot(df['temperature'], kde=True, bins=30, color='red', ax=axs[1, 1])
    axs[1, 1].set_title('Distribution of Temperature')
    axs[1, 0].set_xlabel('Temperature')
    axs[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    st.pyplot(fig)


# Preprocessing function for anomaly detection
def preprocess_data(df):
    # Example: normalizing the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler


# Personalized Recommendations Using Collaborative Filtering (SVD-based)
def personalized_recommendations():
    st.subheader("Personalized Recommendations")
    uploaded_file = st.file_uploader("Upload User-Item Interaction Data (CSV)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(df.head())

        # Assuming the dataset has columns: 'user_id', 'item_id', 'rating'
        if not all(col in df.columns for col in ['user_id', 'item_id', 'rating']):
            st.error("Dataset must contain 'user_id', 'item_id', and 'rating' columns.")
        else:
            # Pivot the dataset to create a user-item matrix
            user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating')
            user_item_matrix = user_item_matrix.fillna(0)

            # Apply SVD for collaborative filtering
            svd = TruncatedSVD(n_components=5, random_state=42)
            matrix_svd = svd.fit_transform(user_item_matrix)

            # Reconstruct the ratings matrix
            reconstructed_matrix = np.dot(matrix_svd, svd.components_)

            # Get the top 10 recommendations for a specific user (example user_id = 1)
            user_id = 1  # Modify this for the user you want recommendations for
            user_ratings = reconstructed_matrix[user_id - 1]
            top_recommendations = user_ratings.argsort()[-10:][::-1]  # Get top 10 items

            st.write(f"Top 10 recommendations for user {user_id}:")
            st.write(top_recommendations)


# About Us Section
def about_us():
    st.markdown("# About Us")
    st.write("""
        Welcome to our platform, where we integrate machine learning and data analytics to improve health outcomes 
        and provide personalized experiences. We specialize in:

        - Predicting health conditions based on various biometric data
        - Providing personalized recommendations using collaborative filtering
        - Classifying emotions based on user data
        - Detecting anomalies in sensor data for early warning systems

        Our mission is to empower individuals and organizations to make informed decisions based on actionable insights.
    """)


# Main function to combine tasks
def main():
    st.title("Welcome to Automated Medical Vendor Machine")  # Updated title

    task = st.sidebar.selectbox("Select Task", ["Health Conditions Prediction", "Personalized Recommendations",
                                                "Emotion Classification", "Anomaly Detection", "About Us"])

    if task == "Health Conditions Prediction":
        st.subheader("Health Condition Prediction")
        uploaded_file = st.file_uploader("Upload Health Data CSV", type="csv")

        if uploaded_file is not None:
            df, X, y, label_encoder = load_and_process_health_data(uploaded_file)
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            st.subheader("Predict Health Condition")
            hydration_level = st.number_input("Hydration Level", min_value=0, max_value=100, value=50)
            glucose_level = st.number_input("Glucose Level", min_value=50, max_value=300, value=100)
            temperature = st.number_input("Temperature", min_value=35.0, max_value=42.0, value=37.0)
            stress_level = st.number_input("Stress Level", min_value=0, max_value=10, value=5)
            infection_biomarker = st.selectbox("Infection Biomarker (0 = No, 1 = Yes)", [0, 1])
            fatigue_level = st.number_input("Fatigue Level", min_value=0, max_value=10, value=3)

            input_data = np.array([[hydration_level, glucose_level, temperature, stress_level, infection_biomarker,
                                    fatigue_level]])
            input_data = StandardScaler().fit_transform(input_data)
            prediction = model.predict(input_data)
            condition = label_encoder.inverse_transform(prediction)[0]

            st.write(f"Predicted Health Condition: {condition}")

    elif task == "Personalized Recommendations":
        personalized_recommendations()

    elif task == "Emotion Classification":
        emotion_classification_app()

    elif task == "Anomaly Detection":
        st.subheader("Anomaly Detection in Sensor Data")
        df = load_data()

        if df is not None:
            df_scaled, scaler = preprocess_data(df)
            df['iso_forest_anomaly'] = detect_anomalies_isolation_forest(df_scaled)
            df['kmeans_anomaly'] = detect_anomalies_kmeans(df_scaled)
            plot_visualizations(df)

    elif task == "About Us":
        about_us()


if __name__ == "__main__":
    main()
