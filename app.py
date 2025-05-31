from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

app = Flask(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
df = None
model = None
scaler = None
model_trained = False
feature_columns = []
best_model_name = "N/A"
best_model_accuracy = 0.0

def load_data_and_train_model():
    global df, model, scaler, model_trained, feature_columns, best_model_name, best_model_accuracy

    try:
        df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully.")

        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        if 'diagnosis' in df.columns and df['diagnosis'].dtype == 'object':
            df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
            print("Diagnosis column mapped to numerical values.")

        if 'diagnosis' in df.columns:
            numerical_df = df.select_dtypes(include=np.number)
            
            if 'diagnosis' in numerical_df.columns:
                X = numerical_df.drop('diagnosis', axis=1)
                y = df['diagnosis']
            else:
                X = numerical_df
                y = df['diagnosis']

            feature_columns = X.columns.tolist()

            if X.isnull().sum().any():
                X = X.fillna(X.mean())
                print("Missing values in features filled with column mean.")

            if X.isnull().sum().sum() > 0:
                print("WARNING: NaNs still present in X after filling. Check data for all-NaN columns.")
                X = X.dropna(axis=1, how='all')
                feature_columns = X.columns.tolist()
                print(f"Dropped columns that were entirely NaN")

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            print("Features scaled.")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.01, random_state=42, stratify=y)

            models = {
                'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
                'Support Vector Machine': SVC(random_state=42, probability=True),
                'Random Forest': RandomForestClassifier(random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42)
            }

            best_accuracy = 0
            best_model = None
            best_model_name = "N/A"

            print("\nTraining and Evaluating Models")
            for name, current_model in models.items():
                print(f"Training {name} - ",end="")
                current_model.fit(X_train, y_train)
                y_pred = current_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"  {name} Accuracy: {accuracy:.4f}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = current_model
                    best_model_name = name

            model = best_model
            model_trained = True
            best_model_accuracy = best_accuracy
            print(f"\nBest Model Selected: {best_model_name} with Accuracy: {best_model_accuracy:.4f}")

        else:
            print("Error: 'diagnosis' column not found in data.csv. Cannot train model.")

    except FileNotFoundError:
        print(f"Error: data.csv not found at {DATA_PATH}. Please ensure the file exists.")
    except Exception as e:
        print(f"Error loading data or training model: {e}")
        model_trained = False

load_data_and_train_model()

def get_df_info(dataframe):
    if dataframe is None:
        return "Data not loaded. Please ensure 'data.csv' is in the 'data' directory."
    info_str = f"There are {dataframe.shape[0]} rows and {dataframe.shape[1]} columns\n"
    info_str += f"DataFrame shape: {dataframe.shape}\n\n"
    info_str += "Column types:\n"
    info_str += dataframe.dtypes.to_string()
    info_str += "\n\nFirst 5 rows:\n"
    info_str += dataframe.head().to_string()
    return info_str

def plot_to_base64(plt_figure):
    img = io.BytesIO()
    plt_figure.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(plt_figure)
    return base64.b64encode(img.getvalue()).decode('utf-8')

def plot_distribution(dataframe, column_name):
    if dataframe is None or column_name not in dataframe.columns:
        return None

    plt.figure(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(dataframe[column_name]):
        sns.histplot(dataframe[column_name], kde=True, bins=30, color='skyblue')
        plt.xlabel(column_name)
        plt.ylabel('Frequency')
    else:
        sns.countplot(y=dataframe[column_name], palette='viridis')
        plt.xlabel('Count')
        plt.ylabel(column_name)
    plt.title(f'Distribution of {column_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    return plot_to_base64(plt.gcf())

def plot_correlation_matrix(dataframe):
    if dataframe is None:
        return None

    numerical_df = dataframe.select_dtypes(include=['number'])
    if numerical_df.empty:
        return None

    plt.figure(figsize=(12, 10))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plot_to_base64(plt.gcf())

def plot_scatter_density(dataframe, col1, col2):
    if dataframe is None or col1 not in dataframe.columns or col2 not in dataframe.columns:
        return None
    if not pd.api.types.is_numeric_dtype(dataframe[col1]) or not pd.api.types.is_numeric_dtype(dataframe[col2]):
        return None

    plt.figure(figsize=(10, 8))
    sns.jointplot(x=col1, y=col2, data=dataframe, kind='kde', fill=True, cmap='Blues', height=7)
    plt.suptitle(f'Scatter Plot with Density for {col1} vs {col2}', y=1.02)
    plt.tight_layout()
    return plot_to_base64(plt.gcf())

@app.route('/')
def index():
    return render_template('index.html', data_loaded=(df is not None),
                           model_trained=model_trained,
                           best_model_name=best_model_name,
                           best_model_accuracy=f"{best_model_accuracy:.4f}")

@app.route('/dataframe_info')
def dataframe_info():
    info = get_df_info(df)
    return render_template('dataframe_info.html', info=info)

@app.route('/plot_column_distribution', methods=['GET', 'POST'])
def plot_column_distribution():
    plot_data = None
    selected_column = None
    column_names = df.columns.tolist() if df is not None else []

    if request.method == 'POST':
        selected_column = request.form.get('column_name')
        if selected_column and df is not None:
            plot_data = plot_distribution(df, selected_column)
        elif df is None:
            plot_data = None
            selected_column = "Error: Data not loaded."

    return render_template(
        'plot_distribution.html',
        plot_data=plot_data,
        column_names=column_names,
        selected_column=selected_column,
        data_loaded=(df is not None)
    )

@app.route('/plot_correlation')
def plot_correlation():
    plot_data = None
    if df is not None:
        plot_data = plot_correlation_matrix(df)
    return render_template(
        'plot_correlation.html',
        plot_data=plot_data,
        data_loaded=(df is not None)
    )

@app.route('/plot_scatter_density', methods=['GET', 'POST'])
def plot_scatter_density_route():
    plot_data = None
    selected_col1 = None
    selected_col2 = None
    numerical_column_names = df.select_dtypes(include=['number']).columns.tolist() if df is not None else []

    if request.method == 'POST':
        selected_col1 = request.form.get('column_name_1')
        selected_col2 = request.form.get('column_name_2')
        if selected_col1 and selected_col2 and df is not None:
            plot_data = plot_scatter_density(df, selected_col1, selected_col2)
        elif df is None:
            plot_data = None
            selected_col1 = "Error: Data not loaded."
            selected_col2 = ""

    return render_template(
        'plot_scatter_density.html',
        plot_data=plot_data,
        column_names=numerical_column_names,
        selected_col1=selected_col1,
        selected_col2=selected_col2,
        data_loaded=(df is not None)
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_result = None
    error_message = None

    if not model_trained:
        error_message = "Machine learning model not trained. Please ensure 'data.csv' is valid and present."
        return render_template('predict.html', prediction_result=prediction_result,
                               error_message=error_message, feature_columns=feature_columns,
                               model_trained=model_trained)

    if request.method == 'POST':
        user_input = {}
        for feature in feature_columns:
            try:
                user_input[feature] = float(request.form.get(feature))
            except (ValueError, TypeError):
                error_message = f"Invalid input for '{feature}'. Please enter a numerical value."
                return render_template('predict.html', prediction_result=prediction_result,
                                       error_message=error_message, feature_columns=feature_columns,
                                       model_trained=model_trained)

        input_df = pd.DataFrame([user_input], columns=feature_columns)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        if prediction == 1:
            prediction_result = f"Malignant (Probability: {prediction_proba[1]*100:.2f}%)"
        else:
            prediction_result = f"Benign (Probability: {prediction_proba[0]*100:.2f}%)"

    return render_template('predict.html', prediction_result=prediction_result,
                           error_message=error_message, feature_columns=feature_columns,
                           model_trained=model_trained)

if __name__ == '__main__':
    if not os.path.exists('data'):
        os.makedirs('data')
    app.run(debug=True)