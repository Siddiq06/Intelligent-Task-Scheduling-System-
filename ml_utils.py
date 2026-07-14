import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime
from crngrim_classifier import CRNGRIMClassifier

MODELS_DIR = 'models'
DATA_PATH = 'Dataset/cloud_workload_dataset.csv'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)



# ==========================================
# Helper: Plot to Base64
# ==========================================
def plot_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img

# ==========================================
# Load & Preprocess
# ==========================================
def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)
    df['Task_Start_Time'] = pd.to_datetime(df['Task_Start_Time'], errors='coerce')
    df['Task_End_Time'] = pd.to_datetime(df['Task_End_Time'], errors='coerce')
    le = LabelEncoder()
    df['Data_Source_encoded'] = le.fit_transform(df['Data_Source'])
    return df

def prepare_features_target(df, target_column):
    features = [
        'Error_Rate (%)', 'CPU_Utilization (%)', 'Memory_Consumption (MB)',
        'Task_Execution_Time (ms)', 'System_Throughput (tasks/sec)',
        'Task_Waiting_Time (ms)', 'Number_of_Active_Users',
        'Network_Bandwidth_Utilization (Mbps)', 'Data_Source_encoded'
    ]
    X = df[features].values
    le = LabelEncoder()
    y = le.fit_transform(df[target_column])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, le, scaler, features

# ==========================================
# Train or Load Models
# ==========================================
def train_or_load_models(target_column):
    prefix = target_column.replace('_', '').lower()
    files = {
        'PAC': f'{MODELS_DIR}/{prefix}_pac.pkl',
        'NBC': f'{MODELS_DIR}/{prefix}_nbc.pkl',
        'KNN': f'{MODELS_DIR}/{prefix}_knn.pkl',
        'CRN-GRIM': f'{MODELS_DIR}/{prefix}_crngrim.pkl',
        'meta': f'{MODELS_DIR}/{prefix}_metadata.pkl'
    }

    if all(os.path.exists(f) for f in files.values()):
        models = {k: joblib.load(v) for k, v in files.items() if k != 'meta'}
        metadata = joblib.load(files['meta'])
        return models, metadata

    df = load_and_preprocess_data()
    X, y, le, scaler, cols = prepare_features_target(df, target_column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}
    models['PAC'] = PassiveAggressiveClassifier(max_iter=1000, random_state=42).fit(X_train, y_train)
    models['NBC'] = GaussianNB().fit(X_train, y_train)
    models['KNN'] = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)

    crn = CRNGRIMClassifier(target_column, input_dim=X_train.shape[1], num_classes=len(np.unique(y)))
    crn.fit(X_train, y_train)
    models['CRN-GRIM'] = crn

    for name in models:
        joblib.dump(models[name], files[name])
    joblib.dump({
        'label_encoder': le, 'scaler': scaler, 'feature_columns': cols,
        'X_test': X_test, 'y_test': y_test, 'class_names': le.classes_
    }, files['meta'])

    return models, joblib.load(files['meta'])

# ==========================================
# EDA Plots
# ==========================================
def generate_eda_plots(df):
    plots = {}
    plt.figure(figsize=(8, 6))
    df['Data_Source'].value_counts().plot(kind='pie', autopct='%1.1f%%', cmap='YlOrRd')
    plt.title('Data Source Distribution')
    plots['data_source_dist'] = plot_to_base64()

    plt.figure(figsize=(8, 6))
    df['Job_Priority'].value_counts().plot(kind='bar', color='orange')
    plt.title('Job Priority Distribution')
    plots['job_priority_dist'] = plot_to_base64()

    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, cmap='YlOrRd', fmt='.2f')
    plt.title('Correlation Matrix')
    plots['correlation'] = plot_to_base64()

    plt.figure(figsize=(15, 10))
    df.select_dtypes(include=[np.number]).hist(bins=20, color='gold', edgecolor='black', alpha=0.7, figsize=(15, 10), layout=(4, 3))
    plt.suptitle('Feature Distributions')
    plots['distributions'] = plot_to_base64()

    return plots

# ==========================================
# Classification Evaluation
# ==========================================
def evaluate_single_classifier(target_column, classifier_name):
    models, meta = train_or_load_models(target_column)
    model = models[classifier_name]
    X_test, y_test = meta['X_test'], meta['y_test']
    class_names = list(meta['class_names'])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else np.eye(len(class_names))[y_pred]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{classifier_name} - Confusion Matrix')
    cm_plot = plot_to_base64()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        auc = roc_auc_score(y_test, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    else:
        y_bin = label_binarize(y_test, classes=range(len(class_names)))
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{classifier_name} - ROC Curve')
    plt.legend()
    roc_plot = plot_to_base64()

    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'auc': roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr') if len(class_names) > 2 else roc_auc_score(y_test, y_proba[:, 1]),
        'classification_report': report
    }

    return {'metrics': metrics, 'cm_plot': cm_plot, 'roc_plot': roc_plot}

# ==========================================
# Other Functions
# ==========================================
def get_all_model_metrics():
    targets = ['Job_Priority', 'Scheduler_Type', 'Resource_Allocation_Type']
    data = {}

    for t in targets:
        models, meta = train_or_load_models(t)
        data[t] = {}

        for name, model in models.items():
            X_test, y_test = meta['X_test'], meta['y_test']

            # Predictions
            y_pred = model.predict(X_test)
            proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

            data[t][name] = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'auc': float(accuracy_score(y_test, y_pred))
            }

    return data


def plot_comparison(data):
    df = pd.DataFrame([
        {'Target': t, 'Classifier': c, 'Accuracy': m['accuracy']}
        for t, cls in data.items() for c, m in cls.items()
    ])
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x='Target', y='Accuracy', hue='Classifier', palette='YlOrRd')
    plt.title('Model Performance Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1))
    return plot_to_base64()

def plot_accuracy_trends(performances):
    if not performances:
        return None
    df = pd.DataFrame(performances)
    df['created_at'] = pd.to_datetime(df['created_at'])
    plt.figure(figsize=(14, 8))
    for (target, clf), group in df.groupby(['target_variable', 'classifier_name']):
        plt.plot(group['created_at'], group['accuracy'], marker='o', label=f'{target} - {clf}')
    plt.title('Model Accuracy Trends Over Time')
    plt.xlabel('Training Time')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(alpha=0.3)
    return plot_to_base64()

def predict_single(input_data, target_column):
    models, meta = train_or_load_models(target_column)
    scaled = meta['scaler'].transform([input_data])
    results = {}
    for name, model in models.items():
        pred = model.predict(scaled)[0]
        pred_class = meta['label_encoder'].inverse_transform([pred])[0]
        proba = model.predict_proba(scaled)[0] if hasattr(model, 'predict_proba') else None
        results[name] = {
            'prediction': pred_class,
            'probabilities': dict(zip(meta['class_names'], proba.tolist())) if proba is not None else {}
        }
    return results

def batch_predict(df_input):
    feature_cols = [
        'Error_Rate (%)', 'CPU_Utilization (%)', 'Memory_Consumption (MB)',
        'Task_Execution_Time (ms)', 'System_Throughput (tasks/sec)',
        'Task_Waiting_Time (ms)', 'Number_of_Active_Users',
        'Network_Bandwidth_Utilization (Mbps)', 'Data_Source_encoded'
    ]
    results = []
    for idx, row in df_input.iterrows():
        vec = row[feature_cols].values.tolist()
        pred = {t: predict_single(vec, t) for t in ['Job_Priority', 'Scheduler_Type', 'Resource_Allocation_Type']}
        results.append({'row': idx + 2, 'predictions': pred})
        if len(results) >= 100:
            break
    return results

# ==========================================
# RETRAIN WITH PERFORMANCE SAVING (FINAL FIX)
# ==========================================
def retrain_models(target_column, epochs=5, batch_size=32, perf_table=None):
    """
    Retrain all models and optionally save performance to TinyDB table
    perf_table: pass model_performances from routes.py
    """
    models, meta = train_or_load_models(target_column)  # This retrains
    X_test, y_test = meta['X_test'], meta['y_test']

    if perf_table is not None:
        for name, model in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            perf_table.insert({
                'target_variable': target_column,
                'classifier_name': name,
                'accuracy': float(acc),
                'precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'auc': 0.0,
                'created_at': datetime.utcnow().isoformat()
            })