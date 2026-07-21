"""
PTSD Risk Detection - Model Training Script
Generates a synthetic clinical dataset, trains an XGBoost classifier,
computes SHAP values, and saves all artifacts for the Flask web app.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
from xgboost import XGBClassifier
import shap
import joblib
import json


def generate_synthetic_data(n_samples=2000, random_state=42):
    """Generate a clinically-inspired synthetic PTSD dataset."""
    np.random.seed(random_state)

    data = {
        # Demographics
        'Age': np.random.randint(18, 75, n_samples),
        'Gender': np.random.choice(['Male', 'Female', 'Non-Binary'], n_samples, p=[0.45, 0.48, 0.07]),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], n_samples, p=[0.35, 0.35, 0.2, 0.1]),
        'Employment_Status': np.random.choice(['Employed', 'Unemployed', 'Student', 'Retired'], n_samples, p=[0.4, 0.25, 0.2, 0.15]),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], n_samples, p=[0.3, 0.35, 0.25, 0.1]),

        # Clinical Scores (PCL-5 subscales, 0-20 range each)
        'Intrusion_Score': np.random.randint(0, 21, n_samples),
        'Avoidance_Score': np.random.randint(0, 9, n_samples),
        'Negative_Cognition_Score': np.random.randint(0, 29, n_samples),
        'Hyperarousal_Score': np.random.randint(0, 25, n_samples),

        # Trauma & History
        'Trauma_Type': np.random.choice(['Combat', 'Assault', 'Accident', 'Natural_Disaster', 'Childhood_Abuse', 'Other'], n_samples),
        'LEC5_Score': np.random.randint(0, 68, n_samples),  # Life Events Checklist
        'Num_Traumatic_Events': np.random.randint(0, 10, n_samples),
        'Time_Since_Trauma_Months': np.random.randint(1, 360, n_samples),

        # Lifestyle & Support
        'Sleep_Quality': np.random.randint(0, 22, n_samples),  # PSQI score
        'Social_Support_Rating': np.random.randint(1, 8, n_samples),
        'Therapy_Adherence': np.random.uniform(0, 1, n_samples).round(2),
        'Exercise_Frequency': np.random.randint(0, 8, n_samples),
        'Alcohol_Use': np.random.randint(0, 41, n_samples),  # AUDIT score
        'Substance_Use': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),

        # Comorbidities
        'Depression_Score': np.random.randint(0, 28, n_samples),  # PHQ-9
        'Anxiety_Score': np.random.randint(0, 22, n_samples),  # GAD-7
        'Prior_MH_Diagnosis': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }

    df = pd.DataFrame(data)

    # Create a clinically-meaningful target variable
    risk_score = (
        0.20 * (df['Intrusion_Score'] / 20) +
        0.12 * (df['Avoidance_Score'] / 8) +
        0.15 * (df['Negative_Cognition_Score'] / 28) +
        0.15 * (df['Hyperarousal_Score'] / 24) +
        0.08 * (df['Sleep_Quality'] / 21) +
        0.07 * (df['Depression_Score'] / 27) +
        0.05 * (df['Anxiety_Score'] / 21) +
        0.05 * (df['Alcohol_Use'] / 40) +
        0.03 * (df['LEC5_Score'] / 67) +
        0.03 * (df['Num_Traumatic_Events'] / 9) +
        -0.05 * df['Therapy_Adherence'] +
        -0.03 * (df['Social_Support_Rating'] / 7) +
        -0.02 * (df['Exercise_Frequency'] / 7) +
        0.03 * df['Prior_MH_Diagnosis'] +
        0.02 * df['Substance_Use'] +
        np.random.normal(0, 0.08, n_samples)  # noise
    )

    df['PTSD_Diagnosis'] = (risk_score > 0.45).astype(int)

    return df


def train_and_save():
    """Train the XGBoost model and save all necessary artifacts."""
    print("=" * 60)
    print("PTSD EARLY RISK DETECTION - MODEL TRAINING")
    print("=" * 60)

    # Generate dataset
    print("\n[1/6] Generating synthetic clinical dataset...")
    df = generate_synthetic_data(n_samples=2000)

    # Save dataset
    os.makedirs('artifacts', exist_ok=True)
    df.to_csv('artifacts/ptsd_dataset.csv', index=False)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Class distribution:\n{df['PTSD_Diagnosis'].value_counts().to_string()}")

    # Encode categorical features
    print("\n[2/6] Encoding categorical features...")
    label_encoders = {}
    categorical_cols = ['Gender', 'Marital_Status', 'Employment_Status', 'Education_Level', 'Trauma_Type']

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    feature_cols = [c for c in df.columns if c != 'PTSD_Diagnosis']
    X = df[feature_cols]
    y = df['PTSD_Diagnosis']

    # Split and scale
    print("\n[3/6] Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    # Train XGBoost
    print("\n[4/6] Training XGBoost classifier...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    print("\n[5/6] Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    # Cross validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')

    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  CV Mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # SHAP values
    print("\n[6/6] Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    # Save all artifacts
    print("\nSaving artifacts...")
    joblib.dump(model, 'artifacts/xgboost_model.joblib')
    joblib.dump(scaler, 'artifacts/scaler.joblib')
    joblib.dump(label_encoders, 'artifacts/label_encoders.joblib')
    joblib.dump(explainer, 'artifacts/shap_explainer.joblib')

    # Save metrics as JSON
    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4),
        'confusion_matrix': cm.tolist(),
        'cv_scores': cv_scores.tolist(),
        'cv_mean': round(cv_scores.mean(), 4),
        'cv_std': round(cv_scores.std(), 4),
        'roc_fpr': fpr.tolist(),
        'roc_tpr': tpr.tolist(),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': len(feature_cols),
        'feature_names': feature_cols,
        'class_distribution': {
            'No PTSD': int((y == 0).sum()),
            'PTSD': int((y == 1).sum())
        }
    }
    with open('artifacts/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save feature importance
    feature_importance.to_csv('artifacts/feature_importance.csv', index=False)

    # Save SHAP summary data
    shap_summary = {
        'feature_names': feature_cols,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0).tolist(),
        'shap_values_sample': shap_values[:50].tolist(),
        'feature_values_sample': X_test_scaled.iloc[:50].values.tolist()
    }
    with open('artifacts/shap_summary.json', 'w') as f:
        json.dump(shap_summary, f)

    # Save dataset stats
    original_df = generate_synthetic_data(n_samples=2000)
    dataset_stats = {
        'total_samples': len(original_df),
        'age_distribution': {
            'mean': round(original_df['Age'].mean(), 1),
            'std': round(original_df['Age'].std(), 1),
            'min': int(original_df['Age'].min()),
            'max': int(original_df['Age'].max()),
            'bins': np.histogram(original_df['Age'], bins=10)[0].tolist(),
            'bin_edges': np.histogram(original_df['Age'], bins=10)[1].tolist()
        },
        'gender_distribution': original_df['Gender'].value_counts().to_dict(),
        'trauma_type_distribution': original_df['Trauma_Type'].value_counts().to_dict(),
        'employment_distribution': original_df['Employment_Status'].value_counts().to_dict(),
        'lec5_distribution': {
            'mean': round(original_df['LEC5_Score'].mean(), 1),
            'std': round(original_df['LEC5_Score'].std(), 1),
            'bins': np.histogram(original_df['LEC5_Score'], bins=10)[0].tolist(),
            'bin_edges': np.histogram(original_df['LEC5_Score'], bins=10)[1].tolist()
        },
        'class_distribution': original_df['PTSD_Diagnosis'].value_counts().to_dict(),
        'feature_correlations': original_df.select_dtypes(include=[np.number]).corr()['PTSD_Diagnosis'].drop('PTSD_Diagnosis').sort_values(ascending=False).to_dict()
    }
    with open('artifacts/dataset_stats.json', 'w') as f:
        json.dump(dataset_stats, f, indent=2)

    print("\n" + "=" * 60)
    print("ALL ARTIFACTS SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFiles saved in ./artifacts/:")
    for f_name in sorted(os.listdir('artifacts')):
        size = os.path.getsize(f'artifacts/{f_name}')
        print(f"  {f_name:40s} {size:>10,} bytes")


if __name__ == '__main__':
    train_and_save()
