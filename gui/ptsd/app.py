"""
PTSD Early Risk Detection System
Flask Web Application - Clinical Precision Framework
Using Colab-trained model (ptsd_best_model.pkl)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Load Model Artifacts ───────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')


def load_artifacts():
    """Load the Colab-trained model and any available stats artifacts."""
    artifacts = {}
    try:
        # Load the real Colab-trained model (.pkl)
        model_path = os.path.join(ARTIFACTS_DIR, 'ptsd_best_model.pkl')
        artifacts['model'] = joblib.load(model_path)
        print(f"[OK] Loaded Colab model from {model_path}")
        print(f"     Model type: {type(artifacts['model']).__name__}")

        # Try loading optional stats files (if they exist from previous training)
        for fname, key in [
            ('metrics.json', 'metrics'),
            ('dataset_stats.json', 'dataset_stats'),
            ('shap_summary.json', 'shap_summary'),
        ]:
            fpath = os.path.join(ARTIFACTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath, 'r') as f:
                    artifacts[key] = json.load(f)

        if os.path.exists(os.path.join(ARTIFACTS_DIR, 'feature_importance.csv')):
            artifacts['feature_importance'] = pd.read_csv(
                os.path.join(ARTIFACTS_DIR, 'feature_importance.csv')
            ).to_dict('records')

        artifacts['loaded'] = True
        print("[OK] All artifacts loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        import traceback
        traceback.print_exc()
        artifacts['loaded'] = False

    return artifacts


ARTIFACTS = load_artifacts()

# ─── The 28 features the Colab model expects ───────────────────────────────
# From features.txt — exact order & names the model was trained on
MODEL_FEATURES = [
    'age', 'gender', 'marital_status', 'education_level', 'employment_status',
    'trauma_type', 'time_since_trauma_months', 'prior_psychiatric_history',
    'substance_use_history', 'sleep_disturbance', 'days_between_visits',
    'sleep_disturbance_numeric', 'Country', 'Age', 'Gender', 'Exercise Level',
    'Diet Type', 'Sleep Hours', 'Stress Level', 'Mental Health Condition',
    'Work Hours per Week', 'Screen Time per Day (Hours)',
    'Social Interaction Score', 'Happiness Score',
    'emotion_fear', 'emotion_sadness', 'emotion_anger', 'emotion_joy'
]

# Feature configuration for the assessment form — organized into sections
FEATURE_CONFIG = {
    'demographics': {
        'title': 'Demographics',
        'icon': 'person',
        'fields': [
            {'name': 'age', 'label': 'Age', 'type': 'number', 'min': 18, 'max': 100, 'default': 30, 'placeholder': 'e.g. 30'},
            {'name': 'gender', 'label': 'Gender', 'type': 'select', 'options': ['Male', 'Female']},
            {'name': 'marital_status', 'label': 'Marital Status', 'type': 'select', 'options': ['Single', 'Married', 'Divorced', 'Widowed']},
            {'name': 'education_level', 'label': 'Education Level', 'type': 'select', 'options': ['Primary', 'Secondary', 'Graduate', 'Postgraduate']},
            {'name': 'employment_status', 'label': 'Employment Status', 'type': 'select', 'options': ['Employed', 'Unemployed', 'Student', 'Retired']},
            {'name': 'Country', 'label': 'Country', 'type': 'select', 'options': ['India', 'USA', 'Australia', 'Canada', 'Brazil', 'Germany', 'Japan']},
        ]
    },
    'trauma': {
        'title': 'Trauma & History',
        'icon': 'psychology',
        'fields': [
            {'name': 'trauma_type', 'label': 'Primary Trauma Type', 'type': 'select', 'options': ['Combat', 'Sexual Assault', 'Road Accident', 'Natural Disaster', 'Domestic Violence', 'Childhood Abuse', 'Terrorist Attack', 'Workplace Incident']},
            {'name': 'time_since_trauma_months', 'label': 'Months Since Trauma', 'type': 'number', 'min': 1, 'max': 600, 'default': 12, 'placeholder': 'e.g. 12'},
            {'name': 'days_between_visits', 'label': 'Days Between Visits', 'type': 'number', 'min': 1, 'max': 365, 'default': 30, 'placeholder': 'e.g. 30'},
            {'name': 'prior_psychiatric_history', 'label': 'Prior Psychiatric History', 'type': 'select', 'options': ['Yes', 'No']},
            {'name': 'substance_use_history', 'label': 'Substance Use History', 'type': 'select', 'options': ['Yes', 'No']},
            {'name': 'Mental Health Condition', 'label': 'Mental Health Condition', 'type': 'select', 'options': ['Anxiety', 'Depression', 'PTSD', 'Bipolar']},
        ]
    },
    'clinical': {
        'title': 'Clinical Indicators',
        'icon': 'medical_services',
        'fields': [
            {'name': 'sleep_disturbance', 'label': 'Sleep Disturbance Severity', 'type': 'select', 'options': ['Mild', 'Severe']},
            {'name': 'sleep_disturbance_numeric', 'label': 'Sleep Disturbance Score (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 3},
            {'name': 'Sleep Hours', 'label': 'Sleep Hours per Day', 'type': 'range', 'min': 0, 'max': 16, 'step': 0.5, 'default': 7},
            {'name': 'Stress Level', 'label': 'Stress Level', 'type': 'select', 'options': ['Low', 'Moderate', 'High']},
            {'name': 'Happiness Score', 'label': 'Happiness Score (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 6},
            {'name': 'Social Interaction Score', 'label': 'Social Interaction Score (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 5},
        ]
    },
    'lifestyle': {
        'title': 'Lifestyle',
        'icon': 'favorite',
        'fields': [
            {'name': 'Exercise Level', 'label': 'Exercise Level', 'type': 'select', 'options': ['Low', 'Moderate', 'High']},
            {'name': 'Work Hours per Week', 'label': 'Work Hours per Week', 'type': 'number', 'min': 0, 'max': 100, 'default': 40, 'placeholder': 'e.g. 40'},
            {'name': 'Screen Time per Day (Hours)', 'label': 'Screen Time (Hours/Day)', 'type': 'range', 'min': 0, 'max': 16, 'step': 0.5, 'default': 5},
            {'name': 'Diet Type', 'label': 'Diet Type', 'type': 'select', 'options': ['Balanced', 'Vegetarian', 'Vegan', 'Keto', 'Junk Food']},
        ]
    },
    'emotional': {
        'title': 'Emotional Indicators',
        'icon': 'mood',
        'fields': [
            {'name': 'emotion_fear', 'label': 'Fear Level (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 3},
            {'name': 'emotion_sadness', 'label': 'Sadness Level (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 3},
            {'name': 'emotion_anger', 'label': 'Anger Level (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 2},
            {'name': 'emotion_joy', 'label': 'Joy Level (0-10)', 'type': 'range', 'min': 0, 'max': 10, 'step': 0.1, 'default': 6},
        ]
    }
}


# ─── Routes ────────────────────────────────────────────────────────────────
@app.route('/')
def home():
    """Home / Dashboard page."""
    metrics = ARTIFACTS.get('metrics', {})
    stats = ARTIFACTS.get('dataset_stats', {})
    return render_template('home.html',
                           metrics=metrics,
                           stats=stats,
                           loaded=ARTIFACTS.get('loaded', False))


@app.route('/assessment')
def assessment():
    """Patient assessment form page."""
    return render_template('assessment.html',
                           feature_config=FEATURE_CONFIG,
                           loaded=ARTIFACTS.get('loaded', False))


@app.route('/explainability')
def explainability():
    """Model explainability page."""
    shap_data = ARTIFACTS.get('shap_summary', {})
    feature_imp = ARTIFACTS.get('feature_importance', [])
    return render_template('explainability.html',
                           shap_data=shap_data,
                           feature_importance=feature_imp,
                           loaded=ARTIFACTS.get('loaded', False))


@app.route('/analytics')
def analytics():
    """Dataset analytics dashboard."""
    stats = ARTIFACTS.get('dataset_stats', {})
    metrics = ARTIFACTS.get('metrics', {})
    return render_template('analytics.html',
                           stats=stats,
                           metrics=metrics,
                           loaded=ARTIFACTS.get('loaded', False))


@app.route('/performance')
def performance():
    """Model performance report page."""
    metrics = ARTIFACTS.get('metrics', {})
    feature_imp = ARTIFACTS.get('feature_importance', [])
    return render_template('performance.html',
                           metrics=metrics,
                           feature_importance=feature_imp,
                           loaded=ARTIFACTS.get('loaded', False))


# ─── Prediction API ───────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Run prediction using the Colab-trained ptsd_best_model.pkl.
    Expects JSON with the 28 features.
    Returns risk level, probability, and feature analysis.
    """
    if not ARTIFACTS.get('loaded'):
        return jsonify({'error': 'Model not loaded. Check ptsd_best_model.pkl.'}), 500

    try:
        data = request.json
        model = ARTIFACTS['model']

        # Numeric features that the pipeline's StandardScaler expects as float
        numeric_fields = {
            'age', 'time_since_trauma_months', 'days_between_visits',
            'sleep_disturbance_numeric',
            'Age', 'Sleep Hours',
            'Work Hours per Week', 'Screen Time per Day (Hours)',
            'Social Interaction Score', 'Happiness Score',
            'emotion_fear', 'emotion_sadness', 'emotion_anger', 'emotion_joy'
        }

        # Build feature dict matching exact model column names
        features = {}
        for fname in MODEL_FEATURES:
            val = data.get(fname)
            if val is None:
                return jsonify({'error': f'Missing feature: {fname}'}), 400

            if fname in numeric_fields:
                features[fname] = float(val)
            else:
                # Categorical — keep as string for OneHotEncoder
                features[fname] = str(val)

        # Create DataFrame with proper feature order
        input_df = pd.DataFrame([features], columns=MODEL_FEATURES)

        # Predict using the full pipeline (ColumnTransformer + XGBClassifier)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1]) * 100

        # Risk level classification
        if probability < 40:
            risk_level = 'LOW'
            risk_color = '#2e7d32'
            advice = 'Maintain healthy habits and emotional well-being. Regular check-ups are recommended.'
            precautions = [
                {
                    'category': 'Mindfulness & Meditation',
                    'icon': 'self_improvement',
                    'priority': 'recommended',
                    'suggestions': [
                        'Practice 10-15 minutes of daily mindfulness meditation',
                        'Try guided breathing exercises (4-7-8 technique)',
                        'Keep a gratitude journal — write 3 things daily',
                    ]
                },
                {
                    'category': 'Physical Activity',
                    'icon': 'fitness_center',
                    'priority': 'recommended',
                    'suggestions': [
                        'Maintain 30 min of moderate exercise 5 days/week',
                        'Consider yoga or tai chi for mind-body connection',
                        'Take regular walks in nature (green therapy)',
                    ]
                },
                {
                    'category': 'Nutrition & Diet',
                    'icon': 'restaurant',
                    'priority': 'helpful',
                    'suggestions': [
                        'Follow a balanced diet rich in omega-3 fatty acids',
                        'Include magnesium-rich foods (leafy greens, nuts, seeds)',
                        'Stay hydrated — aim for 8 glasses of water daily',
                    ]
                },
                {
                    'category': 'Sleep Hygiene',
                    'icon': 'bedtime',
                    'priority': 'helpful',
                    'suggestions': [
                        'Maintain a consistent sleep schedule (7-9 hours)',
                        'Avoid screens 1 hour before bedtime',
                        'Create a calm, dark, and cool sleep environment',
                    ]
                },
                {
                    'category': 'Social Connection',
                    'icon': 'groups',
                    'priority': 'helpful',
                    'suggestions': [
                        'Stay connected with friends and family',
                        'Engage in community or volunteer activities',
                        'Share feelings with a trusted person regularly',
                    ]
                },
            ]
        elif probability < 70:
            risk_level = 'MODERATE'
            risk_color = '#f57c00'
            advice = 'Consider speaking with a counselor or trusted person. Monitoring recommended.'
            precautions = [
                {
                    'category': 'Professional Support',
                    'icon': 'support_agent',
                    'priority': 'important',
                    'suggestions': [
                        'Schedule an appointment with a licensed counselor',
                        'Consider Cognitive Behavioral Therapy (CBT) sessions',
                        'Explore online therapy platforms for regular check-ins',
                    ]
                },
                {
                    'category': 'Mindfulness & Stress Management',
                    'icon': 'self_improvement',
                    'priority': 'important',
                    'suggestions': [
                        'Practice 20+ minutes of guided meditation daily',
                        'Learn and use grounding techniques (5-4-3-2-1 method)',
                        'Try progressive muscle relaxation before sleep',
                        'Use mindfulness apps (Headspace, Calm, Insight Timer)',
                    ]
                },
                {
                    'category': 'Physical Activity',
                    'icon': 'fitness_center',
                    'priority': 'important',
                    'suggestions': [
                        'Engage in 30-45 min of aerobic exercise daily',
                        'Practice yoga — especially trauma-sensitive yoga',
                        'Try swimming or cycling for low-impact cardio',
                    ]
                },
                {
                    'category': 'Nutrition & Diet',
                    'icon': 'restaurant',
                    'priority': 'recommended',
                    'suggestions': [
                        'Follow an anti-inflammatory diet (Mediterranean-style)',
                        'Increase intake of B-vitamins and omega-3s',
                        'Limit caffeine and processed sugar intake',
                        'Consider probiotics for the gut-brain connection',
                    ]
                },
                {
                    'category': 'Sleep & Relaxation',
                    'icon': 'bedtime',
                    'priority': 'important',
                    'suggestions': [
                        'Establish a strict 7-9 hour sleep routine',
                        'Avoid caffeine after 2 PM',
                        'Use white noise or calming sounds for sleep',
                        'Practice sleep-onset relaxation techniques',
                    ]
                },
                {
                    'category': 'Social & Lifestyle',
                    'icon': 'groups',
                    'priority': 'recommended',
                    'suggestions': [
                        'Join a peer support group (in-person or online)',
                        'Limit alcohol and substance use',
                        'Reduce screen time — especially news/social media',
                        'Spend at least 30 min outdoors daily',
                    ]
                },
                {
                    'category': 'Journaling & Self-awareness',
                    'icon': 'edit_note',
                    'priority': 'recommended',
                    'suggestions': [
                        'Maintain a mood and trigger journal',
                        'Track sleep patterns and emotional fluctuations',
                        'Write about positive experiences and coping wins',
                    ]
                },
            ]
        else:
            risk_level = 'HIGH'
            risk_color = '#c62828'
            advice = 'Please seek support from a mental health professional as soon as possible.'
            precautions = [
                {
                    'category': 'Immediate Professional Help',
                    'icon': 'emergency',
                    'priority': 'critical',
                    'suggestions': [
                        'Seek immediate consultation with a psychiatrist or psychologist',
                        'Consider evidence-based therapies: EMDR, CPT, or Prolonged Exposure',
                        'Discuss medication options with your doctor if needed',
                        'If in crisis, call a helpline: 988 (USA), iCall 9152987821 (India)',
                    ]
                },
                {
                    'category': 'Daily Mindfulness & Grounding',
                    'icon': 'self_improvement',
                    'priority': 'critical',
                    'suggestions': [
                        'Practice grounding exercises multiple times daily',
                        'Use the 5-4-3-2-1 sensory technique during distress',
                        'Engage in guided trauma-focused meditation (20-30 min)',
                        'Try body-scan meditation to reconnect with your body',
                    ]
                },
                {
                    'category': 'Physical Wellness',
                    'icon': 'fitness_center',
                    'priority': 'important',
                    'suggestions': [
                        'Start with gentle exercise — walking, stretching, yoga',
                        'Aim for 30 min of movement daily to regulate cortisol',
                        'Consider trauma-sensitive yoga classes',
                        'Avoid intense workouts if they trigger hyperarousal',
                    ]
                },
                {
                    'category': 'Nutrition for Recovery',
                    'icon': 'restaurant',
                    'priority': 'important',
                    'suggestions': [
                        'Follow a nutrient-dense, anti-inflammatory diet',
                        'Prioritize omega-3s (fish, flaxseed, walnuts)',
                        'Eat magnesium-rich foods to calm the nervous system',
                        'Avoid alcohol, excessive caffeine, and refined sugar',
                        'Consider supplements: Vitamin D, B-complex (consult doctor)',
                    ]
                },
                {
                    'category': 'Sleep Recovery Protocol',
                    'icon': 'bedtime',
                    'priority': 'critical',
                    'suggestions': [
                        'Create a strict wind-down routine 1 hour before bed',
                        'Use sleep-restriction therapy if insomnia persists',
                        'Keep a notepad by bed to write down intrusive thoughts',
                        'Consult a doctor about sleep aids if nightmares are severe',
                    ]
                },
                {
                    'category': 'Safety & Support Network',
                    'icon': 'shield',
                    'priority': 'critical',
                    'suggestions': [
                        'Build a safety plan with trusted people and contacts',
                        'Remove access to harmful substances or objects',
                        'Identify at least 3 people you can call during distress',
                        'Join a trauma survivors support group',
                    ]
                },
                {
                    'category': 'Lifestyle Adjustments',
                    'icon': 'tune',
                    'priority': 'important',
                    'suggestions': [
                        'Reduce work hours if possible — prioritize recovery',
                        'Minimize exposure to triggering content and news',
                        'Spend time in nature — forest bathing, park walks',
                        'Practice creative outlets: art, music, or writing therapy',
                    ]
                },
            ]

        # Build feature-level analysis (impact indicators)
        risk_factors = []
        protective_factors = []

        # Highlight key risk/protective indicators based on input values
        analysis_rules = [
            ('sleep_disturbance_numeric', 6, 'risk', 'High sleep disturbance score'),
            ('Happiness Score', 4, 'protective_below', 'Low happiness score'),
            ('emotion_fear', 6, 'risk', 'Elevated fear level'),
            ('emotion_sadness', 6, 'risk', 'Elevated sadness level'),
            ('emotion_anger', 6, 'risk', 'Elevated anger level'),
            ('emotion_joy', 5, 'protective_above', 'Healthy joy level'),
            ('Social Interaction Score', 4, 'protective_below', 'Low social interaction'),
            ('Sleep Hours', 6, 'protective_above', 'Adequate sleep duration'),
        ]

        explanations = []
        for fname in MODEL_FEATURES:
            val = features[fname]
            direction = 'neutral'
            impact = 0.0

            # Determine direction for numeric features
            for rule_name, threshold, rule_type, label in analysis_rules:
                if fname == rule_name:
                    num_val = float(val) if isinstance(val, (int, float)) else 0
                    if rule_type == 'risk' and num_val >= threshold:
                        direction = 'risk'
                        impact = (num_val - threshold) / 10
                    elif rule_type == 'protective_below' and num_val < threshold:
                        direction = 'risk'
                        impact = (threshold - num_val) / 10
                    elif rule_type == 'protective_above' and num_val >= threshold:
                        direction = 'protective'
                        impact = (num_val - threshold) / 10
                    break

            if isinstance(val, (int, float)):
                explanations.append({
                    'feature': fname,
                    'value': float(val),
                    'shap_value': round(impact if direction == 'risk' else -impact, 4),
                    'direction': direction,
                    'abs_impact': round(abs(impact), 4)
                })

        # Sort by absolute impact
        explanations.sort(key=lambda x: x['abs_impact'], reverse=True)

        return jsonify({
            'prediction': prediction,
            'probability': round(probability / 100, 4),  # 0-1 scale for the gauge
            'probability_pct': round(probability, 2),     # percentage for display
            'risk_level': risk_level,
            'risk_color': risk_color,
            'advice': advice,
            'explanations': explanations[:10],  # Top 10 most impactful
            'precautions': precautions,
            'base_value': 0.5,
            'input_summary': {
                'stress': features.get('Stress Level', 'N/A'),
                'happiness': features.get('Happiness Score', 0),
                'sleep_disturbance': features.get('sleep_disturbance_numeric', 0),
                'fear': features.get('emotion_fear', 0),
                'sadness': features.get('emotion_sadness', 0),
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics')
def api_metrics():
    """Return model metrics as JSON."""
    return jsonify(ARTIFACTS.get('metrics', {}))


@app.route('/api/dataset-stats')
def api_dataset_stats():
    """Return dataset statistics."""
    return jsonify(ARTIFACTS.get('dataset_stats', {}))


@app.route('/api/shap-summary')
def api_shap_summary():
    """Return SHAP summary data."""
    return jsonify(ARTIFACTS.get('shap_summary', {}))


@app.route('/api/feature-importance')
def api_feature_importance():
    """Return feature importance rankings."""
    return jsonify(ARTIFACTS.get('feature_importance', []))


if __name__ == '__main__':
    if not ARTIFACTS.get('loaded'):
        print("\n[WARNING] Model not found!")
        print("   Make sure ptsd_best_model.pkl is in the artifacts/ folder.")
        print("   Then restart the Flask app.\n")
    app.run(debug=True, port=5001)
