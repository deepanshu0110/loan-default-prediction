# Loan Default Prediction - Complete Beginner's Guide
# This project predicts whether a loan applicant will default on their loan

# ============================================================================
# PART 1: SETUP AND DATA GENERATION
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("LOAN DEFAULT PREDICTION PROJECT")
print("=" * 80)

# Generate synthetic loan data
def generate_loan_data(n_samples=10000):
    """
    Generate synthetic loan application data
    """
    print("\n[1/7] Generating synthetic loan data...")
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.exponential(50000, n_samples) + 20000,
        'loan_amount': np.random.exponential(30000, n_samples) + 5000,
        'credit_score': np.random.normal(650, 100, n_samples),
        'employment_length': np.random.randint(0, 40, n_samples),
        'num_credit_lines': np.random.randint(1, 15, n_samples),
        'debt_to_income': np.random.uniform(0, 0.6, n_samples),
        'previous_defaults': np.random.choice([0, 1, 2, 3], n_samples, p=[0.7, 0.2, 0.07, 0.03])
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to realistic ranges
    df['credit_score'] = df['credit_score'].clip(300, 850)
    df['income'] = df['income'].clip(15000, 500000)
    df['loan_amount'] = df['loan_amount'].clip(1000, 200000)
    
    # Generate default status based on features (higher risk = higher default probability)
    default_prob = (
        (df['credit_score'] < 600) * 0.3 +
        (df['debt_to_income'] > 0.4) * 0.25 +
        (df['previous_defaults'] > 0) * 0.2 +
        (df['loan_amount'] > df['income'] * 0.5) * 0.15 +
        np.random.uniform(0, 0.1, n_samples)
    )
    
    df['default'] = (default_prob > 0.4).astype(int)
    
    print(f"✓ Generated {n_samples} loan records")
    print(f"  - Default rate: {df['default'].mean():.2%}")
    
    return df

# Generate data
df = generate_loan_data(10000)

# Save to CSV
df.to_csv('loan_data.csv', index=False)
print("✓ Data saved to 'loan_data.csv'")

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n[2/7] Performing Exploratory Data Analysis...")

# Basic statistics
print("\n--- Dataset Overview ---")
print(f"Shape: {df.shape}")
print(f"\n{df.info()}")
print(f"\n--- Statistical Summary ---")
print(df.describe())

# Check for missing values
print(f"\n--- Missing Values ---")
print(df.isnull().sum())

# Create EDA visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Exploratory Data Analysis - Loan Default Prediction', fontsize=16)

# 1. Income vs Default
axes[0, 0].hist([df[df['default']==0]['income'], df[df['default']==1]['income']], 
                bins=50, label=['No Default', 'Default'], alpha=0.7)
axes[0, 0].set_xlabel('Income ($)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Income Distribution by Default Status')
axes[0, 0].legend()

# 2. Loan Amount Distribution
axes[0, 1].hist(df['loan_amount'], bins=50, color='skyblue', edgecolor='black')
axes[0, 1].set_xlabel('Loan Amount ($)')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Loan Amount Distribution')

# 3. Credit Score vs Default
axes[0, 2].boxplot([df[df['default']==0]['credit_score'], 
                     df[df['default']==1]['credit_score']], 
                    labels=['No Default', 'Default'])
axes[0, 2].set_ylabel('Credit Score')
axes[0, 2].set_title('Credit Score by Default Status')

# 4. Default Rate by Age Groups
age_groups = pd.cut(df['age'], bins=[18, 30, 40, 50, 70])
default_by_age = df.groupby(age_groups)['default'].mean()
default_by_age.plot(kind='bar', ax=axes[1, 0], color='coral')
axes[1, 0].set_xlabel('Age Group')
axes[1, 0].set_ylabel('Default Rate')
axes[1, 0].set_title('Default Rate by Age Group')
axes[1, 0].tick_params(axis='x', rotation=45)

# 5. Correlation Heatmap
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[1, 1], cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Feature Correlation Heatmap')

# 6. Default Rate
default_counts = df['default'].value_counts()
axes[1, 2].pie(default_counts, labels=['No Default', 'Default'], 
               autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
axes[1, 2].set_title('Default Distribution')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("✓ EDA visualizations saved to 'eda_analysis.png'")

# ============================================================================
# PART 3: FEATURE ENGINEERING
# ============================================================================

print("\n[3/7] Performing Feature Engineering...")

# Create new features
df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
df['income_per_credit_line'] = df['income'] / (df['num_credit_lines'] + 1)
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                         labels=['young', 'middle', 'senior', 'elderly'])

# Create credit score buckets
df['credit_score_bucket'] = pd.cut(df['credit_score'], 
                                   bins=[0, 580, 670, 740, 850],
                                   labels=['poor', 'fair', 'good', 'excellent'])

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=['age_group', 'credit_score_bucket'], 
                            drop_first=True)

print("✓ Created new features:")
print("  - loan_to_income_ratio")
print("  - income_per_credit_line")
print("  - age_group (categorical)")
print("  - credit_score_bucket (categorical)")
print(f"\n✓ Total features after encoding: {df_encoded.shape[1] - 1}")

# ============================================================================
# PART 4: DATA PREPARATION
# ============================================================================

print("\n[4/7] Preparing data for modeling...")

# Separate features and target
X = df_encoded.drop('default', axis=1)
y = df_encoded['default']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set size: {X_train.shape[0]} samples")
print(f"✓ Test set size: {X_test.shape[0]} samples")
print(f"✓ Train default rate: {y_train.mean():.2%}")
print(f"✓ Test default rate: {y_test.mean():.2%}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled using StandardScaler")

# ============================================================================
# PART 5: MODEL TRAINING
# ============================================================================

print("\n[5/7] Training models...")

# Model 1: Logistic Regression
print("\n--- Training Logistic Regression ---")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
print("✓ Logistic Regression trained")

# Model 2: Gradient Boosting
print("\n--- Training Gradient Boosting ---")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                     max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
print("✓ Gradient Boosting trained")

# ============================================================================
# PART 6: MODEL EVALUATION
# ============================================================================

print("\n[6/7] Evaluating models...")

# Create evaluation plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Evaluation - Loan Default Prediction', fontsize=16)

# 1. ROC Curves
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred_proba)
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_pred_proba)

axes[0, 0].plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, lr_pred_proba):.3f})')
axes[0, 0].plot(gb_fpr, gb_tpr, label=f'Gradient Boosting (AUC={roc_auc_score(y_test, gb_pred_proba):.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('ROC Curve Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Precision-Recall Curves
lr_precision, lr_recall, _ = precision_recall_curve(y_test, lr_pred_proba)
gb_precision, gb_recall, _ = precision_recall_curve(y_test, gb_pred_proba)

axes[0, 1].plot(lr_recall, lr_precision, 
                label=f'Logistic Regression (AP={average_precision_score(y_test, lr_pred_proba):.3f})')
axes[0, 1].plot(gb_recall, gb_precision, 
                label=f'Gradient Boosting (AP={average_precision_score(y_test, gb_pred_proba):.3f})')
axes[0, 1].set_xlabel('Recall')
axes[0, 1].set_ylabel('Precision')
axes[0, 1].set_title('Precision-Recall Curve Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (Gradient Boosting)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': gb_model.feature_importances_
}).sort_values('importance', ascending=False).head(10)

axes[1, 0].barh(feature_importance['feature'], feature_importance['importance'])
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Top 10 Feature Importances (Gradient Boosting)')
axes[1, 0].invert_yaxis()

# 4. Confusion Matrix (Gradient Boosting)
cm = confusion_matrix(y_test, gb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix (Gradient Boosting)')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("✓ Evaluation visualizations saved to 'model_evaluation.png'")

# Print detailed metrics
print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(f"ROC AUC Score: {roc_auc_score(y_test, lr_pred_proba):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, lr_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))

print("\n=== GRADIENT BOOSTING RESULTS ===")
print(f"ROC AUC Score: {roc_auc_score(y_test, gb_pred_proba):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, gb_pred_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, gb_pred))

# ============================================================================
# PART 7: DEPLOYMENT - RISK SCORING TOOL
# ============================================================================

print("\n[7/7] Creating Risk Scoring Tool...")

def predict_loan_risk(applicant_data, model=gb_model, scaler=scaler):
    """
    Predict loan default risk for a new applicant
    
    Parameters:
    -----------
    applicant_data : dict
        Dictionary containing applicant information
    
    Returns:
    --------
    dict : Risk assessment results
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([applicant_data])
    
    # Feature engineering (same as training)
    input_df['loan_to_income_ratio'] = input_df['loan_amount'] / input_df['income']
    input_df['income_per_credit_line'] = input_df['income'] / (input_df['num_credit_lines'] + 1)
    
    # Create categorical features
    input_df['age_group'] = pd.cut(input_df['age'], bins=[0, 30, 45, 60, 100], 
                                   labels=['young', 'middle', 'senior', 'elderly'])
    input_df['credit_score_bucket'] = pd.cut(input_df['credit_score'], 
                                             bins=[0, 580, 670, 740, 850],
                                             labels=['poor', 'fair', 'good', 'excellent'])
    
    # One-hot encode
    input_encoded = pd.get_dummies(input_df, columns=['age_group', 'credit_score_bucket'])
    
    # Ensure all columns from training are present
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[X.columns]
    
    # Predict
    risk_probability = model.predict_proba(input_encoded)[0, 1]
    prediction = model.predict(input_encoded)[0]
    
    # Determine risk level
    if risk_probability < 0.3:
        risk_level = "LOW"
        recommendation = "APPROVE"
    elif risk_probability < 0.6:
        risk_level = "MEDIUM"
        recommendation = "REVIEW MANUALLY"
    else:
        risk_level = "HIGH"
        recommendation = "REJECT"
    
    return {
        'default_probability': risk_probability,
        'prediction': 'DEFAULT' if prediction == 1 else 'NO DEFAULT',
        'risk_level': risk_level,
        'recommendation': recommendation
    }

# Example usage
print("\n=== RISK SCORING TOOL DEMO ===")

# Example applicant 1: Low risk
applicant1 = {
    'age': 35,
    'income': 75000,
    'loan_amount': 25000,
    'credit_score': 720,
    'employment_length': 8,
    'num_credit_lines': 5,
    'debt_to_income': 0.25,
    'previous_defaults': 0
}

result1 = predict_loan_risk(applicant1)
print("\nApplicant 1 (Expected: Low Risk)")
print(f"  Default Probability: {result1['default_probability']:.2%}")
print(f"  Risk Level: {result1['risk_level']}")
print(f"  Recommendation: {result1['recommendation']}")

# Example applicant 2: High risk
applicant2 = {
    'age': 28,
    'income': 35000,
    'loan_amount': 45000,
    'credit_score': 550,
    'employment_length': 2,
    'num_credit_lines': 8,
    'debt_to_income': 0.45,
    'previous_defaults': 2
}

result2 = predict_loan_risk(applicant2)
print("\nApplicant 2 (Expected: High Risk)")
print(f"  Default Probability: {result2['default_probability']:.2%}")
print(f"  Risk Level: {result2['risk_level']}")
print(f"  Recommendation: {result2['recommendation']}")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)
print("\nGenerated Files:")
print("  1. loan_data.csv - Synthetic loan dataset")
print("  2. eda_analysis.png - Exploratory data analysis visualizations")
print("  3. model_evaluation.png - Model performance metrics")
print("\nNext Steps:")
print("  - Try the predict_loan_risk() function with different applicants")
print("  - Experiment with different model parameters")
print("  - Add SHAP analysis (install with: pip install shap)")
print("=" * 80)
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(gb_model)
shap_values = explainer.shap_values(X_test)

# Plot feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.savefig('shap_importance.png')

# Explain single prediction
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])