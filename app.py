"""
Loan Default Risk Calculator - Streamlit Web App
This app provides an interactive interface to predict loan default risk
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Loan Risk Calculator",
    page_icon="💰",
    layout="wide"
)

# Title and description
st.title("💰 Loan Default Risk Calculator")
st.markdown("""
This tool predicts the likelihood of a loan applicant defaulting on their loan.
Enter the applicant's information below to get a risk assessment.
""")

# Check if model exists, if not train a simple one
@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new one"""
    if os.path.exists('loan_data.csv'):
        # Load the data
        df = pd.read_csv('loan_data.csv')
        
        # Feature engineering
        df['loan_to_income_ratio'] = df['loan_amount'] / df['income']
        df['income_per_credit_line'] = df['income'] / (df['num_credit_lines'] + 1)
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                 labels=['young', 'middle', 'senior', 'elderly'])
        df['credit_score_bucket'] = pd.cut(df['credit_score'], 
                                           bins=[0, 580, 670, 740, 850],
                                           labels=['poor', 'fair', 'good', 'excellent'])
        
        # One-hot encode
        df_encoded = pd.get_dummies(df, columns=['age_group', 'credit_score_bucket'], 
                                    drop_first=True)
        
        # Prepare data
        X = df_encoded.drop('default', axis=1)
        y = df_encoded['default']
        
        # Train model
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                          max_depth=5, random_state=42)
        model.fit(X, y)
        
        return model, X.columns
    else:
        st.error("⚠️ loan_data.csv not found. Please run loan_prediction.py first to generate the data.")
        return None, None

# Load model
model, feature_columns = load_or_train_model()

if model is not None:
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Personal Information")
        age = st.slider("Age", 18, 70, 35, help="Applicant's age in years")
        income = st.number_input("Annual Income ($)", 
                                min_value=15000, 
                                max_value=500000, 
                                value=50000, 
                                step=5000,
                                help="Gross annual income")
        employment_length = st.slider("Employment Length (years)", 
                                     0, 40, 5,
                                     help="Years in current employment")
        
    with col2:
        st.subheader("💳 Credit Information")
        credit_score = st.slider("Credit Score", 
                                300, 850, 650,
                                help="FICO credit score (300-850)")
        loan_amount = st.number_input("Requested Loan Amount ($)", 
                                     min_value=1000, 
                                     max_value=200000, 
                                     value=25000, 
                                     step=1000,
                                     help="Amount of loan requested")
        num_credit_lines = st.slider("Number of Credit Lines", 
                                    1, 15, 5,
                                    help="Total number of credit accounts")
        
    # Additional inputs
    st.subheader("📊 Financial Details")
    col3, col4 = st.columns(2)
    
    with col3:
        debt_to_income = st.slider("Debt-to-Income Ratio", 
                                   0.0, 0.8, 0.3, 0.05,
                                   help="Monthly debt payments / Monthly income")
    
    with col4:
        previous_defaults = st.selectbox("Previous Defaults", 
                                        [0, 1, 2, 3],
                                        help="Number of previous loan defaults")
    
    # Calculate risk button
    st.markdown("---")
    
    if st.button("🔍 Calculate Risk", type="primary", use_container_width=True):
        # Create applicant data
        applicant = {
            'age': age,
            'income': income,
            'loan_amount': loan_amount,
            'credit_score': credit_score,
            'employment_length': employment_length,
            'num_credit_lines': num_credit_lines,
            'debt_to_income': debt_to_income,
            'previous_defaults': previous_defaults
        }
        
        # Feature engineering
        input_df = pd.DataFrame([applicant])
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
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_columns]
        
        # Make prediction
        risk_probability = model.predict_proba(input_encoded)[0, 1]
        prediction = model.predict(input_encoded)[0]
        
        # Determine risk level
        if risk_probability < 0.3:
            risk_level = "LOW"
            recommendation = "✅ APPROVE"
            color = "green"
        elif risk_probability < 0.6:
            risk_level = "MEDIUM"
            recommendation = "⚠️ REVIEW MANUALLY"
            color = "orange"
        else:
            risk_level = "HIGH"
            recommendation = "❌ REJECT"
            color = "red"
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Risk Assessment Results")
        
        # Create three columns for metrics
        metric1, metric2, metric3 = st.columns(3)
        
        with metric1:
            st.metric("Default Probability", f"{risk_probability:.1%}")
        
        with metric2:
            st.metric("Risk Level", risk_level)
        
        with metric3:
            st.metric("Recommendation", recommendation)
        
        # Visual risk indicator
        st.markdown("### Risk Meter")
        progress_color = "🟢" if risk_probability < 0.3 else "🟡" if risk_probability < 0.6 else "🔴"
        st.progress(risk_probability)
        st.markdown(f"{progress_color} **Risk Score: {risk_probability:.1%}**")
        
        # Detailed analysis
        with st.expander("📈 Detailed Analysis"):
            st.markdown("#### Key Risk Factors:")
            
            # Calculate individual risk factors
            factors = []
            
            if credit_score < 600:
                factors.append(f"• **Low credit score** ({credit_score}) - High Risk")
            elif credit_score < 670:
                factors.append(f"• **Fair credit score** ({credit_score}) - Moderate Risk")
            else:
                factors.append(f"• **Good credit score** ({credit_score}) - Low Risk")
            
            loan_to_income = loan_amount / income
            if loan_to_income > 0.5:
                factors.append(f"• **High loan-to-income ratio** ({loan_to_income:.2f}) - High Risk")
            elif loan_to_income > 0.3:
                factors.append(f"• **Moderate loan-to-income ratio** ({loan_to_income:.2f}) - Moderate Risk")
            else:
                factors.append(f"• **Low loan-to-income ratio** ({loan_to_income:.2f}) - Low Risk")
            
            if debt_to_income > 0.4:
                factors.append(f"• **High debt-to-income ratio** ({debt_to_income:.1%}) - High Risk")
            elif debt_to_income > 0.3:
                factors.append(f"• **Moderate debt-to-income ratio** ({debt_to_income:.1%}) - Moderate Risk")
            else:
                factors.append(f"• **Low debt-to-income ratio** ({debt_to_income:.1%}) - Low Risk")
            
            if previous_defaults > 0:
                factors.append(f"• **Previous defaults** ({previous_defaults}) - High Risk")
            else:
                factors.append(f"• **No previous defaults** - Low Risk")
            
            for factor in factors:
                st.markdown(factor)
            
            # Show applicant summary
            st.markdown("#### Applicant Summary:")
            summary_data = {
                'Attribute': ['Age', 'Income', 'Loan Amount', 'Credit Score', 
                             'Employment Length', 'Credit Lines', 'Debt-to-Income', 'Previous Defaults'],
                'Value': [f"{age} years", f"${income:,}", f"${loan_amount:,}", 
                         credit_score, f"{employment_length} years", num_credit_lines,
                         f"{debt_to_income:.1%}", previous_defaults]
            }
            st.dataframe(pd.DataFrame(summary_data), hide_index=True, use_container_width=True)
        
        # Recommendations based on risk level
        st.markdown("### 💡 Recommendations")
        if risk_level == "LOW":
            st.success("""
            ✅ **Approve the loan application**
            - Low risk of default
            - All risk factors within acceptable ranges
            - Proceed with standard terms and conditions
            """)
        elif risk_level == "MEDIUM":
            st.warning("""
            ⚠️ **Manual review recommended**
            - Moderate risk of default
            - Consider additional documentation or collateral
            - May require adjusted terms (higher interest rate or shorter term)
            - Request additional financial information
            """)
        else:
            st.error("""
            ❌ **Reject the loan application or require significant changes**
            - High risk of default
            - Multiple risk factors present
            - Consider requiring co-signer, significant down payment, or collateral
            - Suggest credit improvement steps before reapplication
            """)

    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This loan default risk calculator uses machine learning to predict 
        the likelihood of a loan applicant defaulting on their loan.
        
        **Model**: Gradient Boosting Classifier
        
        **Risk Levels**:
        - 🟢 **LOW** (< 30%): Approve
        - 🟡 **MEDIUM** (30-60%): Review
        - 🔴 **HIGH** (> 60%): Reject
        
        **Key Factors**:
        - Credit Score
        - Debt-to-Income Ratio
        - Loan-to-Income Ratio
        - Previous Defaults
        - Employment History
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        if os.path.exists('model_evaluation.png'):
            st.image('model_evaluation.png', use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        ### 🎯 Tips
        - All fields are required
        - Use realistic values
        - Credit scores range: 300-850
        - Debt-to-income should be < 0.43
        """)

else:
    st.error("""
    ### ⚠️ Setup Required
    
    Please run `loan_prediction.py` first to generate the required data:
    
    ```bash
    python loan_prediction.py
    ```
    
    This will create the `loan_data.csv` file needed for predictions.
    """)