from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your pre-trained model and scaler
with open('loan_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print("✅ Model and Scaler loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # 1. Numerical Inputs
            applicant_income = float(request.form['ApplicantIncome'])
            coapplicant_income = float(request.form['CoapplicantIncome'])
            age = float(request.form['Age'])
            dependents = float(request.form['Dependents'])
            credit_score = float(request.form['Credit_Score'])
            existing_loans = float(request.form['Existing_Loans'])
            dti_ratio = float(request.form['DTI_Ratio'])
            savings = float(request.form['Savings'])
            collateral_value = float(request.form['Collateral_Value'])
            loan_amount = float(request.form['Loan_Amount'])
            loan_term = float(request.form['Loan_Term'])
            education_level = int(request.form['Education_Level'])

            # 2. Categorical Dropdowns
            employment = request.form['Employment_Status']
            marital = request.form['Marital_Status']
            purpose = request.form['Loan_Purpose']
            property_area = request.form['Property_Area']
            gender = request.form['Gender']
            employer = request.form['Employer_Category']

            # 3. One-Hot Encoding Construction (Matches your notebook's 27 feature order)
            emp_salaried = 1.0 if employment == "Salaried" else 0.0
            emp_self = 1.0 if employment == "Self-employed" else 0.0
            emp_unemployed = 1.0 if employment == "Unemployed" else 0.0
            
            marital_single = 1.0 if marital == "Single" else 0.0
            
            purpose_car = 1.0 if purpose == "Car" else 0.0
            purpose_edu = 1.0 if purpose == "Education" else 0.0
            purpose_home = 1.0 if purpose == "Home" else 0.0
            purpose_personal = 1.0 if purpose == "Personal" else 0.0
            
            area_semiurban = 1.0 if property_area == "Semiurban" else 0.0
            area_urban = 1.0 if property_area == "Urban" else 0.0
            
            gender_male = 1.0 if gender == "Male" else 0.0
            
            emp_gov = 1.0 if employer == "Government" else 0.0
            emp_mnc = 1.0 if employer == "MNC" else 0.0
            emp_private = 1.0 if employer == "Private" else 0.0
            emp_unemp_cat = 1.0 if employer == "Unemployed" else 0.0

            # 4. Feature Engineering
            dti_ratio_sq = dti_ratio ** 2
            credit_score_sq = credit_score ** 2

            # 5. Assemble all 27 features in exact training column sequence
            input_features = [
                applicant_income, coapplicant_income, age, dependents, existing_loans,
                savings, collateral_value, loan_amount, loan_term, education_level,
                emp_salaried, emp_self, emp_unemployed, marital_single, purpose_car,
                purpose_edu, purpose_home, purpose_personal, area_semiurban, area_urban,
                gender_male, emp_gov, emp_mnc, emp_private, emp_unemp_cat,
                dti_ratio_sq, credit_score_sq
            ]

            # Convert to a 2D array for the scaler and model
            features_array = np.array([input_features])
            
            # 6. SCALE THE FEATURES using your saved scaler
            scaled_features = scaler.transform(features_array)
            
            # Make the prediction using scaled inputs
            prediction = model.predict(scaled_features)
            
            if prediction[0] == 1 or prediction[0] == 'Y':
                result_text = "🎉 Congratulations! The loan is APPROVED."
            else:
                result_text = "❌ Sorry, the loan is REJECTED."
                
            return render_template('index.html', prediction_text=result_text)
            
        except Exception as e:
            return render_template('index.html', prediction_text=f"❌ Error during prediction: {e}")

print("🚀 Starting Flask Server... Please open http://127.0.0.1:5000 in your browser.")
app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)