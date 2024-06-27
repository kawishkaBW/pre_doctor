from flask import Flask, render_template, redirect, request, url_for
from flask_sqlalchemy import SQLAlchemy
import secrets
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configure SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///registrations.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define Registration model
class Registration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    serial_key = db.Column(db.String(8), unique=True, nullable=False)
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    address = db.Column(db.String(200), nullable=False)
    telephone = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(100))
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    blood_group = db.Column(db.String(3), nullable=False)
    married = db.Column(db.String(3), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

# Load diabetes model
with open('model/diabetes.pickle', 'rb') as fileDiabetes:
    modelDiabetes = pickle.load(fileDiabetes)

# Load heart disease model
with open('model/hearts.pickle', 'rb') as fileHeart:
    modelHeart = pickle.load(fileHeart)

# Load kidney disease model
with open('model/kidney.pickle', 'rb') as fileKidney:
    modelKidney = pickle.load(fileKidney)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Function to predict diabetes using the loaded model
def diabetes_prediction(feature_list):
    try:
        feature_array = np.array(feature_list).reshape(1, -1)
        pred_value = modelDiabetes.predict(feature_array)
        return pred_value[0]
    except Exception as e:
        logging.error(f"Error in diabetes prediction: {e}")
        return 0

# Function to predict heart disease using the loaded model
def heart_prediction(feature_list):
    try:
        feature_array = np.array(feature_list).reshape(1, -1)
        pred_value = modelHeart.predict(feature_array)

        if pred_value[0] == 0:
            return "You Are SAFE"
        elif pred_value[0] == 1:
            return "You Are at Risk"
        else:
            return "Unknown"

    except Exception as e:
        logging.error(f"Error in heart disease prediction: {e}")
        return "Prediction Error"

# Function to predict kidney disease using the loaded model
def kidney_prediction(feature_list):
    try:
        feature_array = np.array(feature_list).reshape(1, -1)
        pred_value = modelKidney.predict(feature_array)

        if pred_value[0] == 0:
            return "You Are SAFE"
        elif pred_value[0] == 1:
            return "You Are at Risk"
        else:
            return "Unknown"

    except Exception as e:
        logging.error(f"Error in kidney disease prediction: {e}")
        return "Prediction Error"

# Route to home page
@app.route('/')
def home():
    return render_template('home.html')

# Route to registration form
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            first_name = request.form['first_name']
            last_name = request.form['last_name']
            address = request.form['address']
            telephone = request.form['telephone']
            email = request.form.get('email')
            age = request.form['age']
            gender = request.form['gender']
            blood_group = request.form['blood_group']
            married = request.form['married']

            serial_key = secrets.token_hex(4)

            new_registration = Registration(
                serial_key=serial_key,
                first_name=first_name,
                last_name=last_name,
                address=address,
                telephone=telephone,
                email=email,
                age=age,
                gender=gender,
                blood_group=blood_group,
                married=married
            )
            db.session.add(new_registration)
            db.session.commit()

            return redirect(url_for('main', 
                                    serial_key=serial_key,
                                    first_name=first_name,
                                    last_name=last_name,
                                    address=address,
                                    telephone=telephone,
                                    email=email,
                                    age=age,
                                    gender=gender,
                                    blood_group=blood_group,
                                    married=married))
        
        except Exception as e:
            logging.error(f"Error in registration: {e}")
            return render_template('error.html', message="An error occurred during registration. Please try again later.")
    
    return render_template('register.html')

# Route to main page after registration
@app.route('/main')
def main():
    serial_key = request.args.get('serial_key')
    first_name = request.args.get('first_name')
    last_name = request.args.get('last_name')
    address = request.args.get('address')
    telephone = request.args.get('telephone')
    email = request.args.get('email')
    age = request.args.get('age')
    gender = request.args.get('gender')
    blood_group = request.args.get('blood_group')
    married = request.args.get('married')

    return render_template('main.html', 
                           serial_key=serial_key,
                           first_name=first_name,
                           last_name=last_name,
                           address=address,
                           telephone=telephone,
                           email=email,
                           age=age,
                           gender=gender,
                           blood_group=blood_group,
                           married=married)

# Route to diabetes prediction form
@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    pred = 0

    if request.method == 'POST':
        try:
            pregnancies = int(request.form['pregnancies'])
            glucose = int(request.form['glucose'])
            blood_pressure = int(request.form['blood_pressure'])
            skin_thickness = int(request.form['skin_thickness'])
            insulin = int(request.form['insulin'])
            bmi = float(request.form['bmi'])
            diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
            age = int(request.form['age'])

            feature_list = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

            pred = diabetes_prediction(feature_list)

        except Exception as e:
            logging.error(f"Error in diabetes form submission: {e}")
            pred = 0 
    
    return render_template('diabetes.html', pred=pred)

# Route to heart disease prediction form
@app.route('/heart', methods=['GET', 'POST'])
def heart():
    pred = 0

    if request.method == 'POST':
        try:
            age = int(request.form['age'])
            gender = 1 if request.form['gender'].lower() == 'male' else 0  # Assuming gender is binary for the model
            cp = float(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = float(request.form['fbs'])
            restecg = float(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = float(request.form['ca'])
            thal = float(request.form['thal'])

            feature_list = [age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

            pred = heart_prediction(feature_list)

        except Exception as e:
            logging.error(f"Error in heart disease form submission: {e}")
            pred = 0
    
    return render_template('heart.html', pred=pred)
# Route to kidney disease prediction form
@app.route('/kidney', methods=['GET', 'POST'])
def kidney():
    pred = 0

    if request.method == 'POST':
        try:
            age = float(request.form['age'])
            blood_pressure = float(request.form['blood_pressure'])
            specific_gravity = float(request.form['specific_gravity'])
            albumin = float(request.form['albumin'])
            sugar = float(request.form['sugar'])
            red_blood_cells = float(request.form['red_blood_cells'])
            pus_cell = float(request.form['pus_cell'])
            pus_cell_clumps = float(request.form['pus_cell_clumps'])
            bacteria = float(request.form['bacteria'])
            blood_glucose_random = float(request.form['blood_glucose_random'])
            blood_urea = float(request.form['blood_urea'])
            serum_creatinine = float(request.form['serum_creatinine'])
            sodium = float(request.form['sodium'])
            potassium = float(request.form['potassium'])
            haemoglobin = float(request.form['haemoglobin'])
            packed_cell_volume = float(request.form['packed_cell_volume'])
            white_blood_cell_count = float(request.form['white_blood_cell_count'])
            red_blood_cell_count = float(request.form['red_blood_cell_count'])
            hypertension = float(request.form['hypertension'])
            diabetes_mellitus = float(request.form['diabetes_mellitus'])
            coronary_artery_disease = float(request.form['coronary_artery_disease'])
            appetite = float(request.form['appetite'])
            peda_edema = float(request.form['peda_edema'])
            aanemia = float(request.form['aanemia'])

            feature_list = [age, blood_pressure, specific_gravity, albumin, sugar, red_blood_cells, pus_cell, pus_cell_clumps,
                            bacteria, blood_glucose_random, blood_urea, serum_creatinine, sodium, potassium, haemoglobin,
                            packed_cell_volume, white_blood_cell_count, red_blood_cell_count, hypertension, diabetes_mellitus,
                            coronary_artery_disease, appetite, peda_edema, aanemia]

            pred = kidney_prediction(feature_list)

        except Exception as e:
            logging.error(f"Error in kidney disease form submission: {e}")
            pred = "Prediction Error"
    
    return render_template('kidney.html', pred=pred)

if __name__ == '__main__':
    app.run(debug=True)

