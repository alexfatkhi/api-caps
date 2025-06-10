import json
import sys
import pandas as pd
import joblib
import os

def load_model_files():
    try:
        # Muat model dan label encoder
        model = joblib.load("train1_model_v2.joblib")
        le = joblib.load("label_train_v2.joblib")

        # Muat kolom fitur
        with open("selected_gejala_v2.json", "r") as f:
            feature_columns = json.load(f)

        return model, le, feature_columns
    except Exception as e:
        raise Exception(f"Error loading model files: {str(e)}")

def predict_disease(symptoms, model, le, feature_columns):
    try:
        # Buat kamus input dengan semua fitur diset ke 0
        input_dict = {feat: 0 for feat in feature_columns}

        # Set gejala yang dipilih ke 1
        for symptom in symptoms:
            if symptom in input_dict:
                input_dict[symptom] = 1

        # Buat DataFrame input tanpa nama fitur
        input_vector = [input_dict[col] for col in feature_columns]
        input_df = pd.DataFrame([input_vector])

        # Lakukan prediksi
        predicted_class_index = model.predict(input_df)[0]
        predicted_label = le.inverse_transform([predicted_class_index])[0]

        return predicted_label.upper()
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

def main():
    try:
        # Load model files
        model, le, feature_columns = load_model_files()

        # Read symptoms from temporary file
        with open("temp_input.json", "r") as f:
            symptoms = json.load(f)

        # Make prediction
        prediction = predict_disease(symptoms, model, le, feature_columns)

        # Return result
        result = {"success": True, "prediction": prediction}
        print(json.dumps(result))

    except Exception as e:
        error_result = {"success": False, "error": str(e)}
        print(json.dumps(error_result))
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
