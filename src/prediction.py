import pandas as pd

class Predictor:
    def __init__(self, models, scaler, label_encoders, feature_names):
        self.models = models
        self.scaler = scaler
        self.label_encoders = label_encoders
        self.feature_names = feature_names
        
    def predict_new_patient(self, patient_data):
        patient_df = pd.DataFrame([patient_data])
        
        for col, encoder in self.label_encoders.items():
            if col in patient_df.columns:
                patient_df[col] = encoder.transform(patient_df[col])
                
        for feature in self.feature_names:
            if feature not in patient_df.columns:
                patient_df[feature] = 0
                
        patient_df = patient_df[self.feature_names]
        patient_scaled = self.scaler.transform(patient_df)
        
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict(patient_scaled)[0]
            pred_proba = model.predict_proba(patient_scaled)[0]
            predictions[name] = {
                'prediction': pred,
                'probability': pred_proba[1] if pred == 1 else pred_proba[0]
            }
            
        return predictions
