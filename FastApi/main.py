from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os
app = FastAPI()

MODEL_PATH = "models/catboost_model_updated_Final.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

class RawFeatures(BaseModel):
    Age_of_the_patient: float = Field(..., alias="Age_of_the_patient")
    Specific_gravity_of_urine: float = Field(..., alias="Specific_gravity_of_urine")
    Random_blood_glucose_level_mg_dl: float = Field(..., alias="Random_blood_glucose_level_mg_dl")
    Blood_urea_mg_dl: float = Field(..., alias="Blood_urea_mg_dl")
    Serum_creatinine_mg_dl: float = Field(..., alias="Serum_creatinine_mg_dl")
    Sodium_level_mEq_L: float = Field(..., alias="Sodium_level_mEq_L")
    Potassium_level_mEq_L: float = Field(..., alias="Potassium_level_mEq_L")
    Hemoglobin_level_gms: float = Field(..., alias="Hemoglobin_level_gms")
    White_blood_cell_count_cells_cumm: float = Field(..., alias="White_blood_cell_count_cells_cumm")
    Red_blood_cell_count_millions_cumm: float = Field(..., alias="Red_blood_cell_count_millions_cumm")
    Estimated_Glomerular_Filtration_Rate_eGFR: float = Field(..., alias="Estimated_Glomerular_Filtration_Rate_eGFR")
    Urine_protein_to_creatinine_ratio: float = Field(..., alias="Urine_protein_to_creatinine_ratio")
    Urine_output_ml_day: float = Field(..., alias="Urine_output_ml_day")
    Serum_albumin_level: float = Field(..., alias="Serum_albumin_level")
    Parathyroid_hormone_PTH_level: float = Field(..., alias="Parathyroid_hormone_PTH_level")
    Serum_calcium_level: float = Field(..., alias="Serum_calcium_level")
    Duration_of_hypertension_years: float = Field(..., alias="Duration_of_hypertension_years")
    Interleukin_6_IL6_level: float = Field(..., alias="Interleukin_6_IL6_level")
    Family_history_of_chronic_kidney_disease: float = Field(..., alias="Family_history_of_chronic_kidney_disease")
    Serum_phosphate_level: float = Field(..., alias="Serum_phosphate_level")
    Pus_cells_in_urine: float = Field(..., alias="Pus_cells_in_urine")
    Albumin_in_urine: float = Field(..., alias="Albumin_in_urine")
    Anemia_yes_no: str = Field(..., alias="Anemia_yes_no")

    class Config:
        allow_population_by_field_name = True


@app.post("/predict")
def predict(data: RawFeatures):
    try:
      
        creatinine_eGFR_ratio = data.Serum_creatinine_mg_dl / (data.Estimated_Glomerular_Filtration_Rate_eGFR + 1e-6)
        age_hypertension_ratio = data.Age_of_the_patient / (data.Duration_of_hypertension_years + 1)
        serum_albumin_adjusted = data.Serum_albumin_level / (data.Urine_output_ml_day + 1)
        total_blood_count = data.White_blood_cell_count_cells_cumm + data.Red_blood_cell_count_millions_cumm
        blood_creatinine_ratio = data.Blood_urea_mg_dl / (data.Serum_creatinine_mg_dl + 1e-6)
        eGFR_flag = 1 if data.Estimated_Glomerular_Filtration_Rate_eGFR < 60 else 0
        albumin_stress_index = data.Albumin_in_urine / (data.Serum_creatinine_mg_dl + 1)
        hypertension_burden = data.Duration_of_hypertension_years / (data.Age_of_the_patient + 1)
        electrolyte_imbalance = abs(data.Sodium_level_mEq_L - data.Potassium_level_mEq_L)
        blood_toxicity_score = data.Blood_urea_mg_dl * 0.6 + data.Serum_creatinine_mg_dl * 0.4
        inflammation_ratio = data.Interleukin_6_IL6_level / (data.Serum_albumin_level + 1)

      
        input_vector = [
            data.Age_of_the_patient,
            data.Specific_gravity_of_urine,
            data.Random_blood_glucose_level_mg_dl,
            data.Blood_urea_mg_dl,
            data.Serum_creatinine_mg_dl,
            data.Sodium_level_mEq_L,
            data.Potassium_level_mEq_L,
            data.Hemoglobin_level_gms,
            data.White_blood_cell_count_cells_cumm,
            data.Red_blood_cell_count_millions_cumm,
            data.Estimated_Glomerular_Filtration_Rate_eGFR,
            data.Urine_protein_to_creatinine_ratio,
            data.Urine_output_ml_day,
            data.Serum_albumin_level,
            data.Parathyroid_hormone_PTH_level,
            data.Serum_calcium_level,
            data.Duration_of_hypertension_years,
            data.Interleukin_6_IL6_level,
            data.Family_history_of_chronic_kidney_disease,
            data.Serum_phosphate_level,
            data.Pus_cells_in_urine,
            data.Albumin_in_urine,
            1 if data.Anemia_yes_no.lower() == "yes" else 0,
            creatinine_eGFR_ratio,
            age_hypertension_ratio,
            serum_albumin_adjusted,
            total_blood_count,
            blood_creatinine_ratio,
            eGFR_flag,
            albumin_stress_index,
            hypertension_burden,
            electrolyte_imbalance,
            blood_toxicity_score,
            inflammation_ratio,
        ]

    
        print("Input vector for prediction:", input_vector) 
        prediction = model.predict([input_vector])[0]
        return {"prediction": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
