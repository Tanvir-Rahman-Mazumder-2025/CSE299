const express = require('express');
const app = express();
const bodyparser = require('body-parser');
const fs = require('fs');
const path = require("path");
const ejs = require("ejs");
const db = require('./config/db')
const user = require("./model/user")
const joi = require('joi')
const Post = require('./model/post');
const post = require('./model/post');
const axios = require('axios')

app.set('view engine', 'ejs');

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
const directoryPath = path.join(__dirname, 'files');




app.get("/",(req,res)=>{
    res.render("ckd.ejs");
})


app.post('/predict', async (req, res) => {
  try {
    
    const rawFeatures = {
      Age_of_the_patient: parseFloat(req.body.Age_of_the_patient),
      Specific_gravity_of_urine: parseFloat(req.body.Specific_gravity_of_urine),
      Random_blood_glucose_level_mg_dl: parseFloat(req.body.Random_blood_glucose_level_mg_dl),
      Blood_urea_mg_dl: parseFloat(req.body.Blood_urea_mg_dl),
      Serum_creatinine_mg_dl: parseFloat(req.body.Serum_creatinine_mg_dl),
      Sodium_level_mEq_L: parseFloat(req.body.Sodium_level_mEq_L),
      Potassium_level_mEq_L: parseFloat(req.body.Potassium_level_mEq_L),
      Hemoglobin_level_gms: parseFloat(req.body.Hemoglobin_level_gms),
      White_blood_cell_count_cells_cumm: parseFloat(req.body.White_blood_cell_count_cells_cumm),
      Red_blood_cell_count_millions_cumm: parseFloat(req.body.Red_blood_cell_count_millions_cumm),
      Estimated_Glomerular_Filtration_Rate_eGFR: parseFloat(req.body.Estimated_Glomerular_Filtration_Rate_eGFR),
      Urine_protein_to_creatinine_ratio: parseFloat(req.body.Urine_protein_to_creatinine_ratio),
      Urine_output_ml_day: parseFloat(req.body.Urine_output_ml_day),
      Serum_albumin_level: parseFloat(req.body.Serum_albumin_level),
      Parathyroid_hormone_PTH_level: parseFloat(req.body.Parathyroid_hormone_PTH_level),
      Serum_calcium_level: parseFloat(req.body.Serum_calcium_level),
      Duration_of_hypertension_years: parseFloat(req.body.Duration_of_hypertension_years),
      Interleukin_6_IL6_level: parseFloat(req.body.Interleukin_6_IL6_level),
      Family_history_of_chronic_kidney_disease: parseFloat(req.body.Family_history_of_chronic_kidney_disease),
      Serum_phosphate_level: parseFloat(req.body.Serum_phosphate_level),
      Pus_cells_in_urine: parseFloat(req.body.Pus_cells_in_urine),
      Albumin_in_urine: parseFloat(req.body.Albumin_in_urine),
      Anemia_yes_no: req.body.Anemia_yes_no
    };

  
    const response = await axios.post('http://localhost:8000/predict', rawFeatures);

   
  res.render('result', { prediction: response.data.prediction });

  } catch (error) {
    console.error(error);
    res.status(500).send("Error communicating with prediction API");
  }
});

app.listen(3000);

