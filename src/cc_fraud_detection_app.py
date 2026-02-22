from flask import Flask, request, render_template
from pickle import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
import json
import os

# Get project directory
main_dir  = "/workspaces/final-project-csmb20"
proj_dir  = "final-project-csmb20"
model_dir = "models"
templ_dir = "src/templates"
json_dir  = "data/json files"

# Define app
app = Flask(__name__) # app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), templ_dir))

# Load models
fraud_model_path = os.path.join(main_dir,proj_dir,model_dir,"fraud_cc_forest_default_42.sav")
type_model_path  = os.path.join(main_dir,proj_dir,model_dir,"fraud_type_xgb_default_42.sav")

with open(fraud_model_path, "rb") as f:
    model_fraud = load(f)

with open(type_model_path, "rb") as t:
    model_type = load(t)

# Load large input dictionaries    
city_dict_path        = os.path.join(main_dir,proj_dir,json_dir,"json_city_mapping.json")
state_dict_path       = os.path.join(main_dir,proj_dir,json_dir,"json_state_mapping.json")
city_state_dict_path  = os.path.join(main_dir,proj_dir,json_dir,"json_city_state_mapping.json")
city_pop_dict_path    = os.path.join(main_dir,proj_dir,json_dir,"json_city_pop_mapping.json") 
city_lat_dict_path    = os.path.join(main_dir,proj_dir,json_dir,"json_city_lat_mapping.json") 
city_long_dict_path   = os.path.join(main_dir,proj_dir,json_dir,"json_city_long_mapping.json")
merchant_dict_path    = os.path.join(main_dir,proj_dir,json_dir,"json_merchant_mapping.json") 

with open(city_state_dict_path, "r") as cs:
    city_state_mapping_dict = json.load(cs)

with open(city_pop_dict_path, "r") as cp:
    city_pop_mapping_dict = json.load(cp)

with open(city_lat_dict_path, "r") as la:
    city_lat_mapping_dict = json.load(la) 

with open(city_long_dict_path, "r") as lo:
    city_long_mapping_dict = json.load(lo) 

with open(city_dict_path, "r") as c:
    city_mapping_dict = json.load(c)

with open(state_dict_path, "r") as s:
    state_mapping_dict = json.load(s)

with open(merchant_dict_path, "r") as m:
    merchant_mapping_dict = json.load(m)

# Create ouput dictionaries
class_dict_fraud = {
    "0": "Legitimate transaction",
    "1": "Fraudulent transaction",
}

class_dict_type = {
    "0": "POS fraud",
    "1": "Online fraud",
    "2": "Undetermined fraud type"
}

# Define scaler

data_fraud_max = np.array([[379897000000000, 1, 94.470, 23, 3541.740, 13, 459921, 64.756, -69.267]])
data_fraud_min = np.array([[60414207185, 0, 13.469, 0, 1, 0, 47, 26.118, -145.672]])

data_type_max = np.array([[13, 1]])
data_type_min = np.array([[0, 0]])

@app.route("/", methods = ["GET", "POST"])
def detection():
    if request.method == "POST":
        
        val1  = float(request.form["val1"])  # cc_num
        val2  = float(request.form["val2"])  # gender
        val3  = float(request.form["val3"])  # age
        val4  = float(request.form["val4"])  # transaction hour
        val5  = float(request.form["val5"])  # transaction amount
        val6  = float(request.form["val6"])  # category
        val10 = float(request.form["val10"]) # transaction month
        val11 = request.form["val11"]        # city (string)
        val12 = request.form["val12"]        # state
        val13 = float(request.form["val13"]) # merchant

        # Convert data from dictionaries
        val7  = [value for key, value in city_pop_mapping_dict.items() if key == val11]  # city population 
        val8  = [value for key, value in city_lat_mapping_dict.items() if key == val11]  # latitude
        val9  = [value for key, value in city_long_mapping_dict.items() if key == val11] # longitude
        val11 = [value for key, value in city_mapping_dict.items() if key == val11]      # city
        val12 = [value for key, value in state_mapping_dict.items() if key == val12]     # state

        val7  = val7[0][0] 
        val8  = val8[0][0] 
        val9  = val9[0][0]
        val11 = val11[0]
        val12 = val12[0]
        
        # Run fraud model
        
        data_fraud        = np.array([[val1, val2, val3, val4, val5, val6, val7, val8, val9]])
        data_fraud_scaled = (data_fraud - data_fraud_min)/(data_fraud_max - data_fraud_min) #scaler.fit_transform(data_fraud)
        data_fraud_scaled = data_fraud_scaled.tolist()
        prediction_fraud  = str(model_fraud.predict(data_fraud_scaled)[0])
        pred_class_fraud  = class_dict_fraud[prediction_fraud]

        # Run fraud type model

        if prediction_fraud == "1":
            val14     = int(prediction_fraud)
            data_type = np.array([[val6, val14]]) #np.array([[val1, val2, val3, val10, val4, val5, val6, val11, val7, val12, val8, val9, val13, val14]])
            data_type_scaled = (data_type - data_type_min)/(data_type_max - data_type_min)
            data_type_scaled = data_type_scaled.tolist()
            prediction_type = str(model_type.predict(data_type_scaled)[0])
            pred_class_type = class_dict_type[prediction_type]
        else:
            pred_class_type = "No fraud type (legitimate transaction)"

    else:
        pred_class_fraud  = None
        pred_class_type   = None
    
    return render_template("webapp_fraud_cc.html", pred_fraud = pred_class_fraud, pred_type = pred_class_type, city_state_data = city_state_mapping_dict, merchant_data = merchant_mapping_dict)

    if __name__ == "__main__":
        app.run(debug=True, port=8000)