from flask import Flask
from flask import request
import xgboost as xgb
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import os
import datetime

model = pickle.load(open('xgboost.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

features = ['population', 'intersection', 'weather_1', 'state_highway_indicator',
           'tow_away', 'injured_victims', 'party_count', 'pcf_violation_category',
           'hit_and_run', 'type_of_collision', 'motor_vehicle_involved_with',
           'pedestrian_action', 'road_surface', 'road_condition_1', 'lighting',
           'control_device', 'alcohol_involved', 'statewide_vehicle_type_at_fault',
           'chp_vehicle_type_at_fault', 'collision_time', 'party_number',
           'victim_role', 'victim_sex', 'victim_age', 'victim_seating_position',
           'victim_safety_equipment_1', 'victim_ejected', 'party_type', 'at_fault',
           'party_sex', 'party_age', 'party_sobriety', 'direction_of_travel',
           'party_safety_equipment_1', 'financial_responsibility', 'cellphone_use',
           'other_associate_factor_1', 'movement_preceding_collision',
           'vehicle_year', 'statewide_vehicle_type', 'party_race', 'DOW', 'month',
           'year']

selected_features = ['year',
                     'month',
                     'DOW',
                     'collision_time',
                     'type_of_collision',
                     'injured_victims',
                     'victim_role',
                     'victim_age',
                     'hit_and_run',
                     'cellphone_use',
                     'alcohol_involved',
                     'population']

other_features = [feature for feature in features if feature not in selected_features]

default_features = {'population_Rural': 0,
                     'state_highway_indicator_Yes': 0,
                     'tow_away_Yes': 0,
                     'pcf_violation_category_dui': 0,
                     'pcf_violation_category_pedestrian violation': 0,
                     'pcf_violation_category_speeding': 0,
                     'pcf_violation_category_wrong side of road': 0,
                     'hit_and_run_misdemeanor': 0,
                     'type_of_collision_head-on': 0,
                     'type_of_collision_hit object': 0,
                     'type_of_collision_overturned': 0,
                     'type_of_collision_pedestrian': 0,
                     'type_of_collision_rear end': 0,
                     'type_of_collision_sideswipe': 0,
                     'motor_vehicle_involved_with_fixed object': 0,
                     'motor_vehicle_involved_with_other motor vehicle': 0,
                     'motor_vehicle_involved_with_pedestrian': 1,
                     'pedestrian_action_crossing not in crosswalk': 0,
                     'pedestrian_action_in road': 0,
                     'lighting_daylight': 1,
                     'statewide_vehicle_type_at_fault_motorcycle or scooter': 0,
                     'statewide_vehicle_type_at_fault_pedestrian': 0,
                     'chp_vehicle_type_at_fault_cars': 1,
                     'chp_vehicle_type_at_fault_motors': 0,
                     'chp_vehicle_type_at_fault_pedestrians': 0,
                     'victim_role_Driver': 0,
                     'victim_role_Non-Injured': 0,
                     'victim_role_Passenger': 0,
                     'victim_role_Pedestrian': 0,
                     'victim_sex_male': 0,
                     'victim_sex_unknown': 0,
                     'victim_seating_position_2': 0,
                     'victim_seating_position_4': 0,
                     'victim_seating_position_5': 0,
                     'victim_seating_position_6': 0,
                     'victim_seating_position_7': 0,
                     'victim_safety_equipment_1_G': 0,
                     'victim_safety_equipment_1_L': 0,
                     'victim_safety_equipment_1_M': 0,
                     'victim_safety_equipment_1_P': 0,
                     'victim_safety_equipment_1_other': 1,
                     'victim_ejected_Not_Ejected': 0,
                     'victim_ejected_Partially_Ejected': 0,
                     'party_type_driver': 0,
                     'party_sex_male': 0,
                     'party_sobriety_C': 0,
                     'party_sobriety_H': 0,
                     'party_safety_equipment_1_L': 0,
                     'party_safety_equipment_1_M': 0,
                     'party_safety_equipment_1_P': 0,
                     'party_safety_equipment_1_other': 1,
                     'financial_responsibility_N': 0,
                     'financial_responsibility_Y': 0,
                     'cellphone_use_C': 0,
                     'movement_preceding_collision_slowing/stopping': 0,
                     'movement_preceding_collision_stopped': 0,
                     'statewide_vehicle_type_emergency vehicle': 0,
                     'statewide_vehicle_type_other bus': 0,
                     'statewide_vehicle_type_passenger car with trailer': 0,
                     'statewide_vehicle_type_pickup or panel truck': 0,
                     'at_fault': 0,
                     'vehicle_year': 2004.0,
                     'year': 2013,
                     'party_number': 2,
                     'DOW': 3,
                     'collision_time': 4.0,
                     'party_count': 2.0,
                     'alcohol_involved': 0.0,
                     'injured_victims': 1.0,
                     'victim_age': 74.0,
                     'party_age': 74.0,
                     'month': 4}

classes = {0: 'Not injured', 1: 'Light injury', 2: 'Severe injury', 3: 'Killed'}


app = Flask(__name__)

@app.route('/')
def index():
    html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>California Traffic Collision Prediction Model</title>
                <style>
                h1 {text-align: center;}
                p {text-align: center; font-size: 20px;}
                h2 {text-align: center;}
                h3 {text-align: center;}
                img {
                  display: block;
                  margin-left: auto;
                  margin-right: auto;
                }
                * {
                  box-sizing: border-box;
                }
                .column {
                  float: left;
                  width: 33.33%;
                  padding: 5px;
                }
                /* Clearfix (clear floats) */
                .row::after {
                  content: "";
                  clear: both;
                  display: table;
                }
                .collapsible {
                  background-color: #777;
                  color: white;
                  cursor: pointer;
                  padding: 18px;
                  width: 100%;
                  border: none;
                  text-align: left;
                  outline: none;
                  font-size: 15px;
                }
                .active, .collapsible:hover {
                  background-color: #555;
                }
                .content {
                  padding: 0 18px;
                  display: none;
                  overflow: hidden;
                  background-color: #f1f1f1;
                }
                </style>
            </head>
            <body>
            <header class="w3-container w3-center w3-padding-32"> 
              <h1><b>California Traffic Collision Prediction Model</b></h1>
              <p>The Statewide Integrated Traffic Records System (SWITRS) is collected and maintained by the 
                California Highway Patrol (CHP).<br>
                SWITRS contains all collisions that were reported to CHP from local and governmental agencies.<br>
            </p>
            </header>
            <img src="https://www.azizilawfirm.com/wp-content/uploads/2017/05/report-a-car-accident-float.jpg" alt="collisions map">
            <h2>
                <a href="https://www.itc.tech/">An ITC Data Science project</a>
                <br>Participants: Ariana Gordon, Eran Perelman, Eyal Hashimshoni and Royi Razi<br>
            </h2>
            <p>Our goal is to provide support for emergency call operators in prioritizing dispatch 
            of first responders to motor vehicle accident sites in an effort to save lives. 
            <br>Our model will predict severity of injuries so the column titled 'victim_degree_of_injury' 
            will be our target variable.</p>
            <h3>Sources:<br>
                <br><a href="https://www.kaggle.com/alexgude/california-traffic-collision-data-from-switrs">
                California Traffic Collision Data from SWITRS (on Kaggle)</a><br>
                <br>
                <a href="https://tims.berkeley.edu/help/SWITRS.php">
                The Statewide Integrated Traffic Records System</a><br>
            </h3>
            <h2>
                Collision form<br>
            </h2>
            <form action="/predict_single_ui">
            '''
    html += get_features_table()
    html += '''
        <h3><input type="submit" value="Submit"></h3>
        </form>
        <p>Click the "Submit" button when finished.</p>
        <a href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">Premium content</a>
        </body>
        </html>
        '''
    return html


@app.route("/predict_single_ui")
def predict_single_ui():
    example_features = default_features
    dict_for_prediction = request.args.to_dict(flat=False)
    
    # date
    year = int(dict_for_prediction['date'][0][:4])
    month = int(dict_for_prediction['date'][0][5:7])
    day = int(dict_for_prediction['date'][0][8:10])
    DOW = datetime.date(day=day, month=month, year=year).weekday()
    example_features['year'] = year
    example_features['month'] = month
    example_features['DOW'] = DOW
    
    # 'collision_time'
    collision_time = int(dict_for_prediction['appt'][0][:2])
    example_features['collision_time'] = collision_time
    
    # type_of_collision
    if dict_for_prediction['type_of_collision'][0] == 'A':
        example_features['type_of_collision_head-on'] = 1
    elif dict_for_prediction['type_of_collision'][0] == 'B':
        example_features['type_of_collision_sideswipe'] = 1
    elif dict_for_prediction['type_of_collision'][0] == 'C':
        example_features['type_of_collision_rear end'] = 1
    elif dict_for_prediction['type_of_collision'][0] == 'E':
        example_features['type_of_collision_hit object'] = 1
    elif dict_for_prediction['type_of_collision'][0] == 'F':
        example_features['type_of_collision_overturned'] = 1
    elif dict_for_prediction['type_of_collision'][0] == 'G': 
        example_features['type_of_collision_pedestrian'] = 1
    
    # injured_victims
    example_features['injured_victims'] = int(dict_for_prediction['injured_victims'][0])
    
    # victim_role
    if dict_for_prediction['victim_role'][0] == 'Driver':
        example_features['victim_role_Driver'] = 1
    if dict_for_prediction['victim_role'][0] == 'Non-Injured':
        example_features['victim_role_Non-Injured'] = 1
    if dict_for_prediction['victim_role'][0] == 'Passenger':
        example_features['victim_role_Passenger'] = 1
    if dict_for_prediction['victim_role'][0] == 'Pedestrian':
        example_features['victim_role_Pedestrian'] = 1
        
    # victim_age
    example_features['victim_age'] = int(dict_for_prediction['victim_age'][0])
    
    # population
    if 'population' in dict_for_prediction:
        example_features['population_Rural'] = 1
    
    # hit_and_run
    if 'hit_and_run' in dict_for_prediction:
        example_features['hit_and_run_misdemeanor'] = 1
    
    if 'cellphone_use' in dict_for_prediction:
        example_features['cellphone_use_C'] = 1
    
    # alcohol_involved
    if 'alcohol_involved' in dict_for_prediction:
        example_features['alcohol_involved'] = 1
    
    probs = get_prediction(example_features)
    prediction = np.argmax(probs)
    print(probs)
    
    html = '''
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
            h1 {
                  text-align: center;
                }
            img {
              display: block;
              margin-left: auto;
              margin-right: auto;
            }
            
        body, table, input, select, textarea {
        }
.graph {
margin-bottom:1em;
  font:normal 100%/150% arial,helvetica,sans-serif;
   display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%;
}
.graph caption {
font:bold 150%/120% arial,helvetica,sans-serif;
padding-bottom:0.33em;
}
.graph tbody th {
text-align:right;
}
@supports (display:grid) {
@media (min-width:32em) {
.graph {
display:block;
      width:600px;
      height:300px;
		}
		.graph caption {
			display:block;
		}
		.graph thead {
			display:none;
		}
		.graph tbody {
			position:relative;
			display:grid;
			grid-template-columns:repeat(auto-fit, minmax(2em, 1fr));
			column-gap:2.5%;
			align-items:end;
			height:100%;
			margin:3em 0 1em 2.8em;
			padding:0 1em;
			border-bottom:2px solid rgba(0,0,0,0.5);
			background:repeating-linear-gradient(
				180deg,
				rgba(170,170,170,0.7) 0,
				rgba(170,170,170,0.7) 1px,
				transparent 1px,
				transparent 20%
			);
		}
		.graph tbody:before,
		.graph tbody:after {
			position:absolute;
			left:-3.2em;
			width:2.8em;
			text-align:right;
			font:bold 80%/120% arial,helvetica,sans-serif;
		}
		.graph tbody:before {
			content:"100%";
			top:-0.6em;
		}
		.graph tbody:after {
			content:"0%";
			bottom:-0.6em;
		}
		.graph tr {
			position:relative;
			display:block;
		}
		.graph tr:hover {
			z-index:999;
		}
		.graph th,
		.graph td {
			display:block;
			text-align:center;
		}
		.graph tbody th {
			position:absolute;
			top:-3em;
			left:0;
			width:100%;
			font-weight:normal;
			text-align:center;
      white-space:nowrap;
			text-indent:0;
			transform:rotate(-45deg);
		}
		.graph tbody th:after {
			content:"";
		}
		.graph td {
			width:100%;
			height:100%;
			background:#F63;
			border-radius:0.5em 0.5em 0 0;
			transition:background 0.5s;
		}
		.graph tr:hover td {
			opacity:0.7;
		}
		.graph td span {
			overflow:hidden;
			position:absolute;
			left:50%;
			top:50%;
			width:0;
			padding:0.5em 0;
			margin:-1em 0 0;
			font:normal 85%/120% arial,helvetica,sans-serif;
/* 			background:white; */
/* 			box-shadow:0 0 0.25em rgba(0,0,0,0.6); */
			font-weight:bold;
opacity:0;
transition:opacity 0.5s;
      color:white;
}
.toggleGraph:checked + table td span,
.graph tr:hover td span {
width:4em;
margin-left:-2em; /* 1/2 the declared width */
opacity:1;
}
} /* min-width:32em */
} /* grid only */
            </style>
            </head>
            <body>
            '''
    html += '<h1>'
    html += classes[prediction]
    html += '</h1>'
    html += '<img src="'
    if prediction == 0:
        html += 'https://i.ibb.co/d05kByC/image.jpg'
    elif prediction == 1:
        html += 'https://i.ibb.co/n1XPsZP/1.jpg'
    elif prediction == 2:
        html += 'https://i.ibb.co/GQXJtp9/2.jpg'
    elif prediction == 3:
        html += 'https://i.ibb.co/LCXqC4X/3.jpg'
    # html += str(prediction) + '.jpg'
    html += '" alt="'
    html += classes[prediction]
    html += '" style="width:50%;">'
    html += '<br>'
    html += '''
        <table class="graph">
        <caption>Injury probability</caption>
        <thead>
            <tr>
                <th scope="col">Item</th>
                <th scope="col">Percent</th>
            </tr>
        </thead><tbody>
        '''
    html += '<tr style="height:' + str(100 * float(probs[0])) + '%">'
    html += '''    
            <th scope="row">Not injured</th>
            <td><span>
            '''
    html += str(round(100 * float(probs[0]), 2))
    html += '''      
            %</span></td>
            </tr>
            '''
    html += '<tr style="height:' + str(100 * float(probs[1])) + '%">'      
    html += '''         
            <th scope="row">Light injury</th>
            <td><span>
             '''
    html += str(round(100 * float(probs[1]), 2))        
    html += '''          
            %</span></td>
            </tr>
             '''
    html += '<tr style="height:' + str(100 * float(probs[2])) + '%">'      
    html += '''         
            <th scope="row">Severe injury</th>
            <td><span>
            '''
    html += str(round(100 * float(probs[2]), 2))       
    html += '''        
            %</span></td>
            </tr>
             '''
    html += '<tr style="height:' + str(100 * float(probs[3])) + '%">'      
    html += '''     
            <th scope="row">Killed</th>
            <td><span>
            '''
    html += str(round(100 * float(probs[3]), 2))        
    html += '''             
            %</span></td>
            </tr>
            </tbody>
            </table>
            '''
    html += '</body>'
    html += '</html>'
    return html


@app.route("/api", methods=["POST"])
def multiple_predictions():
    # Validate the request body contains JSON
    if request.is_json:
        req = request.get_json()
        answer = get_answer(req)
        json_pred = json.dumps(answer)
        
        return json_pred, 200

    else:
        return "Request was not JSON", 400


def get_features_table():
    # date features
    html = '''
            <table style="width:100%">
            <tr>
            '''
    
    html += '<th>'
    html += 'The date when the collision occurred: '
    html += '<input type="date" id="date" name="date" value="2020-10-18">'
    html += '</th>'
    
    # collision time feature
    html += '<th>'
    html += 'The time when the collision occurred (24 hour time): '
    html += '<input type="time" id="appt" name="appt" value="09:30">'
    html += '</th>'
    
    # Type of Collision feature
    html += '<th>'
    html += '<label for="type_of_collision">Type of Collision: </label>'
    html += '<select name="type_of_collision" id="type_of_collision">'
    html += '<option value="A">Head-On</option>'
    html += '<option value="B">Sideswipe</option>'
    html += '<option value="C">Rear End</option>'
    html += '<option value="D">Broadside</option>'
    html += '<option value="E">Hit Object</option>'
    html += '<option value="F">Overturned</option>'
    html += '<option value="G">Vehicle/Pedestrian</option>'
    html += '<option value="H">Other</option>'
    html += '</select>'
    html += '</th>'

    
    # Injured victims feature
    html += '<th>'
    html += '<label for="injured_victims">Injured victims: </label>'
    html += '<input type="number" id="injured_victims" name="injured_victims" min="0" max="150" value="2">'
    html += '</th>'
    
    # Victim Role feature
    html += '<th>'
    html += '<label for="victim_role">Victim Role: </label>'
    html += '<select name="victim_role" id="victim_role">'
    html += '<option value="Driver">Driver</option>'
    html += '<option value="Passenger">Passenger</option>'
    html += '<option value="Pedestrian">Pedestrian</option>'
    html += '<option value="Bicyclist">Bicyclist</option>'
    html += '<option value="Non-Injured">Non-Injured</option>'
    html += '<option value="Other">Other</option>'
    html += '</select>'
    html += '</th>'
    
    # Victim Age feature
    html += '<th>'
    html += '<label for="victim_age">Victim Age: </label>'
    html += '<input type="number" id="victim_age" name="victim_age" min="0" max="125" value="36">'
    html += '</th>'
    
    """
    for feature in selected_features:
        html += '<th>'
        html += feature
        html += '<label for="'
        html += feature
        html += '">'
        html += '</label>'       
        html += '</th>'
    html += '</tr><tr>'
    for feature in selected_features:
        html += '<th><input type="text" id="'
        html += feature
        html += '" name="'
        html += feature
        html += '" ></th>'
    """
    html += '</tr></table><br>'
    
    # Other features
    html += '<button type="button" class="collapsible">Other features</button>'
    html += '<div class="content">'
    # html += '<p>This is other features section</p>'
    
    # Population rural feature
    html += '<th>'
    html += '<input type="checkbox" id="population" name="population" value="yes">'
    html += '<label for="population">Rural population</label><br>'
    html += '</th>'
    
    # Hit And Run feature
    html += '<th>'
    html += '<input type="checkbox" id="hit_and_run" name="hit_and_run" value="yes">'
    html += '<label for="hit_and_run">Hit And Run</label><br>'
    html += '</th>'
    
    # Cellphone use feature
    html += '<th>'
    html += '<input type="checkbox" id="cellphone_use" name="cellphone_use" value="yes">'
    html += '<label for="cellphone_use">Cellphone use</label><br>'
    html += '</th>'
    
    # Alcohol Involved feature
    html += '<th>'
    html += '<input type="checkbox" id="alcohol_involved" name="alcohol_involved" value="yes">'
    html += '<label for="alcohol_involved">Alcohol Involved</label><br>'
    html += '</th>'
    html += '</div>'

    html += '<script>'
    html += 'var coll = document.getElementsByClassName("collapsible");'
    html += 'var i;'

    html += 'for (i = 0; i < coll.length; i++) {'
    html += '  coll[i].addEventListener("click", function() {'
    html += '    this.classList.toggle("active");'
    html += '    var content = this.nextElementSibling;'
    html += '    if (content.style.display === "block") {'
    html += '      content.style.display = "none";'
    html += '    } else {'
    html += '      content.style.display = "block";'
    html += '    }'
    html += '  });'
    html += '}'
    html += '</script>'

    return html


# Get a request and return multiple predictions
def get_predictions_json(req):
    predictions = []
    for r in req:
        predictions.append(get_prediction(r))
    return predictions


# get one prediction from the model and return a prediction (probabily)
def get_prediction(dict_for_prediction):
    # convert dictionary to a list
    list_to_predict = np.array(list(dict_for_prediction.values())).reshape(1, -1)
    
    # Scale the list using the scaler
    scaled_list = scaler.transform(list_to_predict)   
    
    # predict with the XGBoost model
    dpred = xgb.DMatrix(data=scaled_list)
    pred_xgb = model.predict(dpred)
    return pred_xgb[0]


def main():
    app.run()


if __name__ == '__main__':
    port = os.environ.get('PORT')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
