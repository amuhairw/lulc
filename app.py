import ee
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Initialize an app
app = Flask(__name__)

# Load the serialized model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize GEE
ee.Initialize()

# Define the region of interest (ROI)
roi = ee.Geometry.Rectangle([-1.995, 29.997, -1.893, 30.141])

def get_features(longitude, latitude):
    # Create an ee.Geometry instance from the coordinates
    poi_geometry = ee.Geometry.Point(longitude, latitude)

    # Sample features for the given point of interest
    dataclean = fusedclean.sampleRegions(collection=poi_geometry,
                                         properties=[label],
                                         scale=scaleFactor)

    # Load the sample's features
    sample = dataclean.getInfo()

    # Find the band ordering in the loaded data
    band_order = sample['properties']['band_order']

    # Convert the loaded data to ee.List
    nested_list = dataclean.reduceColumns(ee.Reducer.toList(len(band_order)), band_order).values().get(0)

    # Convert the nested_list to a Pandas dataframe
    data = pd.DataFrame(nested_list.getInfo(), columns=band_order)
    return data

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    longitude = float(features['longitude'])
    latitude = float(features['latitude'])
    
    if roi.contains(ee.Geometry.Point(longitude, latitude)).getInfo():
        # Get the features for the given location
        final_features = get_features(longitude, latitude)
        
        # Get predictions from the model using the features
        prediction = model.predict(final_features)
        
        # Convert the prediction to an integer
        output = int(prediction[0])
        
        if output == 1:
            text = "built-up land"
        else:
            text = "not built-up land"
        
        return render_template('index.html',
                               prediction_text='The area at {}, {} is classified as {}'.format(longitude, latitude, text))
    else:
        return render_template('index.html',
                               prediction_text='The area at {}, {} is out of bounds.'.format(longitude, latitude))

if __name__ == "__main__":
    app.run(debug=True)
