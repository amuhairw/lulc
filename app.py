!pip install geemap -qqq

import geemap
import ee
# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()

lat, lon = -1.9441, 30.069
offset = 0.11

corners = [
    [lon+offset, lat+offset], # bottom up right
    [lon+offset, lat-offset], # top right
    [lon-offset, lat-offset], # top left
    [lon-offset, lat+offset]] # bottom left
region = ee.Geometry.Polygon(corners)

center = region.centroid()
center.getInfo()

center_list = center.getInfo()["coordinates"]
Map = geemap.Map()
Map.centerObject(center, 14)
Map

countryData = ee.FeatureCollection("TIGER/2018/Countries")
print(type(countryData))

countryData

state_nine=countryData.filter(ee.Filter.eq("STATEFP","09"))

import numpy as np
a=np.array(state_nine.getInfo())

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, figsize=(5,5))
sns.heatmap(trainCM/trainCM.sum(axis=1), annot=True)
ax.set_xlabel('model predictions', fontsize=20)
ax.set_ylabel('actual', fontsize=20)
plt.title("Training data confusion matrix", fontsize=20)

#80, 90, 10
# TODO: complete the following codes
train_exercise =data.filter(ee.Filter.lt('random', 0.8))
val_exercise = data.filter(ee.Filter.And(ee.Filter.gte('random', 0.8),ee.Filter.lt('random', 0.9)))
test_exercise = data.filter(ee.Filter.gte('random', 0.9))

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # A linear layer of 500, 400
        self.fc1 = nn.Linear(500, 400)
        # A relu activation
        self.relu1 = nn.ReLU()
        # A linear layer of 400, 100
        self.fc2 = nn.Linear(400, 100)
        # A relu activation
        self.relu2 = nn.ReLU()
        # A linear layer of 100, 2
        self.fc3 = nn.Linear(100, 2)
        # A sigmoid activation layer
        # self.sigmoid =
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            # x = self.sigmoid(x)
            return x

import torch.nn as nn
torch.manual_seed(42)
# Answer
tensor_input = torch.zeros((9,32,32)).unsqueeze(0).float().to(device)
out = model(tensor_input)

print("Output is: ", out)

import ee

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

# Initialize an app
app = Flask(__name__)

# Load the serialized model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize GEE
service_account = 'ml4eo-420815.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, '.private-key.json')
ee.Initialize(credentials)

# Initialize variables required for GEE dataset preprocessing (similar to the examples in Exercise 6_1)
lat = -1.9441
lon = 30.0619
offset = 0.51
region = [
    [lon + offset, lat - offset],
    [lon + offset, lat + offset],
    [lon - offset, lat + offset],
    [lon - offset, lat - offset]]

roi = ee.Geometry.Polygon([region])

se2bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A']
trainingbands = se2bands + ['avg_rad']
label = 'smod_code'
scaleFactor = 1000

# Remember this function from Exercise 5_03, what does it do?
def se2mask(image):
    quality_band = image.select('QA60')
    cloudmask = 1 << 10
    cirrusmask = 1 << 11
    mask = quality_band.bitwiseAnd(cloudmask).eq(0) & (quality_band.bitwiseAnd(cirrusmask).eq(0))
    return image.updateMask(mask).divide(10000)

def get_fused_data():
    """
    This function contains the preprocessing steps followed to obtain the preprocessed, merged dataset in 6_1.
    This function is called when the server starts to prepare the dataset.
    """
    mean = 0.2062830612359614
    std = 1.1950717918110398

    # Convert the mean and std to ee.Number
    vmu = ee.Number(mean)
    vstd = ee.Number(std)

    # Load the COPERNICUS/S2 dataset and filter dates "2015-07-01" to "2015-12-31"
    se2 = ee.ImageCollection("COPERNICUS/S2").filterDate("2015-07-01", "2015-12-31")
    # Use the filterBounds function to filter the area specified in ROI
    se2 = se2.filterBounds(roi)

    # Keep pixels that have less than 20% cloud
    se2 = se2.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))

    # Update the mask
    se2 = se2.map(se2mask)

    # Get the median image
    se2 = se2.median()

    # Select the `se2bands`
    se2 = se2.select(se2bands)

    # Load the NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG dataset and filter dates "2015-07-01" to "2015-12-31"
    viirs = ee.Image(ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filterDate(
        "2015-07-01", "2015-12-31").filterBounds(roi).median().select('avg_rad').clip(roi))

    # Substract the mean and divide by the standard deviation for the viirs samples
    viirsclean = viirs.subtract(vmu).divide(vstd)

    # Fuse the two datasets
    fusedclean = se2.addBands(viirsclean)

    return fusedclean

    # Prepare the fused data
    gee_data = get_fused_data()

def get_features(longitude, latitude):
    # Create an ee.Geometry instance from the coordinates
    poi_geometry = ee.Geometry.Point([longitude, latitude])

    # Sample features for the given point of interest, keeping only the training bands
    dataclean = gee_data.select(trainingbands).sampleRegions(collection= poi_geometry, properties=[label], scale=scaleFactor)


    # Use getInfo to load the sample's features
    sample = dataclean.getInfo()

    # Find the band ordering in the loaded data
    band_order = sample['properties']['band_order']

    # Convert the loaded data to ee.List
    nested_list = dataclean.reduceColumns(ee.Reducer.toList(len(band_order)), band_order).values().get(0)

    # Convert the `nested_list` to a Pandas dataframe
    data = pd.DataFrame(nested_list.getInfo(), columns=band_order)
    return data

# function to check if point is withing the roi
def validate_location(longitude, latitude, roi):
    point = ee.Geometry.Point(longitude, latitude)
    return roi.contains(point).getInfo()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.to_dict()
    longitude = float(features['longitude'])
    latitude = float(features['latitude'])
    # Get the features for the given location
    final_features = get_features(longitude, latitude)

if validate_location(longitude, latitude, roi):
    # TODO: get the features for the given location
    final_features = get_features(longitude, latitude)

    # Get predictions from the model using the loaded features
    prediction = model.predict(final_features)

    # Convert the prediction to an integer
    output = int(prediction[0])

    if output == 1:
        text = "built up land"
    else:
        text = "not built up land"

    # Return a response based on the output of the model
    return render_template('index.html', prediction_text='The area at {}, {} location is {}'.format(longitude, latitude, text))
else:
    return render_template('index.html', prediction_text='The area at {}, {} location is is out of bounds.'.format(longitude, latitude))

if __name__ == "__main__":
    app.run(debug=True)
