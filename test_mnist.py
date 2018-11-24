# import pkgs
import argparse
import pandas as pd
import numpy as np
import os

# setup and handle switches
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data', required=True, help='path to test split')
ap.add_argument('-m', '--model', required=True, help='path to trained model')
args = vars(ap.parse_args())

# conv var
testFile = args['data']

# Load data
data = pd.read_csv(testFile, skiprows=1)
print(data.shape)

# convert data to numpy array
data = np.array(data.astype('float') / 255.0)

# reshape to a 4d so keras conv2d can handle properly
data = data.reshape(data.shape[0], 28, 28, 1)

# load the trained network model
model = load_model(args['model'])

# make predictions
predictions = model.predict(data, batch_size=64, verbose=1).argmax(axis=1)

df = pd.DataFrame(data=predictions)

### stopping here for now. going to flesh the model out out later in a diff file.
