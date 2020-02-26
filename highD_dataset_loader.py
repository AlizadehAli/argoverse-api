from highD.src.data_management.read_csv import *
import pandas

challenge_pred_data = pandas.read_csv("/home/jean/Tensorflow/highD-dataset-v1.0/highD-predictionChallenge/00_predictionCarsPublic.csv")
challenge_input_data = pandas.read_csv("/home/jean/Tensorflow/highD-dataset-v1.0/highD-predictionChallenge/00_observation.csv")

data = pandas.read_csv( "/home/jean/Tensorflow/highD-dataset-v1.0/data/01_tracks.csv")

print("ok")

