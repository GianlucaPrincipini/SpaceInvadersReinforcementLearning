import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from utils import plotLearning

input_file_name = sys.argv[1] + "_scores.dat"
output_file_name = sys.argv[1] + "_cut.png"
x_axis_limit = int(sys.argv[2])
points = []
with (open(input_file_name, "rb")) as openfile:
    points = pickle.load(openfile)
points = points[:x_axis_limit]
plotLearning(points, filename=output_file_name, window=100)