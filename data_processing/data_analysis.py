
from pandas import read_excel
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Reading dataset
diabetes = pd.read_csv('dataset_diabetes/diabetic_data.csv')
diabetes.head(5)

