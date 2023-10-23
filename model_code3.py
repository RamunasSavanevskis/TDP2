import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv("example_data.csv")
print(df.corr()["HEPI"])
df_np = df.to_numpy()
x_train = df_np[:, :-1]
y_train = df_np[:, -1]
sklearn_model = LinearRegression().fit(x_train, y_train)
hardness = float(input("Measured hardness input: "))
resistivity = float(input("Measured resistivity input: "))
attenuation = float(input("Measured attenuation input: "))
velocity = float(input("Measured ultrasound longitudinal velocity input: "))

#given_values = np.array([[1, 1.9, 0.3,12.1]])
given_values = np.array([[hardness,resistivity,attenuation,velocity]])
def get_pred(model,x):
    return model.predict(x)
sklearn_y_predictions = sklearn_model.predict(x_train)
prediction = get_pred(sklearn_model, given_values)
print ("Mean absolute error : ", mean_absolute_error(sklearn_y_predictions, y_train))
print ("Mean squared error : ", mean_squared_error(sklearn_y_predictions, y_train))
print("Predicted Value:", prediction[0])
