from Regression import load_data_regression, preprocess_data_regression, model1_regression_vecteur, preprocess_data_regression_vecteurs, model1_regression, plot_ypredVSy_test, model_conv2dTranspose_Regression, train_model, plot_training_history_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf

"Regression avec convolution"
# data=load_data_regression()
X_combined, Y = preprocess_data_regression()
X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)
# # trained_model, trained_history = train_model(X_combined, Y, model1_regression(), epochs=25)
# # trained_model.save("C:/Users/LD/Desktop/Projet P5/Model1")
loaded_model = tf.keras.models.load_model("C:/Users/LD/Desktop/Projet P5/Model1")

# # plot_training_history_regression(trained_history)
y_pred = loaded_model.predict(X_test)
plot_ypredVSy_test(y_pred, Y_test, sample_size=3)

"Essayer avec des équipes sous forme de vecteurs -> Regression linéaire/Random Forest"
# data = load_data_regression()
# X_combined, Y = preprocess_data_regression_vecteurs()
# X_train, X_test, Y_train, Y_test = train_test_split(X_combined, Y, test_size=0.2, random_state=42)
# trained_model, trained_history = train_model(X_combined, Y, model1_regression_vecteur(), epochs=25)
# plot_training_history_regression(trained_history)
# y_pred = model1_regression_vecteur().predict(X_test)
# plot_ypredVSy_test(y_pred, Y_test, sample_size=3)

