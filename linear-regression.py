# https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

diabetes = datasets.load_diabetes()

# Use only one feature (the 3rd feature)
diabetes_X = diabetes.data[:, np.newaxis, 2]

n_test = 5

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-n_test] # get all except the last 20 elements
diabetes_X_test = diabetes_X[-n_test:] # get last 20 elements

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-n_test]
diabetes_y_test = diabetes.target[-n_test:]

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

print('Coefficients: ', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

plt.scatter(diabetes_X_train, diabetes_y_train,  color='black')
plt.scatter(diabetes_X_test, diabetes_y_test,  color='red')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

