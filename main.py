from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd

data = pd.read_csv('dataset/diabetes.csv')
# print(data.head(10))

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in zero_not_accepted:
    data[col] = data[col].replace(0, np.NaN)
    mean = int(data[col].mean(skipna=True))
    data[col] = data[col].replace(np.NaN, mean)

x = data.iloc[:, :8]
y = data.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# model = DecisionTreeClassifier(criterion='entropy', random_state=10, max_depth=3, min_samples_leaf=5)
#
# model.fit(x_train, y_train)
#
# y_pred = model.predict(x_test)
#
# print(y_pred)
#
# score = accuracy_score(y_test, y_pred)
#
# print(score)

# ------------------------------------------------------------------

model = DecisionTreeClassifier()

param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_
best_model.fit(x_train, y_train)
accuracy = best_model.score(x_test, y_test)
print("best : ", grid_search.best_params_)
print(accuracy)