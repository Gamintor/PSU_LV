import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error

df = pd.read_csv('C:\\Users\\Matija\\Desktop\\cars_processed.csv')
print(df.head())

df = df.drop(columns=['name'])
print(df.columns)

df = pd.get_dummies(df, columns=['fuel', 'seller_type', 'transmission', 'owner'], drop_first=True)
print(df.columns)

X = df.drop(columns=['selling_price'])
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)

y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("--- Rezultati evaluacije ---")
print("R2 (Train):", r2_score(y_train, y_pred_train))
print("R2 (Test):", r2_score(y_test, y_pred_test))
print("RMSE (Test):", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Max error (Test):", max_error(y_test, y_pred_test))
print("MAE (Test):", mean_absolute_error(y_test, y_pred_test))

fig = plt.figure(figsize=[13, 10])
ax = sns.regplot(x=y_pred_test, y=y_test, line_kws={'color': 'green'})
ax.set(xlabel='Predikcija', ylabel='Stvarna vrijednost', title='Rezultati na testnim podacima')
plt.show()

features_sets = [['km_driven', 'year'], ['km_driven', 'year', 'engine'], ['km_driven', 'year', 'engine', 'max_power']]
for features in features_sets:
    X_train_subset = X_train[features]
    X_test_subset = X_test[features]
    
    X_train_s = scaler.fit_transform(X_train_subset)
    X_test_s = scaler.transform(X_test_subset)
    
    linear_model.fit(X_train_s, y_train)
    y_pred_test = linear_model.predict(X_test_s)
    
    print(f"\nEvaluacija za ulazne veličine: {features}")
    print("R2 (Test):", r2_score(y_test, y_pred_test))
    print("RMSE (Test):", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    print("MAE (Test):", mean_absolute_error(y_test, y_pred_test))
