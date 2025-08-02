import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np
df=pd.read_csv("electricity_cost_dataset.csv")
print(df.head())
print(df.shape)
df.columns=df.columns.str.strip().str.replace(' ','_')  # Remove/replace any leading/trailing whitespace from column names
print(df.isnull())#all are false meanig no empty values
print(df.dtypes)
print(df.describe())
print(df.info())
sns.histplot(df['electricity_cost'],kde=True)
plt.show()
df['structure_type'].value_counts().plot(kind='bar',color='orange')
plt.show()
df.corr(numeric_only=True)["electricity_cost"].sort_values(ascending=False).plot(kind='bar', color='green')
plt.show()
sns.scatterplot(x='site_area', y='electricity_cost', data=df, color='blue')
plt.show()
sns.boxplot(x='structure_type', y='electricity_cost', data=df, palette='Set2')
plt.show()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.tight_layout()
    plt.show()
for col in numerical_columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df[col])
    # plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(x='structure_type', y='electricity_cost', data=df)
plt.title('Electricity Cost by Structure Type')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
sns.pairplot(df[numerical_columns.union(['electricity_cost'])], diag_kind='kde')
plt.show()
df=pd.get_dummies(df, columns=['structure_type'], drop_first=True)#avoids multicollinearity by dropping one dummy column.
print(df.head())
X=df.drop(columns=['electricity_cost'],axis=1)
Y=df['electricity_cost']
Xtrain, Xtest, Ytrain, Ytest = sk.model_selection.train_test_split(X, Y, test_size=0.2, random_state=101)
#test_size=0.2: This means 20% of the data will go to testing, and 80% will be used for training.
print(Xtrain.shape, Xtest.shape)
print(Ytrain.shape, Ytest.shape)
Xtrain_Scaled = sk.preprocessing.StandardScaler().fit_transform(Xtrain)
Xtest_Scaled = sk.preprocessing.StandardScaler().fit_transform(Xtest)
print(Xtrain_Scaled.shape, Xtest_Scaled.shape)
Xtrain = pd.DataFrame(Xtrain_Scaled, columns=X.columns)
Xtest = pd.DataFrame(Xtest_Scaled, columns=X.columns)
Xtrain.to_csv("X_train.csv", index=False)
Xtest.to_csv("X_test.csv", index=False)
Ytrain.to_csv("y_train.csv", index=False)
Ytest.to_csv("y_test.csv", index=False)
# sns.histplot(Ytest, kde=True, color='blue')
# plt.show()
# sns.histplot(Ytrain, kde=True, color='blue')
# plt.show()
# a step to flatten them into one-dimensional NumPy arrays before feeding them into many machine learning models.
Ytrain=Ytrain.values.ravel()
Ytest=Ytest.values.ravel()
lr_model=sk.linear_model.LinearRegression()
lr_model.fit(Xtrain, Ytrain)
y_pred_lr=lr_model.predict(Xtest)
rf_model=sk.ensemble.RandomForestRegressor(random_state=42)
rf_model.fit(Xtrain, Ytrain)
y_pred_rf=rf_model.predict(Xtest)

mae=sk.metrics.mean_absolute_error(Ytest, y_pred_lr)
mse=sk.metrics.mean_squared_error(Ytest, y_pred_lr)
rmse=mse**0.5
r2=sk.metrics.r2_score(Ytest, y_pred_lr)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")


Rf_mae=sk.metrics.mean_absolute_error(Ytest, y_pred_rf)
Rf_mse=sk.metrics.mean_squared_error(Ytest, y_pred_rf)
Rf_rmse=mse**0.5
Rf_r2=sk.metrics.r2_score(Ytest, y_pred_rf)
print(f"MAE: {Rf_mae}, MSE: {Rf_mse}, RMSE: {Rf_rmse}, R2: {Rf_r2}")
sns.scatterplot(x=Ytest, y=y_pred_rf, color='green',alpha=0.6)
sns.scatterplot(x=Ytest, y=y_pred_lr, color='red',alpha=0.6)

plt.show()

from sklearn.model_selection import GridSearchCV

parameters = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
# grid_search = GridSearchCV(estimator = rf_model,param_grid = parameters,scoring = 'r2',
#                            cv = 5,
#                            n_jobs = -1,verbose= 2)
# grid_search.fit(Xtrain, Ytrain)
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_
# best_estim=grid_search.best_estimator_
import joblib
import json
loaded_model = joblib.load("best_rf_model.pkl")
y_pred_best = loaded_model.predict(Xtest)
# joblib.dump(best_estim, "best_rf_model.pkl")
# with open('best_rf_params.json', 'w') as f:
#     json.dump(grid_search.best_params_, f)

# mae = sk.metrics.mean_absolute_error(Ytest, y_pred_best)
# mse = sk.metrics.mean_squared_error(Ytest, y_pred_best)
# rmse = np.sqrt(mse)
# r2 = sk.metrics.r2_score(Ytest, y_pred_best)
# metrics_dict = {
#     "MAE": mae,
#     "MSE": mse,
#     "RMSE": rmse,
#     "R2": r2
# }
# with open("rf_metrics.json", "w") as f:
#     json.dump(metrics_dict, f, indent=4)
with open('rf_metrics.json', 'r') as f:
    metrics = json.load(f)
print("MAE:", metrics["MAE"])
print("RMSE:", metrics["RMSE"])
print("R²:", metrics["R2"])
feature_importance = pd.Series(loaded_model.feature_importances_, index=X.columns)
feature_importance.sort_values(ascending=True).plot(kind='barh', figsize=(10, 6), color='teal')
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(Ytest, y_pred_best, color='purple', alpha=0.6)
plt.plot([Ytest.min(), Ytest.max()], [Ytest.min(), Ytest.max()], 'g--', lw=2)
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.title("Actual vs Predicted Electricity Costs")
plt.tight_layout()
plt.show()


# def predict_cost(model, input_data):
#     return model.predict([input_data])[0]
