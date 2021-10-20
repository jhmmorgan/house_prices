import pandas as pd
import numpy as np
import math

# Use this cell to begin, and add as many cells as you need to complete your analysis!
# Libaries
import matplotlib.pyplot as plt
import seaborn as sns

# Statistics
from scipy import stats
from scipy.stats import norm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_squared_log_error as MSLE

# Root Mean Squared Error - RMSE evaluation


# Lets read in the data...
# However, before we do, the data description shows NA as being a valid value for many of the categories
# Normally meaning none, i.e. Alley == NA means no alley access.

# Get the default NA values from Pandas and remove "NA". Use this as the default list of NA's
_na_values = ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN', '#N/A N/A', '#N/A', 'N/A', 'n/a', '<NA>', '#NA', 'NULL', 'null', 'NaN', '-NaN', 'nan', '-nan', '']

# Now we can read the files
src_path      = "house-prices-advanced-regression-techniques/"
test_df       = pd.read_csv(src_path + "test.csv", keep_default_na = False, na_values = _na_values)
train_df      = pd.read_csv(src_path + "train.csv", keep_default_na = False, na_values = _na_values)
sample_sub_df = pd.read_csv(src_path + "sample_submission.csv")

train_df.head()
# Using Ids as Indexes
for df in [train_df, test_df]:
    df.set_index("Id", inplace=True)
train_df.head()


# Data Type Check
df = pd.DataFrame({"Column": train_df.columns, "Dtype": train_df.dtypes.astype("str").tolist(), "Sample1": train_df.loc[1].tolist(),
                   "Sample2": train_df.loc[50].tolist(), "Sample3": train_df.loc[500].tolist()})
print(df.to_string())

# MSSubClass, MoSold, YrSold are categorical, but stored as numbers
# OverallQal, OverallCond are also categorical, however with a scale of 1 to 10 so are ok to remain numbers
# CentralAir is a Y/N and so should be a boolean

for df in [train_df, test_df]:
    df.replace({"MSSubClass": {20: "SC20", 30: "SC30", 40: "SC40", 45: "SC45", 50: "SC50", 60: "SC60", 70: "SC70", 75: "SC75",
                               80: "SC80", 85: "SC85", 90: "SC90", 120: "SC120", 150: "SC150", 160: "SC160", 180: "SC180", 190: "SC190"}, 
                "MoSold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"},
                "CentralAir": {"Y": True, "N": False},
                "LotFrontage": {"NA" : np.nan}},
               inplace=True)
    df["YrSold"] = pd.Categorical(df.YrSold)
    df["LotFrontage"] = pd.to_numeric(df.LotFrontage)


# Filling the Missing Values

_train_df      = train_df.drop(columns = "SalePrice")

combined_df = pd.concat([_train_df, test_df])
MSZoning_series = combined_df.groupby('Neighborhood').MSZoning.agg(lambda x:x.value_counts().index[0])
LotFrontage_series = combined_df.groupby('Neighborhood').LotFrontage.median()

combined_df_filled = combined_df.fillna({'MSZoning': combined_df['Neighborhood'].apply(lambda x: MSZoning_series[x]),
                                         'LotFrontage': combined_df['Neighborhood'].apply(lambda x: LotFrontage_series[x])})

train_df_filled = combined_df_filled.iloc[:len(train_df)]
train_df_filled = pd.concat([train_df_filled, train_df['SalePrice']], axis=1)
test_df_filled = combined_df_filled.iloc[len(train_df):]

train_df_filled.isnull().sum().sum()

train_df_filled[["MSSubClass", "MoSold", "YrSold", "CentralAir"]].head()



# Missing Values Check
# We need to check this for both the train and test data.
# First, we need to drop the sales price and then combine the data frames.
combined_df    = pd.concat([_train_df, test_df_filled])
missing_values = combined_df.isnull().sum().sum()
missing_values.sum()

# Great, there are no N/A's, except lot frontage, mainly due to us igoring "NA" as being an invalid / missing item.


# Lets explore the data...


# Figure
plt.figure(figsize=(12, 4))

# Distribution Plot
plt.subplot(1, 2, 1)
#sns.distplot(train_df_filled['SalePrice'], fit=norm)
sns.histplot(train_df_filled["SalePrice"], stat = "density", kde = True)
plt.title('Distribution Plot')

# Probability Plot
plt.subplot(1, 2, 2)
stats.probplot(train_df_filled['SalePrice'], plot=plt)

plt.tight_layout()
plt.show()
plt.clf()

# Applying Log-Transform
train_df_filled['SalePrice'] = np.log1p(train_df_filled.SalePrice)

# Figure
plt.figure(figsize=(12, 4))

# Distribution Plot
plt.subplot(1, 2, 1)
sns.distplot(train_df_filled['SalePrice'], fit=norm)
plt.title('Distribution Plot')

# Probability Plot
plt.subplot(1, 2, 2)
stats.probplot(train_df_filled['SalePrice'], plot=plt)

plt.tight_layout()
plt.show()
plt.clf()



# Overview of the features based on their data type
numerical_features = train_df_filled.select_dtypes(include = [np.number, bool]).columns
print(f'Numerical Features ({len(numerical_features)}):\n{numerical_features}')
categorical_features = train_df_filled.select_dtypes(exclude = [np.number, bool]).columns
print(f'Categorical Features ({len(categorical_features)}):\n{categorical_features}')

# We want to split the numerical and categorical features into groups to view the data better
# To do this, we'll group these in sets of 10

# How many groups are needed?
#  Each will be a 4x4 grid. Total of 16 charts per plot
#  Each plot will have two charts, total of 8 features per plot
numerical_groups   = math.ceil(len(numerical_features.values)/8)
categorical_groups = math.ceil(len(categorical_features.values)/8)

total_groups       = numerical_groups + categorical_groups

numerical_step     = 8
categorical_step   = 8

group_num = np.empty(int(numerical_groups), dtype = pd.Series)
for grp in np.arange(numerical_groups):
#  print(grp * numerical_step)
  st = int(grp * numerical_step)
  en = int((grp+1) * numerical_step - 1)+1
  group_num[int(grp)] = numerical_features[st:en]


group_cat = np.empty(int(categorical_groups), dtype = pd.Series)
for grp in np.arange(categorical_groups):
  #print(grp * numerical_step)
  st = int(grp * categorical_step)
  en = int((grp+1) * categorical_step - 1)+1
  group_cat[int(grp)] = categorical_features[st:en]


# EDA of all groups
#groups = np.concatenate([group_num, group_cat])
groups = group_num
for grp in groups:
  i = 1
  for feature in grp:
    # Distribution Plot
    width  = 4
    height = 4
    _=plt.subplot(height, width, i)
    _=sns.histplot(train_df_filled[feature], kde=True, stat="density", linewidth=0)
    _=plt.title('Distribution')
    i += 1
      
    # Scatter Plot
    _=plt.subplot(height, width, i)
    _=sns.scatterplot(data=train_df_filled, x=feature, y='SalePrice', alpha=0.5)
    _=plt.title('Relationship')
    i += 1
  
  plt.tight_layout()
  plt.show()
  plt.clf()

### Outliers
# LotFrontage > 250
# LotArea > 100000
# BsmtFinSF1 > 4000
# BsmtFinSF2 > 1200
# TotalBsmtSF > 5000
   # 1stFlrSF > 4000
# GrLivArea > 4000
# KitchenAbvGr = 0
# WoodDeckSF > 750
# OpenPorchSF > 500
# EnclosedPorch > 500
# MiscVal > 5000

  # LowQualFin = 0 = False, =>1 = True
  # BsmtFullBath 0 = False, =>1 = True
  # BsmtHalfBath 0 = False, =>1 = True
  # HalfBath 0 = False, =>1 = True
  # BedroomAbvGr  >= 5 = 5
  # KitchenAbvGr >=2 = 2
  # Fireplaces >= 2 = 2
  # GarageCars >= 3 = 3
  # PoolArea === HasPool PoolArea 0 = False, >0 = True
  
  # Drop Street, Utilities, Condition2
  # RoofMatl = CompShg or Other
  # ExterQual = Gd/Ex = Good, TA/FA = Average
  # Heating = GasA or Other
  # Electrical = SBrkr or Other
  # KitchenQual = Gd/Ex = Good, TA/FA = Average
  # Functional = Typ, Other
  # SaleType = WD, New, Other

train_df_cleaned = train_df_filled.drop(train_df_filled[(train_df.LotFrontage>200)|
                                          (train_df.LotArea>100000)|
                                          (train_df.LotFrontage > 250)|
                                          (train_df.LotArea > 100000)|
                                          (train_df.BsmtFinSF1 > 4000)|
                                          (train_df.BsmtFinSF2 > 1200)|
                                          (train_df.TotalBsmtSF > 5000)|
                                          (train_df.GrLivArea > 4000)|
                                          (train_df.KitchenAbvGr == 0)|
                                          (train_df.WoodDeckSF > 750)|
                                          (train_df.OpenPorchSF > 500)|
                                          (train_df.EnclosedPorch > 500)|
                                          (train_df.MiscVal > 5000)].index)



print(f'Reduction in train data is {np.round(100*(len(train_df)-len(train_df_cleaned))/len(train_df), 2)}%')




# Let's encode the categorical features
test_df_cleaned = test_df_filled.copy()
_train_df      = train_df_filled.drop(columns = "SalePrice")
combined_df_cat = pd.concat([_train_df, test_df])[categorical_features].reset_index(drop=True)

encoder_mapping = pd.DataFrame(index = categorical_features, columns = {"encoder", "mapping"})
for i in np.arange(len(categorical_features)):
  le = LabelEncoder()
  encoder_mapping.iloc[i]["encoder"] = le.fit(list(combined_df_cat.iloc[:,i]))
  encoder_mapping.iloc[i]["mapping"] = dict(zip(le.classes_, range(len(le.classes_))))


for feature in encoder_mapping.index:
  train_df_cleaned.replace({feature: encoder_mapping.loc[feature]["mapping"]}, inplace=True)
  test_df_cleaned.replace({feature: encoder_mapping.loc[feature]["mapping"]}, inplace=True)


# Parameters
SEED      = 42
test_size = 0.3   #  30% test, 70% train

# Now let's split the data
#experiment = train_df_cleaned[np.expm1(train_df_cleaned.SalePrice) < 300000]
#X = experiment.drop(["SalePrice"], axis = "columns")          # Independant columns (all the features used for prediction)
#y = experiment["SalePrice"]                                   # Target Columns - the Price range

X = train_df_cleaned.drop(["SalePrice"], axis = "columns")          # Independant columns (all the features used for prediction)
y = train_df_cleaned["SalePrice"]                                   # Target Columns - the Price range

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=SEED)





scaler           = StandardScaler()
X_train2         = pd.DataFrame(scaler.fit_transform(X_train))
X_test2          = pd.DataFrame(scaler.transform(X_test))

X_train2.columns = X_train.columns.values
X_test2.columns  = X_test.columns.values

X_train2.index   = X_train.index.values
X_test2.index    = X_test.index.values

X_train          = X_train2
X_test           = X_test2

scores = []
forest = RandomForestRegressor(n_estimators = 20, random_state = SEED)
acc = cross_val_score(forest, X_train, y_train, scoring = "neg_mean_squared_log_error", cv = 5)
scores.append(acc.mean())

results = pd.DataFrame({"Metrics":["MSLE"], "Accuracy": scores})
print("Initial Cross Validation Results")
print(results)


param_grid = {
    "bootstrap":[True, False],
    "max_depth":[80, 100, None],
    "max_features":[4, 10, "auto"],
    "min_samples_leaf":[1, 4],
    "min_samples_split":[2, 6, 10],
    "n_estimators":[75, 250]
}

forest = RandomForestRegressor(random_state=SEED)
grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

_ = grid_search.fit(X_train, y_train)

print("Best parameters are:")
print(grid_search.best_params_)

scores2 = []
best_grid = grid_search.best_estimator_
acc2 = cross_val_score(best_grid, X_train, y_train, scoring = "neg_mean_squared_log_error", cv = 5)
scores2.append(acc2.mean())
results = pd.DataFrame({"Metrics":["MLSE"], "Accuracy":scores,"Accuracy Tuned Param": scores2})
print("Results of the hyper tuned grid cross validation")
print(results)

# How well does this perform on data it's never seen before? Let's use the test data
best_grid      = grid_search.best_estimator_
y_pred         = best_grid.predict(X_test)
y_pred         = np.expm1(y_pred)
y_test         = np.expm1(y_test)
acc_test_MSLE  = MSLE(y_test, y_pred)
acc_test_MAE   = round(MAE(y_test, y_pred),2)
acc_test_MSE   = round(MSE(y_test, y_pred, squared = False),2)

results = pd.DataFrame({"MSLE Accuracy":scores,
                        "MSLE Accuracy Tuned Param": scores2,
                        "MSLE Accuracy Test Set":acc_test_MSLE,
                        "MAE": acc_test_MAE})
print("Results when using the test data:")
print(results)

t = np.linspace(min(y_test), max(y_test), len(y_test))

plt.plot(t, t, c = "red")
plt.scatter(y_pred, y_test,alpha = 0.5)
plt.show()
plt.clf()


lasso = LassoCV(alphas=None, max_iter=100000, normalize=True)
lasso.fit(X_train, y_train)
best_alpha = lasso.alpha_
scores = np.sqrt(-cross_val_score(lasso, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
print(f'Lasso Regression: Best Alpha={round(best_alpha, 4)} RMSE={round(np.mean(scores), 4)}')

y_pred2 = np.expm1(lasso.predict(X_test))
y_pred3 = np.mean( np.array([ y_pred, y_pred2 ]), axis=0 )

plt.plot(t, t, c = "red")
plt.scatter(y_pred3, y_test,alpha = 0.5)
plt.show()
plt.clf()

MSLE(y_test, y_pred3)

round(MAE(y_test, y_pred3),2)

train_df_filled["SalePrice"] = np.expm1(train_df_filled["SalePrice"])

plt.plot(t, np.linspace(0,0,len(t)), c = "red")
plt.scatter(y_test, y_test-y_pred3, alpha = 0.3)
plt.xlim(min(y_test), 250000)
plt.xlim(250000, max(t))
plt.show()
plt.clf()

name = []
importance = []
for n, i in zip(X, best_grid.feature_importances_):
    name.append(n)
    importance.append(round(i*100,2))

features = pd.DataFrame({"Features": name, "Importance (%)":importance})
features = features.sort_values(by = "Importance (%)", ascending = False)
print("Importance of features")
print(features)

plt.scatter(train_df.YearBuilt, train_df.SalePrice)
plt.show()
plt.clf()


#######################
scaler           = StandardScaler()
test             = pd.DataFrame(scaler.fit_transform(test_df_cleaned))

y_test_final = np.expm1(best_grid.predict(test))

# Submitting Prediction
submission = pd.DataFrame(index = test_df.index, data = {'SalePrice': y_test_final})
submission.to_csv('submission.csv', index=True)
print('Submission is successful.')


