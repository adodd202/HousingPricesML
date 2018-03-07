from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, AdaBoostRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn import clone
from sklearn import tree
import xgboost as xgb
import numpy as np
import lightgbm as lgb
import tensorflow as tf
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import KFold 
import warnings
from sklearn.svm import SVR
import pandas as pd
import math


# Turn off warnings
warnings.filterwarnings('ignore')



# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# This is class that you can create to instantiate a model, be it lasso or xgb.
#
# ~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~
#	- def train_validate(): for training and validating only on training set
# 	- def train_predict(): for training on test set and submitting on test set to Kaggle
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Model():

	def __init__(self, alpha = .85, random_state = 1, model = "lasso"):
		self.model_name = model

		# Creating the correct model.
		if model == "gboost":
			self.model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
		elif model == "gboost_deep":
			self.model = GradientBoostingRegressor(n_estimators=5000, learning_rate=0.02,
                                   max_depth=6, max_features='sqrt',
                                   min_samples_leaf=17, min_samples_split=12, 
                                   loss='huber', random_state =5)
		elif model == "ridge":
			self.model = Ridge(alpha = .0005, random_state = 42)
		elif model == "lasso":
			self.model = Lasso(alpha = .0005, random_state = 42)
		elif model == "elastic":
			self.model = ElasticNet(alpha=.0005, l1_ratio=0.9, random_state=42)
		elif model == "rf":
			self.model = RandomForestRegressor(n_estimators = 300, max_depth = 350, random_state = 42)
		elif model == "krr":
			self.model = KernelRidge(alpha=0.035448,degree=1.0689655, coef0=7206.896)
		elif model =="xgb_new":
			self.model = xgb.XGBRegressor(n_estimators= 6698, max_depth= 1, gamma= 0, min_child_weight= 1)
		elif model == "xgb":
			self.model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

		elif model == "xgb_deep":
			self.model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=7, 
                             min_child_weight=1.7817, n_estimators=4000,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
		elif model == "adaboost":
			self.model = AdaBoostRegressor(tree.DecisionTreeRegressor(),
                          n_estimators=500, random_state=42)
		elif model == "svr":
			self.model = SVR(C=10, epsilon=0.0, kernel = 'rbf')
		elif model == "lgb":
			self.model = lgb.LGBMRegressor(
				objective='regression',
				num_leaves=5,
                learning_rate=0.05,
                n_estimators=720,
                max_bin = 55,
                bagging_fraction = 0.8,
                bagging_freq = 5,
                feature_fraction = 0.2319,
                feature_fraction_seed=9,
                bagging_seed=9,
                min_data_in_leaf=6,
                min_sum_hessian_in_leaf=11)
		else:
			raise ValueError('Please give a model that exists in models.py!')

	def train_validate(self, X, y, n_folds = 5):
		# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
		# Fits lasso to n_folds, computes error across all folds averaged and saves model.
		# Note that this fits on 80% of data in 5 folds
		#
		# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
		# Inputs: 
		#	- X: this is the full training set, ~1450 observations
		#	- y: this the labels from the training set, ~1450 labels
		#	- n_folds: number of folds to run for validation
		# Outputs:
		#	- y_pred_all: this is the predictions from each fold combined with each other
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Either stratified or not
		skf = KFold(n_splits=n_folds)
		#skf = StratifiedKFold(n_splits=n_folds)

		y_pred_all = np.zeros(len(y))
		log_error_sum = 0

		i = 0
		for train_index, test_index in skf.split(X, y):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			print("Round {}".format(i))

			# Cloning to avoid residual fits
			instance = clone(self.model)

			# NOTE: If you don't want to transform, comment this.
			# Forward transform (normalizing for effect of national average)
			# y_transform = transform_y(X_train, y_train, direction = "forward")

			# Fitting
			instance.fit(X_train, y_train) #, y_transform)

			# Get predictions
			y_pred = instance.predict(X_test)

			# NOTE: If you don't want to transform, comment this.
			# Reverse to original data (adding effect of year back in)
			# y_pred = transform_y(X_test, y_pred, direction = "reverse")

			# Removing the clone
			del instance

			# Anything less than $0 is put to 50,000
			mask = y_pred < 0
			y_pred[mask] = 50000

			# Adding to error sum.
			log_error_part = get_error(y_test, y_pred)
			log_error_sum += log_error_part

			# Predicting on out of fold
			y_pred_all[test_index] = y_pred 

			i += 1 

		cv_error = get_error(y_test, y_pred, type = "rmse")
		print("Log error across all validation folds for {} is {}".format(self.model_name, cv_error))
		return y_pred_all, cv_error

	def train_predict(self, X_train, y_train, X_predict):
		# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
		# This function will train on the entire training dataset and will then create predictions on the Kaggle test set.
		# This function is mean primarily for the purpose of submission.
		#
		# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
		# Input:
		#	- X_train: Full train dataset to train on
		#	- y_train: Full y train to train on
		#	- X_predict: Full test dataset (without labels)
		# Output:
		#	- y_pred: predictions on X
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		instance = clone(self.model)
		instance.fit(X_train, y_train)

		y_pred = instance.predict(X_predict)
		return y_pred



### DOESN'T HELP, MAYBE TRY SEASONALITY LATER? PROBS NOT...already have month column.
def transform_y(X, y, direction):
	# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
	# If direction is forward, then we want to scale the costs so that the higher housing market years
	# like 2007 have y values that are scaled lower (to normalize for price), while 2009 houses are scaled
	# up (to normalize for price).
	# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
	# Input:
	#	- X: Training dataset
	#	- y: Training labels
	#	- direction: "forward" or "reverse"
	# Output:
	#	- y: Changed y values
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	time_series = pd.read_csv('./Data/time_series.csv')

	# Undo log transform
	y = np.exp(y)

	# Combining X and y and adding to a dataframe
	X_and_y  = np.column_stack((X, y))
	X_and_y = pd.DataFrame(X_and_y)

	# Changing all columns to strings, merging dataframe with time series info.
	X_and_y.columns = X_and_y.columns.astype(str)
	X_and_y = pd.merge(X_and_y, time_series,  how='left', left_on=["61","60"], right_on = ['YrSold','MoSold'])

	# Getting column for scaling
	colScale = 125

	# Convert to np array
	X_and_y = X_and_y.as_matrix()

	if direction == "forward":
		print("forward")
		print(X_and_y[1:3,[60, 61, 119, colScale]])

		# Transforming y here by 0-1 scalar (usually like 85%-100% or something)
		X_and_y[:,119] = np.multiply(X_and_y[:,119], X_and_y[:,colScale])
		print(X_and_y[1:3,[60, 61, 119, colScale]])

		# Redo log transform
		X_and_y[:,119] = np.array(list(map(math.log, X_and_y[:,119])))
		y = X_and_y[:,119]

	if direction == "reverse":
		print("reverse")
		print(X_and_y[1:3,[60, 61, 119, colScale]])

		# Transforming y here by 0-1 scalar (usually like 85%-100% or something)
		X_and_y[:,119] = np.divide(X_and_y[:,119], X_and_y[:,colScale])
		print(X_and_y[1:3,[60, 61, 119, colScale]])

		# Redo log transform
		X_and_y[:,119] = np.array(list(map(math.log, X_and_y[:,119])))
		y = X_and_y[:,119]


	del X_and_y
	del time_series

	return y


# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# get_error returns root mean square log error or root mean square error.
# rmsle is good for Kaggle comparison
# rmse is good for seeing the actual average error
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_error(y_true, y_pred, type = "rmsle"):
	assert len(y_true) == len(y_pred), "Length of prediction and test do not match!"

	list_types = ["rmsle","rmse"]
	assert type in list_types, "Error type you requested is not an option. Choose 'rmse' or 'rmsle' please."

	# Root mean square error (RMSE)
	if type == "rmse":
		return np.square(y_pred - y_true).mean() ** 0.5

	# Root mean square log error (RMSLE)
	if type == "rmsle":

		return np.square(np.log(y_pred) - np.log(y_true)).mean() ** 0.5

