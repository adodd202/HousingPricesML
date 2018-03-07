from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.cross_validation import cross_val_score
from sklearn import clone

import lightgbm as lgb
from models import Model, get_error
import xgboost as xgb
import numpy as np
import pandas as pd
import warnings
from bayes_opt import BayesianOptimization
import random
from itertools import combinations



# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# This function is the main workhorse of this script. It does the following:
# 	1. Imports datasets and performs scaling (train and test csvs from data folder)
#	2. Calls all base learners being used and trains them.
#	3. Calls the stacker
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def main():

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 1. Imports and scaling
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Importing roger's dataset
	X_train, y_train, X_predict = import_data(name_data = "roger")

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 2. Base learners are trained and predictions are made on the training set.
	# These are then added to the stacking matrix.
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	numRounds = 3
	add_interaction = 0
	models = ["lasso", "elastic", "lgb", "gboost", "gboost_deep"] # "xgb", "xgb_deep"#if you're not Andrew, add krr, ridge

	X_stack_train = np.zeros((y_train.shape[0], len(models)*numRounds))
	X_stack_predict = np.zeros((X_predict.shape[0], len(models)*numRounds))

	for i in range(numRounds):
		print("Will now run {} models and average their predictions...Round {} of {}.".format(len(models), i+1, numRounds))
		j = 0

		# Add interaction terms here X1*X2, X1*X2*X3
		# 120 terms, 3600 
		# X_train = X_train + inter_terms
		if add_interaction:
			X_train_inter, X_predict_inter = interaction_terms(X_train, X_predict)
			X_train = np.concatenate((X_train, X_train_inter), axis = 1)
			X_predict = np.concatenate((X_predict, X_predict_inter), axis = 1)

		for model in models:
			indexCol = len(models) * i + j

			# Making user feedback
			val = ((indexCol)/(len(models)*numRounds))*100
			print("============================ {}% Complete ============================".format(round(val,5)))
			instance = Model(model = model)

			# Running model train, validation and prediction
			instance_pred, cv_error = instance.train_validate(X_train, y_train)

			if cv_error < .2:
				instance_test_pred = instance.train_predict(X_train, y_train, X_predict)

				# Adding the predictions to the stacking matrices (training/validation, and prediction)
				X_stack_train[:,indexCol] = instance_pred
				X_stack_predict[:,indexCol] = instance_test_pred
			j += 1

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# 3. Stacking implementation 
	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	# Initializing Stacker and giving it the inputs:
	# X_stack: combined base model predictions on training set
	# y_train: true housing prices of training set 
	# X_test: Kaggle test set to predict on

	# We can optimize the stack here
	#optimize_stack(X_stack_train, y_train)

	stack = Stacker(model = "average")
	stack.stack_valid_predict(X_stack_train, y_train, X_stack_predict, output_csv = "yes")


# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
# This is a stacking class that can stack with either xgboost or average stacker.
# Later I will add best base and average stacking.
#
# ~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~
#	- __init__: this initializes the stacking object, either xgboost metamodel, or an averaging function
#				can be used
#	- stack: this performs the stacking process
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Stacker():
	def __init__(self, model = "xgb"):
		# Inputs:
		#	- model: this initializes the model

		self.model_name = model

		# Checking if the model parameter is a valid option.
		list_models = ["xgb", "average", "rf", "lasso"]
		if model not in list_models:
			raise ValueError('Please give a model that exists in housing_main.py: class "mean_stacker!"')

		# Defining xgb meta-model, current params are bad
		if model == "xgb":
			self.model = xgb.XGBRegressor(
					colsample_bytree=0.1,
					gamma=100,
					learning_rate=0.01,
					max_depth=5,
					min_child_weight=50,
					n_estimators=8000,                                                                  
					reg_alpha=0,
					reg_lambda=0.6,
					subsample=1,
					seed=42,
					silent=1)

		# Defining rf metamodel
		if model == "rf":
			self.model = RandomForestRegressor(n_estimators = 10, max_depth = 30, random_state = 42)

		if model == "lasso":
			self.model = Lasso(alpha = .0005, random_state = 42)

	def stack_valid_predict(self, X_stack_train_full, y_stack_train_full, X_stack_predict, output_csv = "no"):
		# ~~~~~~~~~~~~~~~~~~ Summary ~~~~~~~~~~~~~~~~~~~~
		# This function performs stacking on the training dataset
		#
		# ~~~~~~~~~~~~~~~~ Parameters ~~~~~~~~~~~~~~~~~~~
		# Input:
		#	- X_train: This is the base models predictions, shape is [1450 x n_base_models]
		#	- y_train: This is the target dataset for the base model predictions, shape is [1450 x 1]
		#	- X_test: The Kaggle test dataset
		# Output:
		#	- y_pred: predictions on X_test
		#	- output_csv: (optional) This can be submitted to Kaggle competition
		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Creating train/test split
		X_train_stack, X_test_stack, y_train_stack, y_test_stack = train_test_split(X_stack_train_full, y_stack_train_full, 
                                                       test_size=0.2, random_state=42)
		
		# Fitting the model chosen.
		if self.model_name != "average":

			# Turn off warnings
			warnings.filterwarnings('ignore')

			instance = clone(self.model)
			instance.fit(X_train_stack, y_train_stack)
			y_pred_stack = instance.predict(X_test_stack)

			# Removing the clone
			del instance

			rmse = get_error(y_test_stack, y_pred_stack, type = "rmse")

			# Getting score for training set stacking...
			print("The rmsle on the stacking for {} stacking is {}".format(self.model_name, rmse))

		else:
			y_pred_stack = np.mean(X_stack_train_full, axis = 1)

			rmse = get_error(y_stack_train_full, y_pred_stack, type = "rmse")

			# Getting score for training set stacking...
			print("The rmsle on the stacking for {} stacking is {}".format(self.model_name, rmse))

		# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		# Now that we have an idea of how good the model is, we would like to train it on the full train set,
		# and predict on the Kaggle test set so we can generate a csv.

		# Checking if user input was valid.
		output_csv_list = ["yes", "no"]
		if output_csv not in output_csv_list:
			raise ValueError('Please give either a "yes" or "no" to the parameter "output_csv" in "stack_valid_predict"!')

		# Train/predict on Kaggle test dataset...
		if output_csv == "yes":
			if self.model_name != "average":
				instance = clone(self.model)
				instance.fit(X_stack_train_full, y_stack_train_full)
				y_pred = instance.predict(X_stack_predict)

				# Removing the clone
				del instance
			# If we are using averaging
			else:
				y_pred = np.mean(X_stack_predict, axis = 1)

			# Undoing the log transform
			y_pred = np.exp(y_pred)

			# Getting error term and making filename
			rmse = '%.5f'%(rmse)
			filename = "./Predictions/" + "CV_" + str(rmse) + ".csv"

			# Adding id column
			id_num = np.arange(1461,2920)

			# Putting id and predictions together
			y_pred = np.column_stack((id_num.astype(int), y_pred))
			df = pd.DataFrame(y_pred)

			# Adding correct column names
			df.columns = ['Id', 'SalePrice']
			df[['Id']] = df[['Id']].astype(int)

			# Convert to a csv
			df.to_csv(filename, index=False)

def interaction_terms(X_train, X_predict):
	# 20 features

	# 1. 
	# Combine X_train, X_predict across horizontal
	n_train = X_train.shape[0]
	X = np.concatenate((X_train, X_predict), axis = 0)

	# Subset columns
	# Get list of column indices to use
	num_row = X.shape[0]
	num_cols = X.shape[1]
	list_cols = random.sample(range(0, num_cols), 20)

	# Get combos of 2 columns
	cols_2_combo = list(combinations(list_cols, 2))

	# Preallocate memory:
	interaction_terms = np.zeros((num_row, len(cols_2_combo)))

	# For each iteration in combinations, generate new data X1*X2, X1*X3, X2*X3, etc...
	for i in range(len(cols_2_combo)):
		interaction_terms[:,i] = np.multiply(X[:,cols_2_combo[i][0]], X[:,cols_2_combo[i][1]])

	# 3. Split back
	# Split across horizontal line to get back X_train, X_predict
	train_interactions = interaction_terms[0:n_train,:]
	predict_interactions = interaction_terms[n_train:,:]

	return train_interactions, predict_interactions


##############################################################################################################################
############################################## OPTIMIZATION REGION ###########################################################
##############################################################################################################################

def optimize():
	# Lasso optimization
	# lasso_BO = BayesianOptimization(lasso_func, {'alpha': (0.000001, 1000)})
	# lasso_BO.explore({'alpha': [.00001, 0.001, 0.01, 0.1, 1, 10, 100]})
	# lasso_BO.maximize(n_iter=100)
	# print(lasso_BO.res['max'])

	# KRR optimization
	# krr_BO = BayesianOptimization(krr_func, {'alpha': (0,10000), 'degree': (1,5), 'coef0': (0, 10000)})
	# krr_BO.explore({'alpha': [0.001, 0.1, 1], 'degree':[2, 3, 4], 'coef0':[0, .5, 10]})
	# krr_BO.maximize(n_iter=100)
	# print(krr_BO.res['max'])

	#  Elastic optimization
	# elastic_BO = BayesianOptimization(elastic_func, {'alpha': (0,10000), 'l1_ratio': (0,1)})
	# elastic_BO.explore({'alpha': [0.001, 0.1, 1, 10, 100, 1000, 5000], 'l1_ratio': [0, .1, .2, .3, .5, .7, .9]})
	# elastic_BO.maximize(n_iter=100)
	# print(elastic_BO.res['max'])

	# Random forest optimization
	# rf_BO = BayesianOptimization(rf_func, {'n_estimators': (1,1000), 'max_depth': (1,500)})
	# rf_BO.explore({'n_estimators': [25, 50, 100, 200, 400], 'max_depth':[10, 40, 80, 320, 500]})
	# rf_BO.maximize(n_iter=100)
	# print(rf_BO.res['max'])

	# svr_BO = BayesianOptimization(svr_func, {'C': (0, 10), 'epsilon':(0,10)})
	# svr_BO.explore({'C': [.001, 0.01, 0.1, 1, 10, 100, 1000], 'epsilon':[.001, .01, .1, 1, 10, 100, 1000]})
	# svr_BO.maximize(n_iter=100)
	# print(svr_BO.res['max'])

	# XGBoost optimization
	# xgb_BO = BayesianOptimization(xgb_func, {'min_child_weight': (1, 20),
 #                                            'colsample_bytree': (0.1, 1),
 #                                            'max_depth': (5, 15),
 #                                            'subsample': (0.5, 1),
 #                                            'gamma': (0, 10),
 #                                            'alpha': (0, 10),
 #                                            'num_rounds': (0, 10000),
 #                                            })

	# xgb_BO.maximize(init_points=5, n_iter=100)
	# print(xgb_BO.res['max'])

	# Lgb optimization
	lgb_BO = BayesianOptimization(lgb_func, {'num_leaves': (1, 20),
                                            'lr': (0.01, .05),
                                            'num_estimators': (5, 15),
                                            'max_bin': (2, 100),
                                            'bagging_fraction': (0, 1),
                                            'bagging_freq': (1, 10),
                                            'feature_fraction': (0, 1),
                                            'min_data_in_leaf': (1, 10),
                                            'min_sum_hessian_in_leaf': (1, 20),
                                            })
	lgb_BO.explore({'num_leaves': [5,5,5,4,5,6,7], 'lr':[.01, .01, .01, .01, .01, .01, .01],
			'num_estimators': [200, 300, 500, 700, 800, 900, 1000], 'max_bin':[10, 40, 80, 100, 55, 30, 53],
			'bagging_fraction': [.7, .8, .6, .2, .9, .7, .3], 'bagging_freq':[4,5,2,7,2,4,5],
			'feature_fraction': [.2, .3, .25, .3, .21, .15, .13], 'min_data_in_leaf':[5,6,5,4,4,7,7],
			'min_sum_hessian_in_leaf': [8,2,5,10,11,15,17]})
	lgb_BO.maximize(init_points=5, n_iter=100)
	print(lgb_BO.res['max'])

	# print(lasso_BO.res['max'])
	# print(rf_BO.res['max'])
	# print(krr_BO.res['max'])
	# print(xgb_BO.res['max'])
	# print(elastic_BO.res['max'])

def lasso_func(alpha):
	X_train, y_train, X_predict = import_data(name_data = "roger")
	val = cross_val_score(Lasso(alpha = alpha, random_state = 2), X_train, y_train, cv=3).mean()
	return val

def elastic_func(alpha, l1_ratio):
	X_train, y_train, X_predict = import_data(name_data = "roger")
	val = cross_val_score(ElasticNet(alpha = alpha, l1_ratio = l1_ratio, random_state = 2),
								 X_train, y_train, cv=3).mean()
	return val

def svr_func(C, epsilon):
	X_train, y_train, _ = import_data(name_data = "roger")
	warnings.filterwarnings('ignore')
	val = cross_val_score(SVR(C=C, epsilon=epsilon, kernel = 'rbf'), X_train, y_train, cv = 3).mean()
	return val

def rf_func(n_estimators, max_depth):
	X_train, y_train, X_predict = import_data(name_data = "roger")
	val = cross_val_score(RandomForestRegressor(n_estimators = int(n_estimators), max_depth = int(max_depth), random_state = 42),
											 X_train, y_train, cv=3).mean()
	return val


def krr_func(alpha, degree, coef0):
	X_train, y_train, X_predict = import_data(name_data = "roger")
	val = cross_val_score(KernelRidge(alpha=alpha, kernel='polynomial', degree=degree, coef0=coef0),
											 X_train, y_train, cv=3).mean()
	return val

def lgb_func(num_leaves, lr, num_estimators, max_bin, bagging_fraction, bagging_freq, 
						feature_fraction, min_data_in_leaf, min_sum_hessian_in_leaf):
	warnings.filterwarnings('ignore')
	X_train, y_train, X_predict = import_data(name_data = "roger")
	val = cross_val_score(lgb.LGBMRegressor(
				objective='regression',
				num_leaves=int(num_leaves),
                learning_rate=lr,
                n_estimators=int(num_estimators),
                max_bin = int(max_bin),
                bagging_fraction = bagging_fraction,
                bagging_freq = int(bagging_freq),
                feature_fraction = feature_fraction,
                feature_fraction_seed=9,
                bagging_seed=9,
                min_data_in_leaf=int(min_data_in_leaf),
                min_sum_hessian_in_leaf=min_sum_hessian_in_leaf), X_train, y_train, cv=3).mean()
	return val

def optimize_stack(X_train, y_train):

	# Subfunction provides xgboost function for X stack and y stack
	def xgb_func(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,num_rounds):
		warnings.filterwarnings('ignore')

		#X_train, y_train, X_predict = import_data(name_data = "roger")
		xgtrain = xgb.DMatrix(X_train, label=y_train)

		params = {
			'eta': 0.1,
			'silent': 1,
			'eval_metric': 'mae',
			'verbose_eval': True,
			'seed': 42
		}

		params['min_child_weight'] = int(min_child_weight)
		params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
		params['max_depth'] = int(max_depth)
		params['subsample'] = max(min(subsample, 1), 0)
		params['gamma'] = max(gamma, 0)
		params['alpha'] = max(alpha, 0)
		params['num_rounds'] = int(num_rounds)

		cv_result = xgb.cv(params, xgtrain, nfold=3, callbacks=[xgb.callback.early_stop(50)])

		return -cv_result['test-mae-mean'].values[-1]

	# XGBoost optimization
	xgb_BO = BayesianOptimization(xgb_func, {'min_child_weight': (1, 50),
                                            'colsample_bytree': (0.1, 1),
                                            'max_depth': (5, 15),
                                            'subsample': (0, 1),
                                            'gamma': (0, 100),
                                            'alpha': (0, 100),
                                            'num_rounds': (0, 10000),
                                            })

	xgb_BO.maximize(init_points=5, n_iter=100)
	print(xgb_BO.res['max'])

def xgb_stack_func(min_child_weight,colsample_bytree,max_depth,subsample,gamma,alpha,num_rounds):
	warnings.filterwarnings('ignore')


	#X_train, y_train, X_predict = import_data(name_data = "roger")

	xgtrain = xgb.DMatrix(X_train, label=y_train)

	params = {
		'eta': 0.1,
		'silent': 1,
		'eval_metric': 'mae',
		'verbose_eval': True,
		'seed': 42
	}

	params['min_child_weight'] = int(min_child_weight)
	params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
	params['max_depth'] = int(max_depth)
	params['subsample'] = max(min(subsample, 1), 0)
	params['gamma'] = max(gamma, 0)
	params['alpha'] = max(alpha, 0)
	params['num_rounds'] = int(num_rounds)

	cv_result = xgb.cv(params, xgtrain, nfold=3, callbacks=[xgb.callback.early_stop(50)])

	return -cv_result['test-mae-mean'].values[-1]

def import_data(name_data):

	# Below is roger's dataset 
	if name_data == "roger":
		train = pd.read_csv('./Data/train_120feats_Dense_OutlierFree_LogTransform.csv')
		test = pd.read_csv('./Data/test_119feats_Dense_OutlierFree_LogTransform.csv')

	if name_data == "roger1":
		train = pd.read_csv('./Data/KMeans_1_train_120feats_Dense_OutlierFree_NoTransform.csv')
		test = pd.read_csv('./Data/test_119feats_Dense_OutlierFree_LogTransform.csv')

	# Below is peter's dataset
	if name_data == "peter":
		train = pd.read_csv('./Data/train_linear.csv')
		test = pd.read_csv('./Data/test_linear.csv')

	# Below is andrew's dataset
	if name_data == "andrew":
		train = pd.read_csv('./Data/train_dummies.csv')
		test = pd.read_csv('./Data/test_dummies.csv')

	# Change dataframes to numpy arrays
	train = train.as_matrix()
	X_predict = test.as_matrix()

	# Assuming sales price is the last column...we separate out train_dummies
	# We don't need y_train here because it is the same for train_linear, so we'll get it there.
	last_col = train.shape[1]-1
	X_train = train[:,0:last_col]
	y_train = train[:,last_col]

	return X_train, y_train, X_predict

if __name__ == "__main__":
    # execute only if run as a script
    main()
    # optimize()



















