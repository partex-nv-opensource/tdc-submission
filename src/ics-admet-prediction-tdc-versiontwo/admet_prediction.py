"""
Author: Rohit Singhy Yadav
Organization: Partex NV
"""

from tdc.benchmark_group import admet_group
import optuna
from optuna.samplers import TPESampler
import numpy as np
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from fingerprint_gen import * 
import lightgbm as lgb
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, TransformerMixin
from tdc import Evaluator

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("model_output_v2.log"), 
                        logging.StreamHandler()
                    ])
logger = logging.getLogger()

group = admet_group(path='data/')


benchmark_settings = {
    # 'solubility_aqsoldb': ('regression', False, 'MAE', True),
    'cyp2c9_veith': ('binary', True, 'PR-AUC', False),        
    'cyp2d6_veith': ('binary', True, 'PR-AUC', False),        
    'cyp3a4_veith': ('binary', True, 'PR-AUC', False),        
    'cyp2c9_substrate_carbonmangels': ('binary', True, 'PR-AUC', False), 
    'cyp2d6_substrate_carbonmangels': ('binary', True, 'PR-AUC', False), 
    'cyp3a4_substrate_carbonmangels': ('binary', True, 'ROC-AUC', False), 
    'caco2_wang': ('regression', False, 'MAE', True),
    'bioavailability_ma': ('binary', True, 'ROC-AUC', False), 
    'lipophilicity_astrazeneca': ('regression', False, 'MAE', True),
    'hia_hou': ('binary', True, 'ROC-AUC', False),             
    'pgp_broccatelli': ('binary', True, 'ROC-AUC', False),    
    'bbb_martins': ('binary', True, 'ROC-AUC', False),        
    'ppbr_az': ('regression', False, 'MAE', True),
    'vdss_lombardo': ('regression', False, 'Spearman', False),
    'half_life_obach': ('regression', False, 'Spearman', False),
    'clearance_hepatocyte_az': ('regression', False, 'Spearman', False),
    'clearance_microsome_az': ('regression', False, 'Spearman', False),
    'ld50_zhu': ('regression', False, 'MAE', True),
    'herg': ('binary', True, 'ROC-AUC', False),               
    'ames': ('binary', True, 'ROC-AUC', False),               
    'dili': ('binary', True, 'ROC-AUC', False)                
    
}

# Define which model to use for each benchmark
model_mapping = {
    # 'solubility_aqsoldb': 'LightGBM',
    'cyp2c9_veith': 'XGBoost',
    'cyp2d6_veith': 'LightGBM',
    'cyp3a4_veith': 'CatBoost',
    'cyp2c9_substrate_carbonmangels': 'FFN',
    'cyp2d6_substrate_carbonmangels': 'FFN',
    'cyp3a4_substrate_carbonmangels': 'XGBoost',
    'caco2_wang': 'XGBoost',
    'bioavailability_ma': 'LightGBM',
    'lipophilicity_astrazeneca': 'XGBoost',
    'hia_hou': 'CatBoost',
    'pgp_broccatelli': 'FFN',
    'bbb_martins': 'FFN',
    'ppbr_az': 'XGBoost',
    'vdss_lombardo': 'XGBoost',
    'half_life_obach': 'XGBoost',
    'clearance_hepatocyte_az': 'XGBoost',
    'clearance_microsome_az': 'LightGBM',
    'ld50_zhu': 'XGBoost',
    'herg': 'XGBoost',
    'ames': 'XGBoost',
    'dili': 'XGBoost'
}

class LogScaler(BaseEstimator, TransformerMixin):
    def __init__(self, apply_log=True):
        self.apply_log = apply_log

    def fit(self, y):
        if self.apply_log:
            self.min_val_ = np.min(y)
            if self.min_val_ <= 0:
                self.offset_ = -self.min_val_ + 1e-8 
            else:
                self.offset_ = 0
        return self

    def transform(self, y):
        if self.apply_log:
            y_transformed = y + self.offset_
            y_transformed[y_transformed <= 0] = 1e-8
            return np.log(y_transformed)
        else:
            return y

    def inverse_transform(self, y):
        if self.apply_log:
            return np.exp(y) - self.offset_
        else:
            return y

def evaluate_regression(y_true, y_pred):
    metrics = {}
    for metric_name in ['MSE', 'RMSE', 'MAE', 'R2', 'PCC', 'Spearman']:
        evaluator = Evaluator(name=metric_name)
        try:
            metrics[metric_name] = evaluator(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Failed to compute {metric_name}: {str(e)}")
            metrics[metric_name] = np.nan
    return metrics

def evaluate_classification(y_true, y_pred):
    metrics = {}
    for metric_name in ['ROC-AUC', 'PR-AUC', 'range_logAUC', 'Accuracy',
                       'Precision', 'Recall', 'F1']:
        evaluator = Evaluator(name=metric_name)
        try:
            if metric_name == 'Accuracy':
                metrics[metric_name] = evaluator(y_true, (y_pred > 0.5).astype(int))
            else:
                metrics[metric_name] = evaluator(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Failed to compute {metric_name}: {str(e)}")
            metrics[metric_name] = np.nan
    return metrics

def get_objective_value(eval_score, should_minimize_metric):
    """Helper to return score or -score for Optuna based on metric type."""
    if should_minimize_metric:
        return eval_score
    else:
        return -eval_score 

def optimize_xgb_regression(trial, train_features, train_target, val_features, val_target, metric_name, should_minimize):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'verbosity': 0
    }
    model = XGBRegressor(**params)
    model.fit(train_features, train_target)
    predictions = model.predict(val_features)
    evaluator = Evaluator(name=metric_name)
    score = evaluator(val_target, predictions)
    return get_objective_value(score, should_minimize)

def optimize_lgb_regression(trial, train_features, train_target, val_features, val_target, metric_name, should_minimize):
    params = {
        'objective': 'regression_l1' if metric_name == 'MAE' else 'regression_l2', 
        'metric': 'mae' if metric_name == 'MAE' else ('rmse' if metric_name == 'RMSE' else 'l2'), 
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(train_features, train_target)
    predictions = model.predict(val_features)
    evaluator = Evaluator(name=metric_name)
    score = evaluator(val_target, predictions)
    return get_objective_value(score, should_minimize)

def optimize_catboost_regression(trial, train_features, train_target, val_features, val_target, metric_name, should_minimize):
    params = {
        'loss_function': 'MAE' if metric_name == 'MAE' else ('RMSE' if metric_name == 'RMSE' else 'PairLogit'), 
        'eval_metric': metric_name if metric_name in ['MAE', 'RMSE'] else 'PairLogit', 
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'verbose': 0
    }
    model = CatBoostRegressor(**params, task_type="GPU")
    model.fit(train_features, train_target) 
    predictions = model.predict(val_features)
    evaluator = Evaluator(name=metric_name)
    score = evaluator(val_target, predictions)
    return get_objective_value(score, should_minimize)

def optimize_xgb_classification(trial, train_features, train_target, val_features, val_target, metric_name, should_minimize):
    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'eval_metric': 'auc' if metric_name == 'ROC-AUC' else ('aucpr' if metric_name == 'PR-AUC' else 'logloss'),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'verbosity': 0
    }
    model = XGBClassifier(**params)
    model.fit(train_features, train_target, eval_set=[(val_features, val_target)], verbose=False)
    predictions_proba = model.predict_proba(val_features)[:, 1]
    evaluator = Evaluator(name=metric_name)
    score = evaluator(val_target, predictions_proba)
    return get_objective_value(score, should_minimize)

def optimize_lgb_classification(trial, train_features, train_target, val_features, val_target, metric_name, should_minimize):
    params = {
        'objective': 'binary',
        'metric': 'auc' if metric_name == 'ROC-AUC' else ('aucpr' if metric_name == 'PR-AUC' else 'binary_logloss'),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 200),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'verbose': -1
    }
    model = lgb.LGBMClassifier(**params)
    model.fit(train_features, train_target, eval_set=[(val_features, val_target)])
    predictions_proba = model.predict_proba(val_features)[:, 1]
    evaluator = Evaluator(name=metric_name)
    score = evaluator(val_target, predictions_proba)
    return get_objective_value(score, should_minimize)
    
def optimize_catboost_classification(trial, train_features, train_target, val_features, val_target, metric_name, should_minimize):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC' if metric_name == 'ROC-AUC' else ('PRAUC' if metric_name == 'PR-AUC' else 'Logloss'),
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'verbose': 0
    }
    model = CatBoostClassifier(**params, task_type="GPU")
    model.fit(train_features, train_target, eval_set=[(val_features, val_target)], verbose=False)
    predictions_proba = model.predict_proba(val_features)[:, 1]
    evaluator = Evaluator(name=metric_name)
    score = evaluator(val_target, predictions_proba)
    return get_objective_value(score, should_minimize)

class FFN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, task_type):
        super(FFN, self).__init__()
        self.task_type = task_type
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() if task_type == 'binary' else None

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.task_type == 'binary':
            out = self.sigmoid(out)
        return out

def train_ffn(model, train_loader, val_loader, criterion, optimizer, device, epochs=10,
              optimization_metric_name=None, should_minimize_metric=None): 
    best_val_metric = float('inf') if (should_minimize_metric is True) else float('-inf')
    
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if model.task_type == 'binary':
                loss = criterion(outputs.squeeze(), targets.float())
            else:
                loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()


    if val_loader is not None and optimization_metric_name is not None and should_minimize_metric is not None:
        model.eval()
        val_preds, val_targets_list = [], []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_preds.extend(outputs.cpu().numpy())
                val_targets_list.extend(targets.cpu().numpy())

        val_preds = np.array(val_preds).squeeze()
        val_targets_arr = np.array(val_targets_list)
        
        evaluator = Evaluator(name=optimization_metric_name)
        current_val_metric = evaluator(val_targets_arr, val_preds)
        return get_objective_value(current_val_metric, should_minimize_metric)
    return None 

def optimize_ffn(trial, train_features, train_target, val_features, val_target, task_type, metric_name, should_minimize):
    hidden_size = trial.suggest_int('hidden_size', 64, 512, log=True)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    ffn_epochs = trial.suggest_int('ffn_epochs', 10, 50) 

    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32),
                                torch.tensor(train_target, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(val_features, dtype=torch.float32),
                              torch.tensor(val_target, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFN(input_size=train_features.shape[1], hidden_size=hidden_size,
                output_size=1, task_type=task_type).to(device)
    criterion = nn.BCELoss() if task_type == 'binary' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return train_ffn(model, train_loader, val_loader, criterion, optimizer, device, epochs=ffn_epochs,
                     optimization_metric_name=metric_name, should_minimize_metric=should_minimize)


for benchmark_name, (task_type, log_transform, leaderboard_metric, should_minimize_leaderboard_metric) in benchmark_settings.items():
    predictions_list = []
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting benchmark: {benchmark_name}")
    logger.info(f"Task type: {task_type}, Log transform: {log_transform}")
    logger.info(f"Leaderboard metric for Optuna: {leaderboard_metric} (Minimize: {should_minimize_leaderboard_metric})")
    logger.info(f"{'='*50}\n")

    for seed in tqdm([1, 2, 3, 4, 5], desc=f"Running {benchmark_name}"):
        logger.info(f"\n{'='*30}")
        logger.info(f"Running seed: {seed}")
        logger.info(f"{'='*30}")

        benchmark = group.get(benchmark_name)
        predictions_dict_for_seed = {} 
        name = benchmark['name']
        train_val, test_df = benchmark['train_val'], benchmark['test'] 
        train_df, valid_df = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed) 

        train_features = generate_fingerprints(train_df['Drug'])
        valid_features = generate_fingerprints(valid_df['Drug'])
        test_features = generate_fingerprints(test_df['Drug'])

        imputer = SimpleImputer(strategy='mean')
        train_features = imputer.fit_transform(train_features)
        valid_features = imputer.transform(valid_features)
        test_features = imputer.transform(test_features)

        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        valid_features = scaler.transform(valid_features)
        test_features = scaler.transform(test_features)

        model_name_for_benchmark = model_mapping[benchmark_name] 
        logger.info(f"Using model: {model_name_for_benchmark} for benchmark {benchmark_name} with seed {seed}")

        optuna_direction = 'minimize'

        if task_type == 'regression':
            target_scaler = LogScaler(apply_log=log_transform)
            target_scaler.fit(train_df['Y'].values)
            train_target = target_scaler.transform(train_df['Y'].values)
            valid_target = target_scaler.transform(valid_df['Y'].values)

            if model_name_for_benchmark == 'XGBoost':
                study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                study.optimize(lambda trial: optimize_xgb_regression(
                    trial, train_features, train_target, valid_features, valid_target,
                    leaderboard_metric, should_minimize_leaderboard_metric), n_trials=20)
                final_model = XGBRegressor(**study.best_params, tree_method='gpu_hist')
            elif model_name_for_benchmark == 'LightGBM':
                study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                study.optimize(lambda trial: optimize_lgb_regression(
                    trial, train_features, train_target, valid_features, valid_target,
                    leaderboard_metric, should_minimize_leaderboard_metric), n_trials=20)
                final_model = lgb.LGBMRegressor(**study.best_params)
            elif model_name_for_benchmark == 'CatBoost':
                study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                study.optimize(lambda trial: optimize_catboost_regression(
                    trial, train_features, train_target, valid_features, valid_target,
                    leaderboard_metric, should_minimize_leaderboard_metric), n_trials=10)
                final_model = CatBoostRegressor(**study.best_params, task_type="GPU", verbose=0)
            
            final_model.fit(train_features, train_target)
            y_pred_test = target_scaler.inverse_transform(final_model.predict(test_features)).reshape(-1)
            predictions_dict_for_seed[name] = y_pred_test

            metrics = evaluate_regression(test_df['Y'].values, y_pred_test)
            logger.info(f"Seed {seed} - Regression Metrics (Leaderboard: {leaderboard_metric}):")
            for metric_key, value in metrics.items(): 
                logger.info(f"{metric_key}: {value:.4f}")

        elif task_type == 'binary':
            train_target = train_df['Y'].values
            valid_target = valid_df['Y'].values
            test_target = test_df['Y'].values

            if model_name_for_benchmark == 'FFN':
                study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                study.optimize(lambda trial: optimize_ffn(
                    trial, train_features, train_target, valid_features, valid_target, task_type,
                    leaderboard_metric, should_minimize_leaderboard_metric), n_trials=20)

                best_params = study.best_params
                train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32),
                                           torch.tensor(train_target, dtype=torch.float32))

                final_ffn_epochs = best_params.pop('ffn_epochs', 20) 

                train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                final_model = FFN(input_size=train_features.shape[1],
                           hidden_size=best_params['hidden_size'],
                           output_size=1, task_type=task_type).to(device)
                criterion = nn.BCELoss()
                optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

                final_model.train()
                for epoch in range(final_ffn_epochs):
                    for inputs, targets_batch in train_loader:
                        inputs, targets_batch = inputs.to(device), targets_batch.to(device)
                        optimizer.zero_grad()
                        outputs = final_model(inputs)
                        loss = criterion(outputs.squeeze(), targets_batch.float())
                        loss.backward()
                        optimizer.step()

                test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32))
                test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
                final_model.eval()
                y_pred_test_list = [] 
                with torch.no_grad():
                    for inputs_test in test_loader: 
                        inputs_test = inputs_test[0].to(device)
                        outputs = final_model(inputs_test)
                        y_pred_test_list.extend(outputs.cpu().numpy())
                y_pred_test = np.array(y_pred_test_list).squeeze()
                predictions_dict_for_seed[name] = y_pred_test

            else: 
                if model_name_for_benchmark == 'XGBoost':
                    study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                    study.optimize(lambda trial: optimize_xgb_classification(
                        trial, train_features, train_target, valid_features, valid_target,
                        leaderboard_metric, should_minimize_leaderboard_metric), n_trials=20)
                    final_model = XGBClassifier(**study.best_params, tree_method='gpu_hist', use_label_encoder=False) 
                elif model_name_for_benchmark == 'LightGBM':
                    study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                    study.optimize(lambda trial: optimize_lgb_classification(
                        trial, train_features, train_target, valid_features, valid_target,
                        leaderboard_metric, should_minimize_leaderboard_metric), n_trials=20)
                    final_model = lgb.LGBMClassifier(**study.best_params)
                elif model_name_for_benchmark == 'CatBoost':
                    study = optuna.create_study(direction=optuna_direction, sampler=TPESampler(seed=seed))
                    study.optimize(lambda trial: optimize_catboost_classification(
                        trial, train_features, train_target, valid_features, valid_target,
                        leaderboard_metric, should_minimize_leaderboard_metric), n_trials=10)
                    final_model = CatBoostClassifier(**study.best_params, task_type="GPU", verbose=0)

                final_model.fit(train_features, train_target)
                y_pred_test = final_model.predict_proba(test_features)[:, 1]
                predictions_dict_for_seed[name] = y_pred_test

            metrics = evaluate_classification(test_target, y_pred_test)
            logger.info(f"Seed {seed} - Classification Metrics (Leaderboard: {leaderboard_metric}):")
            for metric_key, value in metrics.items(): 
                logger.info(f"{metric_key}: {value:.4f}")

        predictions_list.append(predictions_dict_for_seed)

    results = group.evaluate_many(predictions_list)
    logger.info(f"\n{'='*50}")
    logger.info(f"Final results for {benchmark_name} (optimized for {leaderboard_metric}):")
    logger.info(f"{results}")
    logger.info(f"{'='*50}\n")
