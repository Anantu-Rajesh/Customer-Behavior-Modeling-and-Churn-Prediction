org_filepath='./data/raw/online_retail.xlsx'
customer_filepath='./data/processed/customer_features.csv'
customer_filepath_with_unsupervised_labels='./data/processed/customer_features_with_labels.csv'

CHURN_PARAMS = {
    'naive_bayes': {
        'var_smoothing': 1e-10
    },
    'random_forest': {
        'n_estimators': 600,
        'max_depth': 5,
        'max_features': 'log2',
        'min_samples_leaf': 2,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.01,
        'max_depth': 3,
        'scale_pos_weight': 2,
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0
    }
}

HIGH_VALUE_PARAMS = {
    'random_forest': {
        'n_estimators': 400,
        'max_depth': 10,
        'max_features': 'log2',
        'min_samples_leaf': 5,
        'min_samples_split': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.01,
        'max_depth': 3,
        'scale_pos_weight': 2,
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0
    }
}

HIGH_RISK_PARAMS = {
    'random_forest': {
        'n_estimators': 400,
        'max_depth': 10,
        'max_features': 'log2',
        'min_samples_leaf': 5,
        'min_samples_split': 2,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.01,
        'max_depth': 3,
        'scale_pos_weight': 3,
        'use_label_encoder': False,
        'random_state': 42,
        'verbosity': 0
    }
}