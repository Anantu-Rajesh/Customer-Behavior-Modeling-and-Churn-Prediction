'''
from src import config
from src.models import util
from src.data_preprocessing import load_data as ld
'''
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def _ensemble_predict_proba(ensemble, X, X_tree):
    avg = (
        ensemble['nb'].predict_proba(X)[:, 1] +
        ensemble['svm'].predict_proba(X)[:, 1] +
        ensemble['rf'].predict_proba(X_tree)[:, 1] +
        ensemble['xgb'].predict_proba(X_tree)[:, 1]
    ) / 4.0
    return avg

def _ensemble_predict(ensemble, X, X_tree):
    return (_ensemble_predict_proba(ensemble, X, X_tree) >= ensemble['threshold']).astype(int)


def churn(X_train, y_train, X_test, y_test, X_train_tree, y_train_tree, X_test_tree):
    nb = GaussianNB(var_smoothing=1e-10)
    nb.fit(X_train, y_train)

    svm = SVC(C=0.1, gamma=0.001, kernel='rbf', probability=True)
    svm.fit(X_train, y_train)

    rf = RandomForestClassifier(max_depth=5, max_features='sqrt', min_samples_leaf=5, min_samples_split=2, n_estimators=200)
    rf.fit(X_train_tree, y_train_tree)

    xgb = XGBClassifier(learning_rate=0.01, max_depth=3, n_estimators=200, scale_pos_weight=2)
    xgb.fit(X_train_tree, y_train_tree)

    ensemble = {'nb': nb, 'svm': svm, 'rf': rf, 'xgb': xgb, 'threshold': 0.38}

    y_prob = _ensemble_predict_proba(ensemble, X_test, X_test_tree)
    y_pred = _ensemble_predict(ensemble, X_test, X_test_tree)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results = [{'F1 Score': f1, 'ROC AUC': roc_auc}]
    print(f"Ensemble: Accuracy={acc}, F1 Score={f1}, ROC AUC={roc_auc}\n")

    return ensemble, results

'''if __name__ == "__main__":
    df_labels=ld.load_and_describe_data(config.customer_nlp_filepath_with_labels)
    X_train_linear, X_test_linear, y_train_linear, y_test_linear,scaler=util.churn_data(df_labels,model_type='linear')
    churn_model, churn_results=churn(X_train_linear, y_train_linear, X_test_linear, y_test_linear)'''