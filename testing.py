from sklearn.cross_validation import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import make_scorer, confusion_matrix, accuracy_score


def test_model_with_drug(model, drug, metrics):
    X, y = get_data_keeper().get_train_data(drug)
    y_pred = cross_val_predict(model, X, y, cv=StratifiedKFold(y, n_folds=10, shuffle=True, random_state=42))
    result = dict()
    if 'confusion matrix' in metrics:
        result['confusion matrix'] = confusion_matrix(y, y_pred)
    if 'accuracy' in metrics:
        result['accuracy'] = accuracy_score(y, y_pred)
    if 'feature count' in metrics:
        result['feature count'] = len(model.fit(X, y).get_support(indices=True))
    return result


def test_models_with_drugs(models, drugs, metrics=['confusion matrix', 'accuracy', 'feature count']):
    result = dict()
    for model_name, model in models:
        for drug_name in drugs:
            result[(model_name, drug_name)] = test_model_with_drug(model, drug_name, metrics)
    return result
