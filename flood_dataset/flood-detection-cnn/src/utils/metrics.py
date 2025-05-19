def calculate_accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total

def calculate_precision(y_true, y_pred, average='binary'):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fp = ((y_true == 0) & (y_pred == 1)).sum().item()
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def calculate_recall(y_true, y_pred, average='binary'):
    tp = ((y_true == 1) & (y_pred == 1)).sum().item()
    fn = ((y_true == 1) & (y_pred == 0)).sum().item()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def calculate_f1_score(y_true, y_pred, average='binary'):
    precision = calculate_precision(y_true, y_pred, average)
    recall = calculate_recall(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def evaluate_model(y_true, y_pred):
    accuracy = calculate_accuracy(y_true, y_pred)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1_score(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }