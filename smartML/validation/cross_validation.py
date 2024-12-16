import numpy as np

def cross_validate(model, X, y, k=5):
    n_samples = len(y)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    fold_sizes = n_samples // k
    accuracies = []

    for fold in range(k):
        start = fold * fold_sizes
        end = (fold + 1) * fold_sizes if fold != k - 1 else n_samples
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)
        print(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}")
    
    print(f"\nAverage Accuracy: {np.mean(accuracies):.4f}")
    return np.mean(accuracies), accuracies
