def early_stopping_callback(patience=5):
    """
    Early stopping callback to stop training when validation loss does not improve.
    """
    best_loss = float('inf')
    epochs_without_improvement = 0

    def callback(current_loss):
        nonlocal best_loss, epochs_without_improvement
        if current_loss < best_loss:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                return True  # Stop training
        return False  # Continue training

    return callback

def model_checkpoint_callback(model, filepath, monitor='val_loss'):
    """
    Model checkpoint callback to save the model when validation loss improves.
    """
    best_loss = float('inf')

    def callback(current_loss):
        nonlocal best_loss
        if current_loss < best_loss:
            best_loss = current_loss
            torch.save(model.state_dict(), filepath)
            print(f"Model saved to {filepath}")
    
    return callback

def lr_scheduler_callback(optimizer, factor=0.1, patience=5):
    """
    Learning rate scheduler callback to reduce learning rate when a metric has stopped improving.
    """
    from collections import deque

    best_metric = float('inf')
    metrics_queue = deque(maxlen=patience)

    def callback(current_metric):
        nonlocal best_metric
        metrics_queue.append(current_metric)
        if len(metrics_queue) == patience and all(m >= best_metric for m in metrics_queue):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor
            best_metric = current_metric
            print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")

    return callback