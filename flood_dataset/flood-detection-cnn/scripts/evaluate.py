def evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from src.data.dataset import Dataset
    from src.models.cnn import CNN  # or any other model you want to evaluate
    from src.utils.metrics import accuracy_score, precision_score, recall_score, f1_score
    from tqdm import tqdm

    # Load your trained model
    model = CNN()  # Replace with your model initialization
    model.load_state_dict(torch.load('path_to_your_model.pth'))  # Load your trained model weights

    # Prepare your test dataset and dataloader
    test_dataset = Dataset('path_to_test_data')  # Replace with your test dataset path
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    metrics = evaluate_model(model, test_loader)