import sys
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import matplotlib.pyplot as plt

from model import LinearRegressionModel

def train_model(dataset_path, csv_path, epochs, batch_size, learning_rate, early_stopping_counter):
    """
    Trains a linear regression model.
    
    Args:
        dataset_path (Callable): Returns a dataset instance.
        csv_path (str): The path to the CSV file containing the dataset.
        epochs (int): The number of episodes to train the model.
        batch_size (int): Batch size during training.
        learning_rate (float): The learning rate for the optimizer.
        early_stopping_counter (int): The number of epochs to wait for improvement before stopping.

    Returns:
        tuple: A tuple containing:
            - model (LinearRegressionModel): The trained model.
            - test_loader (DataLoader): The DataLoader for the test set.
            - train_losses (list): Training losses per epoch.
            - value_losses (list): Validation losses per epoch.
    """

    dataset = dataset_path(csv_path)

    # train-test split dataset
    train_split = int(0.7 * len(dataset))
    test_split = len(dataset) - train_split
    train_dataset, test_dataset = random_split(dataset, [train_split, test_split])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #import model
    input_dim = dataset.features.shape[1]
    model = LinearRegressionModel(input_dim)

    loss_fn = nn.MSELoss() #loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) #optimizer

    train_losses = []
    value_losses = []
    best_loss = float('inf')
    stop_counter = 0

    #training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).reshape(-1)

            y_batch = y_batch.reshape(-1)
            loss = loss_fn(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        train_losses.append(running_loss / len(train_loader.dataset))
        print(f"Epoch [{epoch + 1}/{epochs}], Loss = {train_losses[-1]:.4f}")

        #testing loop
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X_test, y_test in test_loader:
                y_test_pred = model(X_test).reshape(-1)
                val_loss = loss_fn(y_test_pred, y_test).item()
                test_loss += val_loss * X_test.size(0)
        
        value_losses.append(test_loss / len(test_loader.dataset))
        print(f"Test loss: {test_loss:.4f}") #the average test loss

        #early stopping to prevent overfitting
        if value_losses[-1] < best_loss:
            best_loss = value_losses[-1]
            stop_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved. Saving model.")
        else:
            stop_counter += 1
            print(f"No improvement for {stop_counter} epochs.")
            if stop_counter >= early_stopping_counter:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))

    return model, test_loader, train_losses, value_losses

def evaluate_model(model, test_loader, loss_fn):
    """
    Evaluate the models performance on the test set.
    Computes the predictions for analysis.
    """
    model.eval()
    test_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_test, y_test in test_loader:
            test_predictions = model(X_test).squeeze(1)
            y_test = y_test.view(-1)

            t_loss = loss_fn(test_predictions, y_test)
            test_loss += t_loss.item() * X_test.size(0)

            predictions.extend(test_predictions.tolist())
            actuals.extend(y_test.tolist())

    # Compute average test loss
    test_loss /= len(test_loader.dataset)
    print(f"Final Test Loss: {test_loss:.4f}")

    predictions = torch.tensor(predictions)
    actuals = torch.tensor(actuals)
    
    predictions_scaled = predictions.numpy().reshape(-1, 1)
    actuals_scaled = actuals.numpy().reshape(-1, 1)

    scaler = test_loader.dataset.dataset.scaler
    unscaled_predictions = scaler.inverse_transform(predictions_scaled)
    unscaled_actuals = scaler.inverse_transform(actuals_scaled)

    original_predictions = np.exp(unscaled_predictions)
    original_actuals = np.exp(unscaled_actuals)

    #compute metrics
    mae = np.mean(np.abs(original_predictions - original_actuals))
    mape = np.mean(np.abs((original_predictions - original_actuals) / original_actuals)) * 100
    r2 = 1 - np.sum((original_predictions - original_actuals) ** 2) / \
            np.sum((original_actuals - np.mean(original_actuals)) ** 2)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")
    
    for i in range(5):
        print(
            f"Actual: {original_actuals[i][0]:.2f}, "
            f"Predicted: {original_predictions[i][0]:.2f}"
        )

    return original_predictions, original_actuals, test_loss, mae, mape, r2

def plot(train_losses, value_losses, actuals, predictions):
    """
    Will return plots for:
    1.training - validation loss
    2.actual - predicted prices
    """
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(value_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.5, label="Actual vs Predicted")
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='dashed', label="Perfect Prediction")
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.title("Actual vs Predicted House Prices")

    plt.show()

if __name__ == "__main__":
    path = input("Pick a dataset: (1) Dutch House, (2) General House: ")
    if path == "1":
        from preprocessing_dutch_house import DutchHousePriceDataset as Dataset
        csv_path = "datasets/dutch_house_prices.csv"
    elif path == "2":
        from preprocessing_gen_house import GeneralHousePriceDataset as Dataset
        csv_path = "datasets/house_prices.csv"
    else:
        print("Invalid input.")
        sys.exit()

    model, test_loader, train_losses, value_losses = train_model(
        dataset_path=Dataset,
        csv_path=csv_path,
        epochs=100,
        batch_size=64,
        learning_rate=5e-3,
        early_stopping_counter=5)
    print("Training complete with enhancements.")

    print("Evaluating model...")
    original_predictions, original_actuals, test_loss, mae_original, mse_original, r2_original = evaluate_model(
        model, test_loader, nn.MSELoss()
    )

    print("Plotting...")
    plot(train_losses, value_losses, original_actuals, original_predictions)
    
    print("Done.")