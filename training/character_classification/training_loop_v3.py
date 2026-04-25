import os
import pickle
import pprint
from typing import Any

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from compiler.character_classification.compiler_v3 import project_root, compile_data
from models.character_classification.ver_2 import build_model

torch.manual_seed(0)

VERSION = "v3"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dump = os.path.join(project_root, "results", "character_classification", f"training_loop_{VERSION}")
if not os.path.exists(dump):
    os.makedirs(dump)

checkpoint_path = os.path.join(dump, "checkpoint")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)


loss_function = torch.nn.CrossEntropyLoss()

def train (
        model : torch.nn.Module,
        training_data : DataLoader,
        validation_data : DataLoader,
        training_tracker : list = None,
        validation_tracker : list = None,
        accuracy : list[Any] = None,
        optimizer : torch.optim.Optimizer = None,
        TOTAL_EPOCHES : int = 100,
        THRESH : float = float('-inf'),
        verbose : bool = False,
        log_freq : int = 10,
        device: torch.device = torch.device('cpu')
        ):
    criterion = loss_function.to(device)
    model.to(device)

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    if training_tracker is None:
        training_tracker = []
    if validation_tracker is None:
        validation_tracker = []
    if accuracy is None:
        accuracy = []

    # Running statistics
    prev_loss = 100.

    for epoch in range(TOTAL_EPOCHES):
        if verbose and epoch % log_freq == 0:                    # Verbose Logging
            print(f"\t\tEpoch {epoch+1}/{TOTAL_EPOCHES}")

        # Training
        model.train()

        training_loss_per_epoch: list[float] = []

        for inputs, class_targets in training_data:

            inputs = inputs.to(device)
            class_targets = class_targets.to(device)

            # Reset Gradient / Forward Pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, class_targets)

            # Backward Pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            training_loss_per_epoch.append(loss.item())

        training_tracker.append(np.mean(training_loss_per_epoch))

        # Validation
        avg_loss = evaluate(model, validation_data, validation_tracker, accuracy, device=device)

        if epoch % log_freq == 0:
            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'training_loss': training_tracker[-1],
                'validation_loss': validation_tracker[-1],
            }
            torch.save(checkpoint, os.path.join(checkpoint_path, f"checkpoint_{epoch}.pth"))

            if verbose:                                         # Verbose Logging
                print(f"\t\t\ttraining avg_loss={training_tracker[-1]:.4f}")
                print(f"\t\t\tvalidation avg_loss={avg_loss:.4f}")
                print(f"\t\t\taccuracy={accuracy[-1]:.2f}%")

        if abs(prev_loss - avg_loss) < THRESH: #early-stopping condition
            break
        prev_loss = avg_loss
    return model

def evaluate(
        model: Module,
        validation_data: DataLoader,
        validation_tracker : list[Any] = None,
        accuracy: list[Any] = None,
        device: torch.device = torch.device('cpu')
) -> np.floating[Any]:
    """
    Evaluates CNN model on a given set of data to evaluate statistics of average loss and accuracy

    Args:
        model (torch.nn.Module) : The model to be evaluated
        validation_data (DataLoader) : Data used to evaluate the model
        validation_tracker (list[Any], optional) : Tracker used to evaluate the model. Done in place.
        If none provided, function creates its own list and discards it.
        accuracy (list[Any], optional) : Accuracy of the model. Done in place.
        If none provided, function creates its own list and discards it.
        device (torch.device) : Device used to evaluate the model.

    Return
        (floating[Any]): Average loss for batch
    """

    if validation_tracker is None:
        validation_tracker = list()
    if accuracy is None:
        accuracy = list()
    criterion = loss_function.to(device)

    model.eval()

    model = model.to(device)

    correct = 0
    total = 0
    validation_loss_per_epoch: list[float] = []

    with torch.no_grad():
        for inputs, class_targets in validation_data:
            inputs = inputs.to(device)
            class_targets = class_targets.to(device)

            # Forward Pass
            outputs = model(inputs)

            # Compute Loss
            loss = criterion(outputs, class_targets)
            validation_loss_per_epoch.append(loss.item())

            # Apply 1-hot max to classify
            predicted = torch.max(outputs, 1)[1]
            correct += (predicted == class_targets).sum().item()
            total += class_targets.size(0)

    # In place operations
    accuracy.append(100. * correct / total)  # convert to percentage
    avg_loss = np.mean(validation_loss_per_epoch)
    validation_tracker.append(avg_loss)
    return avg_loss


def training_loop():
    print(f"Device detected: {torch.cuda.get_device_name()}")

    print("Compiling dataset...")
    training_data, validation_data = compile_data()

    print("Building model...")
    model = build_model()

    model_training, model_validation, model_validation_accuracy = [], [], []
    epsilon = float('-inf')

    print("Training model...")
    model = train(model, training_data, validation_data,
                  training_tracker=model_training, validation_tracker=model_validation, accuracy=model_validation_accuracy,
                  verbose=True, TOTAL_EPOCHES=150, THRESH=epsilon, device=DEVICE,
                  #optimizer=torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4) # Change optimizer if results no good
                  )
    print("Training complete.")

    print("Evaluating model...")
    model_testing, model_testing_accuracy = [], []

    evaluate(model, validation_data, model_testing, accuracy=model_testing_accuracy, device=DEVICE)
    print("Testing complete.")

    print("Saving results...")

    results = {
        "training"              : model_training,
        "validation"            : model_validation,
        "validation accuracy"   : model_validation_accuracy,

        "testing"               : model_testing,
        "testing accuracy"      : model_testing_accuracy,
    }

    pprint.pprint(results)
    print("\nNote: Validation Samples == Testing samples")

    torch.save(model.state_dict(), os.path.join(dump, "final_model.pth"))

    with open(os.path.join(dump, "training_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    return 0

if __name__ == "__main__":
    training_loop()