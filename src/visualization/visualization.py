import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_learning_rate_vs_loss(history, epochs):
    """
    Plot the learning rate versus the loss.
    
    Args:
        history: History object from the model training.
        epochs: Number of epochs the model was trained for.
    """
    try:
        logger.info("Plotting learning rate vs. loss")
        lrs = 1e-5 * (10 ** (np.arange(epochs)/20))
        plt.figure(figsize=(10, 7))
        plt.semilogx(lrs, history.history["loss"])  # x-axis (learning rate) to be log scale
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning rate vs. loss")
        plt.show()
        logger.info("Learning rate vs. loss plot created successfully")
    except Exception as e:
        logger.error(f"Error while plotting learning rate vs. loss: {str(e)}")
        raise

def plot_training_curves(history):
    """
    Plot the training curves for accuracy and loss.
    
    Args:
        history: History object from the model training.
    """
    try:
        logger.info("Plotting training curves")
        pd.DataFrame(history.history).plot(figsize=(10, 7))
        plt.title("Training Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.show()
        logger.info("Training curves plot created successfully")
    except Exception as e:
        logger.error(f"Error while plotting training curves: {str(e)}")
        raise

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix for the predictions.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
    """
    try:
        logger.info("Plotting confusion matrix")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        logger.info("Confusion matrix plot created successfully")
    except Exception as e:
        logger.error(f"Error while plotting confusion matrix: {str(e)}")
        raise