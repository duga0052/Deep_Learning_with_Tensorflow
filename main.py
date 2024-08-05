import logging
import sys
from src.data.data_processing import load_data, preprocess_data
from src.model.improved_model import create_improved_model, compile_model, train_model, evaluate_model, predict_model
from sklearn.metrics import accuracy_score
from src.visualization.visualization import plot_training_curves, plot_confusion_matrix, plot_learning_rate_vs_loss

def setup_logging(log_file='app.log', console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Set up logging configuration.

    Args:
        log_file (str): Name of the log file.
        console_level (int): Logging level for console output.
        file_level (int): Logging level for file output.

    Returns:
        logging.Logger: Configured logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(file_level)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    file_handler.setFormatter(file_format)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def main():
    # Initialize logging
    logger = setup_logging()

    try:
        logger.info("Starting employee attrition prediction process")

        # Load data
        df = load_data('employee_attrition.csv')

        # Preprocess data
        x_train, x_test, y_train, y_test = preprocess_data(df)

        # Create and compile model
        model = create_improved_model(input_shape=(x_train.shape[1],))
        model = compile_model(model, learning_rate=0.0009)

        # Train model
        history = train_model(model, x_train, y_train, epochs=100, use_lr_scheduler=True)

        # Evaluate model
        loss, accuracy = evaluate_model(model, x_test, y_test)
        logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

        # Predict on test data
        y_preds = predict_model(model, x_test)
        acc_score = accuracy_score(y_test, y_preds)
        logger.info(f"Prediction Accuracy: {acc_score:.4f}")

        # Plot results
        plot_training_curves(history)
        plot_confusion_matrix(y_test, y_preds)
        plot_learning_rate_vs_loss(history, epochs=100)

        logger.info("Employee attrition prediction process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during the process: {str(e)}")
        raise

if __name__ == "__main__":
    main()