import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

def create_improved_model(input_shape):
    """
    Create an improved model with multiple layers and sigmoid activation.

    Args:
        input_shape (tuple): Shape of the input data.

    Returns:
        tf.keras.Model: Compiled TensorFlow Keras model.

    Raises:
        ValueError: If input_shape is invalid.
    """
    try:
        logger.info(f"Creating improved model with input shape {input_shape}")
        tf.keras.utils.set_random_seed(42)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=input_shape),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        logger.info("Model created successfully")
        return model
    except ValueError as ve:
        logger.error(f"ValueError while creating model: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def compile_model(model, learning_rate=0.0009):
    """
    Compile the model with binary crossentropy loss and SGD optimizer.

    Args:
        model (tf.keras.Model): The model to compile.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tf.keras.Model: Compiled model.

    Raises:
        ValueError: If model compilation fails.
    """
    try:
        logger.info(f"Compiling model with learning rate {learning_rate}")
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        logger.info("Model compiled successfully")
        return model
    except ValueError as ve:
        logger.error(f"ValueError while compiling model: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error compiling model: {str(e)}")
        raise

def train_model(model, x_train, y_train, epochs=100, use_lr_scheduler=True):
    """
    Train the model with optional learning rate scheduler.

    Args:
        model (tf.keras.Model): The model to train.
        x_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        epochs (int): Number of epochs to train.
        use_lr_scheduler (bool): Whether to use learning rate scheduler.

    Returns:
        tf.keras.callbacks.History: Training history.

    Raises:
        ValueError: If training fails.
    """
    try:
        logger.info(f"Training model for {epochs} epochs")
        callbacks = []
        if use_lr_scheduler:
            lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 0.001 * 0.9**(epoch/3)
            )
            callbacks.append(lr_scheduler)

        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks
        )
        logger.info("Model training completed successfully")
        return history
    except ValueError as ve:
        logger.error(f"ValueError during model training: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

def evaluate_model(model, x_test, y_test):
    """
    Evaluate the model on test data.

    Args:
        model (tf.keras.Model): The trained model.
        x_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: Test loss and accuracy.
    """
    try:
        logger.info("Evaluating model")
        return model.evaluate(x_test, y_test)
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def predict_model(model, x_test):
    """
    Make predictions using the trained model.

    Args:
        model (tf.keras.Model): The trained model.
        x_test (np.ndarray): Test data.

    Returns:
        np.ndarray: Rounded predictions.
    """
    try:
        logger.info("Making predictions")
        return tf.round(model.predict(x_test))
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        raise