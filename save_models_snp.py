
import os
from pathlib import Path
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import pickle
import numpy as np
from logger_config import logger

def save_model(model: tf.keras.Model,
               history: tf.keras.callbacks.History,
               current_path: str,
               target_dir: str,
               model_name: str,
               y_pred: np.ndarray,
               test_label: np.ndarray):
    # Ensure target directory exists
    target_dir_path = Path(current_path) / target_dir
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create model save path
    assert model_name.endswith(".h5"), "model_name should end with '.h5'"
    model_save_path = target_dir_path / model_name
    
    # Save the model
    # print(f"[INFO] Saving model to: {model_save_path}")
    logger.info("Saving model to: %s", model_save_path)
    model.save(model_save_path)
    
    # Save the training history
    with open(target_dir_path / 'basic_history.pickle', 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save the prediction and true labels
    np.save(target_dir_path / "y_pred.npy", y_pred)
    np.save(target_dir_path / "test_label.npy", test_label)
