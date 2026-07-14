import os
import numpy as np
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from grim import GreedyRuleInterpretableMachine

from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CRNGRIMClassifier:
    """
    Hybrid CRN + GRIM classifier with per-target saved models.

    Example:
        clf = CRNGRIMClassifier(
                target_name="Job_Priority",
                input_dim=X_train.shape[1],
                num_classes=len(np.unique(y))
        )
    """

    def __init__(self, target_name, input_dim, num_classes, model_dir="models"):
        self.target_name = str(target_name)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_dir = model_dir

        os.makedirs(model_dir, exist_ok=True)

        # File names INCLUDE target name
        self.crn_path = os.path.join(model_dir, f"{self.target_name}_crn_model.h5")
        self.grim_path = os.path.join(model_dir, f"{self.target_name}_grim_model.joblib.xz")

        self.crn_model = None
        self.grim_model = None

        self.best_model_name = None

    # ============================================================
    #  CRN MODEL (Deep Learning)
    # ============================================================
    def build_crn(self):
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Reshape((self.input_dim, 1))(inputs)

        x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.LSTM(32)(x)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs, name=f"CRN_{self.target_name}")
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ============================================================
    #  GRIM MODEL 
    # ============================================================
    def build_grim(self):
        return GreedyRuleInterpretableMachine()

    # ============================================================
    #  LOAD or TRAIN CRN
    # ============================================================
    def _load_or_train_crn(self, X, y, epochs=5, batch_size=32):
        if os.path.exists(self.crn_path):
            print(f"Loading CRN model for {self.target_name}...")
            self.crn_model = keras.models.load_model(self.crn_path)
        else:
            print(f"Training CRN model for {self.target_name}...")
            self.crn_model = self.build_crn()
            self.crn_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)
            self.crn_model.save(self.crn_path)
            print(f"CRN saved: {self.crn_path}")

    # ============================================================
    #  LOAD or TRAIN GRIM (COMPRESSED)
    # ============================================================
    def _load_or_train_grim(self, X, y):
        if os.path.exists(self.grim_path):
            print(f"Loading GRIM model for {self.target_name} (compressed)...")
            self.grim_model = load(self.grim_path)
        else:
            print(f"Training GRIM model for {self.target_name}...")
            self.grim_model = self.build_grim()
            self.grim_model.fit(X, y)
            dump(self.grim_model, self.grim_path, compress=("xz", 9))
            print(f"GRIM saved compressed: {self.grim_path}")

    # ============================================================
    #  Manual training (optional)
    # ============================================================
    def train_or_load(self, X, y, epochs=20, batch_size=32):
        self._load_or_train_crn(X, y, epochs, batch_size)
        self._load_or_train_grim(X, y)

    # ============================================================
    # fit() → ALWAYS SAFE: load missing → train missing → evaluate → return best
    # ============================================================
    def fit(self, X, y, epochs=20, batch_size=32):
        """
        Ensures CRN & GRIM exist, evaluates both, selects best, returns best.
        """

        self._load_or_train_crn(X, y, epochs, batch_size)
        self._load_or_train_grim(X, y)

        # Evaluate CRN
        crn_probs = self.crn_model.predict(X, verbose=0)
        crn_preds = np.argmax(crn_probs, axis=1)
        crn_acc = accuracy_score(y, crn_preds)

        # Evaluate GRIM
        grim_preds = self.grim_model.predict(X)
        grim_acc = accuracy_score(y, grim_preds)

        print(f"\nCRN Accuracy ({self.target_name})  = {crn_acc:.4f}")
        print(f"GRIM Accuracy ({self.target_name}) = {grim_acc:.4f}")

        # Select best model
        if grim_acc >= crn_acc:
            self.best_model_name = "grim"
            print(f"Selected Model → GRIM ({self.target_name})")
            return self.grim_model
        else:
            self.best_model_name = "crn"
            print(f"Selected Model → CRN ({self.target_name})")
            return self.crn_model

    # ============================================================
    #  Predict
    # ============================================================
    def predict(self, X):
        if self.best_model_name == "crn":
            return np.argmax(self.crn_model.predict(X, verbose=0), axis=1)
        elif self.best_model_name == "grim":
            return self.grim_model.predict(X)
        else:
            raise RuntimeError("Call fit() first — no best model selected.")

    # ============================================================
    #  Predict Probabilities
    # ============================================================
    def predict_proba(self, X):
        if self.best_model_name == "crn":
            return self.crn_model.predict(X, verbose=0)
        elif self.best_model_name == "grim":
            return self.grim_model.predict_proba(X)
        else:
            raise RuntimeError("Call fit() first.")
