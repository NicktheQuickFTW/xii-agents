#!/usr/bin/env python3
"""
Machine Learning Utilities for Single-File Agents
Purpose: Provides simple ML capabilities that agents can use to learn from experience
Version: 1.0.0

Usage:
  Import into agent scripts to provide learning capabilities
  
Requirements:
  - Python 3.8+
  - Install deps: uv pip install scikit-learn numpy pandas
"""

import os
import sys
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

# Try importing optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_squared_error, f1_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Import memory_store if available
try:
    from memory_store import get_memory_store, MemoryStore
    MEMORY_STORE_AVAILABLE = True
except ImportError:
    MEMORY_STORE_AVAILABLE = False

class ModelNotAvailableError(Exception):
    """Raised when required ML libraries are not available"""
    pass

class SimpleModel:
    """Base class for simple ML models"""
    
    def __init__(self, model_type: str, model_params: Optional[Dict[str, Any]] = None):
        """Initialize the model"""
        if not SKLEARN_AVAILABLE:
            raise ModelNotAvailableError(
                "scikit-learn is not installed. Install with: pip install scikit-learn"
            )
        
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_model(self):
        """Create the appropriate model based on type"""
        if self.model_type == "random_forest_classifier":
            return RandomForestClassifier(**self.model_params)
        elif self.model_type == "random_forest_regressor":
            return RandomForestRegressor(**self.model_params)
        elif self.model_type == "logistic_regression":
            return LogisticRegression(**self.model_params)
        elif self.model_type == "linear_regression":
            return LinearRegression(**self.model_params)
        elif self.model_type == "svm_classifier":
            return SVC(**self.model_params)
        elif self.model_type == "svm_regressor":
            return SVR(**self.model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: Union[List[List[float]], np.ndarray], y: Union[List[Any], np.ndarray]) -> float:
        """
        Train the model on the given data
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Performance metric (accuracy for classification, RMSE for regression)
        """
        # Convert to numpy arrays if needed
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        
        # Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate performance
        if self._is_classifier():
            y_pred = self.model.predict(X_test_scaled)
            return accuracy_score(y_test, y_pred)
        else:
            y_pred = self.model.predict(X_test_scaled)
            return np.sqrt(mean_squared_error(y_test, y_pred))
    
    def predict(self, X: Union[List[float], List[List[float]], np.ndarray]) -> Union[Any, List[Any]]:
        """
        Make predictions using the trained model
        
        Args:
            X: Features for prediction (single instance or batch)
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not trained yet. Call fit() first.")
        
        # Handle single instance vs batch
        single_instance = False
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            single_instance = True
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Return single prediction or list based on input
        if single_instance:
            return predictions[0]
        else:
            return predictions.tolist()
    
    def _is_classifier(self) -> bool:
        """Check if the model is a classifier"""
        return self.model_type in ["random_forest_classifier", "logistic_regression", "svm_classifier"]
    
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'model_params': self.model_params,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'SimpleModel':
        """
        Load a model from disk
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['model_type'], data['model_params'])
        model.model = data['model']
        model.scaler = data['scaler']
        model.is_fitted = data['is_fitted']
        
        return model

class LearningAgent:
    """Agent that can learn from experience using simple ML models"""
    
    def __init__(self, agent_id: str, memory_store: Optional[MemoryStore] = None):
        """Initialize the learning agent"""
        if not SKLEARN_AVAILABLE:
            raise ModelNotAvailableError(
                "scikit-learn is not installed. Install with: pip install scikit-learn"
            )
        
        self.agent_id = agent_id
        
        # Set up memory store if available
        if memory_store:
            self.memory = memory_store
        elif MEMORY_STORE_AVAILABLE:
            self.memory = get_memory_store(agent_id, "sqlite")
        else:
            self.memory = None
            print("Warning: memory_store module not available. Learning will not persist between runs.")
        
        # Initialize models dictionary
        self.models = {}
    
    def create_model(self, model_name: str, model_type: str, model_params: Optional[Dict[str, Any]] = None) -> SimpleModel:
        """
        Create a new model for the agent
        
        Args:
            model_name: Identifier for the model
            model_type: Type of model to create
            model_params: Parameters for the model
            
        Returns:
            Created model
        """
        try:
            model = SimpleModel(model_type, model_params)
            self.models[model_name] = model
            return model
        except Exception as e:
            print(f"Error creating model: {e}")
            return None
    
    def get_model(self, model_name: str) -> Optional[SimpleModel]:
        """Get a model by name"""
        return self.models.get(model_name)
    
    def train_model(self, model_name: str, X: List[List[float]], y: List[Any]) -> Optional[float]:
        """
        Train a model on new data
        
        Args:
            model_name: Name of the model to train
            X: Feature data
            y: Target values
            
        Returns:
            Performance metric or None if failed
        """
        model = self.get_model(model_name)
        if model:
            try:
                metric = model.fit(X, y)
                
                # Store training data in memory if available
                if self.memory:
                    self.memory.save({
                        'X': X,
                        'y': y,
                        'metric': metric,
                        'model_name': model_name,
                        'model_type': model.model_type
                    }, f"training_{model_name}_{int(time.time())}")
                
                return metric
            except Exception as e:
                print(f"Error training model: {e}")
                return None
        return None
    
    def make_prediction(self, model_name: str, X: Union[List[float], List[List[float]]]) -> Any:
        """
        Make a prediction using a model
        
        Args:
            model_name: Name of the model to use
            X: Feature data
            
        Returns:
            Prediction(s)
        """
        model = self.get_model(model_name)
        if model:
            try:
                prediction = model.predict(X)
                
                # Store prediction in memory if available
                if self.memory:
                    self.memory.save({
                        'X': X,
                        'prediction': prediction,
                        'model_name': model_name
                    }, f"prediction_{model_name}_{int(time.time())}")
                
                return prediction
            except Exception as e:
                print(f"Error making prediction: {e}")
                return None
        return None
    
    def save_models(self, directory: str) -> Dict[str, bool]:
        """
        Save all models to disk
        
        Args:
            directory: Directory to save models
            
        Returns:
            Dictionary of model names and success status
        """
        os.makedirs(directory, exist_ok=True)
        results = {}
        
        for model_name, model in self.models.items():
            try:
                path = os.path.join(directory, f"{model_name}.pkl")
                model.save(path)
                results[model_name] = True
            except Exception as e:
                print(f"Error saving model {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def load_models(self, directory: str) -> Dict[str, bool]:
        """
        Load models from disk
        
        Args:
            directory: Directory to load models from
            
        Returns:
            Dictionary of model names and success status
        """
        results = {}
        
        # Check if directory exists
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist")
            return {}
        
        # Load all model files
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                model_name = filename.replace('.pkl', '')
                try:
                    path = os.path.join(directory, filename)
                    model = SimpleModel.load(path)
                    self.models[model_name] = model
                    results[model_name] = True
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    results[model_name] = False
        
        return results
    
    def record_feedback(self, prediction_result: Any, actual_result: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Record feedback to improve future predictions
        
        Args:
            prediction_result: The prediction that was made
            actual_result: The actual outcome
            metadata: Additional information about the context
            
        Returns:
            Key of the saved feedback
        """
        if not self.memory:
            print("Warning: No memory store available. Feedback not recorded.")
            return None
        
        feedback = {
            'prediction': prediction_result,
            'actual': actual_result,
            'timestamp': datetime.datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        key = self.memory.save(feedback, f"feedback_{int(time.time())}")
        return key
    
    def get_feedback_data(self) -> List[Dict[str, Any]]:
        """
        Get all feedback data for learning
        
        Returns:
            List of feedback entries
        """
        if not self.memory:
            print("Warning: No memory store available.")
            return []
        
        # Get all keys that start with "feedback_"
        feedback_keys = [k for k in self.memory.list_keys() if k.startswith("feedback_")]
        
        # Load all feedback data
        feedback_data = []
        for key in feedback_keys:
            data = self.memory.load(key)
            if data:
                feedback_data.append(data)
        
        return feedback_data
    
    def learn_from_feedback(self, model_name: str, feature_extractor: Callable[[Dict[str, Any]], List[float]]) -> Optional[float]:
        """
        Learn from feedback by retraining the model
        
        Args:
            model_name: Name of the model to train
            feature_extractor: Function to extract features from feedback metadata
            
        Returns:
            Performance metric or None if failed
        """
        feedback_data = self.get_feedback_data()
        
        if not feedback_data:
            print("No feedback data available for learning")
            return None
        
        # Extract features and targets from feedback
        X = []
        y = []
        
        for entry in feedback_data:
            try:
                features = feature_extractor(entry)
                X.append(features)
                y.append(entry['actual'])
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
        
        if not X:
            print("Could not extract features from any feedback entries")
            return None
        
        # Train the model on the feedback data
        return self.train_model(model_name, X, y)

# Main example
if __name__ == "__main__":
    # Example of using the machine learning utilities
    import time
    import datetime
    
    # Create a learning agent
    agent = LearningAgent("example_agent")
    
    # Create a regression model
    agent.create_model("weather_predictor", "linear_regression")
    
    # Sample weather data (features: [temperature, humidity, pressure])
    X_train = [
        [25, 80, 1012],  # temperature, humidity, pressure
        [20, 65, 1010],
        [30, 70, 1008],
        [22, 90, 1015],
        [28, 75, 1013]
    ]
    y_train = [0, 0, 1, 1, 0]  # 0 = no rain, 1 = rain
    
    # Train the model
    performance = agent.train_model("weather_predictor", X_train, y_train)
    print(f"Model training performance: {performance}")
    
    # Make a prediction
    new_weather = [26, 82, 1011]
    prediction = agent.make_prediction("weather_predictor", new_weather)
    print(f"Weather prediction: {'Rain' if prediction > 0.5 else 'No rain'} (score: {prediction})")
    
    # Record feedback (simulate actual outcome)
    agent.record_feedback(
        prediction, 
        1,  # It actually rained
        {"temperature": new_weather[0], "humidity": new_weather[1], "pressure": new_weather[2]}
    )
    
    # Define a feature extractor function
    def weather_feature_extractor(feedback_entry):
        metadata = feedback_entry.get('metadata', {})
        return [
            metadata.get('temperature', 0),
            metadata.get('humidity', 0),
            metadata.get('pressure', 0)
        ]
    
    # Learn from feedback after some time
    time.sleep(1)  # Simulate passage of time
    performance = agent.learn_from_feedback("weather_predictor", weather_feature_extractor)
    print(f"Updated model performance after feedback: {performance}")
    
    # Make another prediction
    another_weather = [27, 85, 1010]
    new_prediction = agent.make_prediction("weather_predictor", another_weather)
    print(f"Updated weather prediction: {'Rain' if new_prediction > 0.5 else 'No rain'} (score: {new_prediction})")
    
    # Save the model
    agent.save_models("./models") 