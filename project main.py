"""
Mini AI Model Trainer Framework
Assignment Implementation demonstrating OOP concepts used in ML frameworks
"""

from abc import ABC, abstractmethod
import numpy as np


# ============================================================
# OOP Concepts in this file:
# - Class Attribute: model_count in BaseModel
# - Instance Attribute: various attributes in each class
# - Abstraction: BaseModel abstract class
# - Single Inheritance: LinearRegressionModel, NeuralNetworkModel inherit from BaseModel
# - Method Overriding: train() and evaluate() methods in subclasses
# - super(): Used in constructors of subclasses
# - Polymorphism: Trainer works with any BaseModel subclass
# - Composition: Trainer contains Model (has-a relationship)
# - Aggregation: DataLoader stores data (has-a relationship)
# - Magic Method: __repr__ in ModelConfig
# - Instance Method: all methods that operate on instance data
# ============================================================


# ============================================================
# 1. ModelConfig Class
# Purpose: Store model configuration
# ============================================================
class ModelConfig:
    """ModelConfig - Stores model configuration settings"""

    # Instance Attributes: model_name, learning_rate, epochs

    def __init__(self, model_name: str, learning_rate: float = 0.01, epochs: int = 10):
        # Magic Method: __init__ is a magic/constructor method
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs

    def __repr__(self) -> str:
        # Magic Method: __repr__ for string representation
        return f"[Config] {self.model_name} | lr={self.learning_rate} | epochs={self.epochs}"

    def __str__(self) -> str:
        return self.__repr__()


# ============================================================
# 2. BaseModel Abstract Class
# Purpose: Abstract base class for all models
# ============================================================
class BaseModel(ABC):
    """BaseModel - Abstract base class for all ML models"""

    # Class Attribute: shared across all instances
    model_count: int = 0

    def __init__(self, config: ModelConfig):
        # Abstraction: Abstract class defines interface without implementation
        # Instance Attribute: stores the config
        self.config = config
        self.is_trained = False
        # Increment class attribute
        BaseModel.model_count += 1

    @abstractmethod
    def train(self, data) -> None:
        """Abstract method that subclasses must implement for training"""
        # Abstraction: Abstract method that subclasses must implement
        pass

    @abstractmethod
    def evaluate(self, data) -> dict:
        """Abstract method that subclasses must implement for evaluation"""
        # Abstraction: Abstract method that subclasses must implement
        pass


# ============================================================
# 3. LinearRegressionModel Class
# Purpose: Linear regression implementation
# ============================================================
class LinearRegressionModel(BaseModel):
    """LinearRegressionModel - Linear regression implementation"""

    # Inheritance: Single Inheritance from BaseModel

    def __init__(self, learning_rate: float = 0.01, epochs: int = 10):
        # Inheritance: Using super() to call parent constructor
        # Composition: Creates and contains ModelConfig instance
        config = ModelConfig("LinearRegression", learning_rate, epochs)
        super().__init__(config)  # super() - calling parent constructor

    def train(self, data) -> None:
        # Method Overriding: Implementing abstract method from parent
        # Instance Method: operates on instance data
        data_count = len(data)
        print(f"LinearRegression: Training on {data_count} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    def evaluate(self, data) -> dict:
        # Method Overriding: Implementing abstract method from parent
        # Instance Method: operates on instance data
        print(f"LinearRegression: Evaluation MSE = 0.042")
        return {"mse": 0.042}


# ============================================================
# 4. NeuralNetworkModel Class
# Purpose: Neural network implementation
# ============================================================
class NeuralNetworkModel(BaseModel):
    """NeuralNetworkModel - Simple neural network implementation"""

    # Inheritance: Single Inheritance from BaseModel

    def __init__(self, layers: list, learning_rate: float = 0.01, epochs: int = 10):
        # Inheritance: Using super() to call parent constructor
        # Composition: Creates and contains ModelConfig instance
        config = ModelConfig("NeuralNetwork", learning_rate, epochs)
        super().__init__(config)  # super() - calling parent constructor

        # Instance Attribute: stores layers
        self.layers = layers

    def train(self, data) -> None:
        # Method Overriding: Implementing abstract method from parent
        # Instance Method: operates on instance data
        data_count = len(data)
        print(f"NeuralNetwork {self.layers}: Training on {data_count} samples for {self.config.epochs} epochs (lr={self.config.learning_rate})")

    def evaluate(self, data) -> dict:
        # Method Overriding: Implementing abstract method from parent
        # Instance Method: operates on instance data
        print(f"NeuralNetwork: Evaluation Accuracy = 91.5%")
        return {"accuracy": 91.5}


# ============================================================
# 5. DataLoader Class
# Purpose: Store and provide dataset
# ============================================================
class DataLoader:
    """DataLoader - Dataset container"""

    # Aggregation: Contains data (has-a relationship, data can exist independently)

    def __init__(self, data: list):
        # Instance Attribute: stores the dataset
        self.data = np.array(data, dtype=np.float64)

    def get_data(self) -> np.ndarray:
        # Instance Method: returns stored dataset
        return self.data


# ============================================================
# 6. Trainer Class
# Purpose: Orchestrates training pipeline
# ============================================================
class Trainer:
    """Trainer - Orchestrates the model training pipeline"""

    # Composition: Contains Model (has-a relationship)
    # Polymorphism: Works with any subclass of BaseModel

    def __init__(self, model: BaseModel, dataloader: DataLoader):
        # Composition: Model is composed into Trainer
        # Instance Attributes: model and dataloader
        self.model = model
        self.dataloader = dataloader

    def run(self) -> None:
        # Polymorphism: Calls model.train() and model.evaluate()
        # which behave differently based on the actual model type
        # Instance Method: executes training pipeline
        data = self.dataloader.get_data()

        # Train the model
        self.model.train(data)

        # Evaluate the model
        self.model.evaluate(data)


# ============================================================
# Main Program
# ============================================================
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # 1. Create dataset
    data = [1, 2, 3, 4, 5]

    # 2. Create DataLoader
    # Aggregation: DataLoader aggregates the data
    loader = DataLoader(data)

    # 3. Create models
    model1 = LinearRegressionModel()
    model2 = NeuralNetworkModel(layers=[64, 32, 1], learning_rate=0.001, epochs=20)

    # 4. Print configurations
    print(model1.config)
    print(model2.config)

    # 5. Print total models created
    # Class Attribute access: model_count
    print(f"\nModels created: {BaseModel.model_count}")

    # 6. Run Trainer for both models
    # Polymorphism: Trainer works with any BaseModel subclass
    print("\n--- Training LinearRegression ---")
    Trainer(model1, loader).run()

    print("\n--- Training NeuralNetwork ---")
    Trainer(model2, loader).run()
