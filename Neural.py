import numpy as np
class Neural:
    def __init__(self, lr=0.01, epochs=1000):
        """
        Constructor (initializer) of the class.

        This method is automatically called when you create an object of the class.

        Parameters:
        lr (float)      : Learning rate — controls how big each update step is.
                          If it's too large → model may not converge.
                          If it's too small → training will be very slow.

        epochs (int)    : Number of iterations (how many times the model will
                          go through the entire dataset during training).
        """

        self.lr = lr
        self.epochs = epochs

        # Model parameters (initialized later in fit method)
        self.w = None  # weight (controls slope of the line)
        self.b = None  # bias (controls vertical shift)


    def fit(self, X, y):
        """
        Train the model using Gradient Descent.

        The goal is to find optimal values of w and b so that:
        predicted values are as close as possible to real values.

        Parameters:
        X (numpy array) : Input feature(s) (independent variable)
        y (numpy array) : Target values (dependent variable)
        """

        # Number of samples (data points)
        n_samples = X.shape[0]

        # Initialize parameters with zeros
        # We start from a simple guess and improve it step by step
        self.w = 0.0
        self.b = 0.0

        # Training loop (repeats many times)
        for i in range(self.epochs):

            # -------------------------------
            # 1. FORWARD PASS (Prediction)
            # -------------------------------
            # Apply the linear equation:
            # y_pred = w * X + b
            # This is our model's current prediction
            y_pred = self.w * X + self.b

            # -------------------------------
            # 2. LOSS (Error calculation)
            # -------------------------------
            # We use Mean Squared Error (MSE):
            # MSE = average((y - y_pred)^2)
            # This measures how wrong the model is

            # (we calculate it later for printing)

            # -------------------------------
            # 3. GRADIENTS (How to improve)
            # -------------------------------
            # We compute derivatives of the loss function
            # These tell us how to adjust w and b

            # Derivative with respect to weight (w)
            dw = (2 / n_samples) * np.sum((y_pred - y) * X)

            # Derivative with respect to bias (b)
            db = (2 / n_samples) * np.sum(y_pred - y)

            # -------------------------------
            # 4. UPDATE PARAMETERS
            # -------------------------------
            # Move parameters in the opposite direction of the gradient
            # (this reduces the error step by step)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            # -------------------------------
            # 5. LOGGING (Print progress)
            # -------------------------------
            # Every 100 iterations we print the current loss
            if i % 100 == 0:
                loss = np.mean((y - y_pred) ** 2)
                print(f"Epoch {i}, Loss: {loss:.4f}")

        # After training, w and b should represent the best-fit line


    def predict(self, X):
        """
        Make predictions using the trained model.

        This method is used AFTER training.

        Parameters:
        X (numpy array) : Input data

        Returns:
        numpy array     : Predicted values based on learned parameters
        """

        # Apply the learned linear equation
        return self.w * X + self.b
    


        
      
    
    
