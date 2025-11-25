import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import warnings

class ARIMA:
    """
    Direct Multi-Horizon Autoregressive Integrated (ARI) Model.
    
    Instead of recursive forecasting (which accumulates error), this model 
    trains a separate regression model for each forecast horizon to directly 
    predict y_{t+h} from y_{t}, y_{t-1}, ...
    
    Structure:
    1. Differencing (d) to make stationary
    2. Lag features (p) creation
    3. Linear Regression for each horizon h: y'_{t+h} = f_h(y'_t, y'_{t-1}, ...)
    4. Inverse differencing to get final forecast
    """
    def __init__(self, p=1, d=1, q=0, ridge_alpha=0.0):
        self.p = p
        self.d = d
        self.q = q  # MA not supported in direct linear implementation
        self.ridge_alpha = ridge_alpha
        self.models = {}  # Dictionary to store model for each horizon
        self.fitted_ = False
        
    def difference(self, series):
        """Apply differencing of order d."""
        z = np.asarray(series, dtype=float)
        history = []
        for _ in range(self.d):
            history.append(z[0])
            z = np.diff(z)
        return z, history
    
    def inverse_difference(self, z_diff, history):
        """Reverse differencing."""
        z = np.asarray(z_diff, dtype=float)
        for initial_val in reversed(history):
            z = np.cumsum(np.concatenate([[initial_val], z]))
        return z
    
    def create_lag_features(self, series):
        """Create lag features matrix X and target vector y for 1-step ahead."""
        X, y = [], []
        if len(series) <= self.p:
            return np.array([]), np.array([])
            
        for i in range(self.p, len(series)):
            X.append(series[i-self.p:i][::-1]) # Reverse so index 0 is lag 1
            y.append(series[i])
            
        return np.array(X), np.array(y)

    def prepare_horizon_data(self, series_list, horizon):
        """
        Prepare X (lags) and y (target at horizon h) for training.
        Target is y_{t+h} relative to the last lag in X.
        """
        X_all = []
        y_all = []
        
        for series in series_list:
            # 1. Difference the series
            z, _ = self.difference(series)
            
            if len(z) <= self.p + horizon - 1:
                continue
                
            # 2. Create samples
            # We want to predict z[i + horizon - 1] using z[i-p : i]
            # i ranges from p to len(z) - horizon
            
            for i in range(self.p, len(z) - horizon + 1):
                # Features: p lags ending at i-1
                lags = z[i-self.p : i][::-1] # Lags: z[i-1], z[i-2]...
                
                # Target: z at horizon steps ahead
                # If horizon=1, target is z[i]
                # If horizon=h, target is z[i + h - 1]
                target = z[i + horizon - 1]
                
                X_all.append(lags)
                y_all.append(target)
                
        return np.array(X_all), np.array(y_all)

    def fit(self, series_list, horizons=[1]):
        """
        Train separate models for each horizon.
        """
        print(f"Training Direct Multi-Horizon Models (p={self.p}, d={self.d})...")
        
        for h in horizons:
            X, y = self.prepare_horizon_data(series_list, h)
            
            if len(X) == 0:
                print(f"  Warning: Not enough data for horizon {h}")
                continue
                
            if self.ridge_alpha > 0:
                model = Ridge(alpha=self.ridge_alpha)
            else:
                model = LinearRegression()
                
            model.fit(X, y)
            self.models[h] = model
            # print(f"  Horizon {h}: Trained on {len(X)} samples. R2: {model.score(X, y):.4f}")
            
        self.fitted_ = True
        return self
    
    def forecast(self, history, steps=1):
        """
        Forecast up to 'steps' ahead using the specialized model for each step.
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before forecasting")
            
        history = np.asarray(history, dtype=float)
        
        # 1. Difference the history
        z, diff_history = self.difference(history)
        
        if len(z) < self.p:
            # Not enough history for lags, return last value (naive)
            return np.full(steps, history[-1])
            
        # 2. Extract current lags
        current_lags = z[-self.p:][::-1].reshape(1, -1)
        
        forecasts_diff = []
        
        # 3. Predict each horizon using its specific model
        for h in range(1, steps + 1):
            if h in self.models:
                pred_diff = self.models[h].predict(current_lags)[0]
            else:
                # Fallback to closest model if specific horizon missing
                available = sorted(self.models.keys())
                if not available:
                    pred_diff = 0
                else:
                    closest = min(available, key=lambda x: abs(x - h))
                    pred_diff = self.models[closest].predict(current_lags)[0]
            
            forecasts_diff.append(pred_diff)
            
        # 4. Integrate back to original scale
        # Note: We predicted differences relative to the *end of the history*
        # For h=1, pred is z_{t+1}
        # For h=2, pred is z_{t+2}
        # We need to reconstruct the path: y_{t+1} = y_t + z_{t+1}
        # y_{t+2} = y_{t+1} + z_{t+2} = y_t + z_{t+1} + z_{t+2}
        
        # Reconstruct cumulative sum of predicted differences
        # Start from the last observed value in the original scale (undifferenced)
        # But wait, 'difference' might be d > 1.
        
        # Simplest way: Append predicted differences to z, then inverse_difference whole thing
        z_extended = np.concatenate([z, forecasts_diff])
        history_extended = self.inverse_difference(z_extended, diff_history)
        
        # The forecasts are the last 'steps' elements
        return history_extended[-steps:]

    def summary(self):
        print(f"Direct Multi-Horizon ARI({self.p},{self.d}) Model")
        print("="*50)
        print(f"Trained horizons: {sorted(self.models.keys())}")
        print("="*50)
