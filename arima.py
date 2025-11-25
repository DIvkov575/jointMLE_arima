import numpy as np
from scipy.optimize import minimize
import warnings

class MultiSeriesARIMA:
    """
    Multi-series ARIMA(p,d,q) with shared parameters across multiple univariate time series.
    
    Fits a single set of ARIMA parameters to multiple series via Maximum Likelihood Estimation.
    Provides forecasts in the original (undifferenced) scale with proper inverse transformation.
    
    Parameters:
    -----------
    p : int
        Order of autoregressive component (number of AR lags)
    d : int
        Order of differencing (0 = no differencing, 1 = first difference, etc.)
    q : int
        Order of moving average component (number of MA lags)
    """
    def __init__(self, p=1, d=0, q=0):
        if p < 0 or d < 0 or q < 0:
            raise ValueError("Orders p, d, q must be non-negative integers")
        if p == 0 and q == 0:
            raise ValueError("At least one of p or q must be positive for ARIMA model")
            
        self.p = p
        self.d = d
        self.q = q
        self.params_ = None  # θ = [c, φ_1,...,φ_p, θ_1,...,θ_q, log(σ^2)]
        self.fitted_ = False
        self.series_history_ = None  # Store original series for inverse differencing
        
    def difference(self, series, return_history=False):
        """
        Apply differencing of order d to a single series.
        
        Parameters:
        -----------
        series : array-like
            Time series to difference
        return_history : bool
            If True, return differencing history needed for inverse transform
            
        Returns:
        --------
        z : ndarray
            Differenced series
        history : list of arrays (optional)
            Initial values from each differencing step
        """
        z = np.asarray(series, dtype=float)
        history = []
        
        for _ in range(self.d):
            if return_history:
                history.append(z[0])  # Store first value before differencing
            z = np.diff(z)
            
        if return_history:
            return z, history
        return z
    
    def inverse_difference(self, z_diff, history):
        """
        Reverse differencing to return to original scale.
        
        Parameters:
        -----------
        z_diff : array-like
            Differenced values to transform back
        history : list of arrays
            Initial values from each differencing step (from difference method)
            
        Returns:
        --------
        z_original : ndarray
            Values in original scale
        """
        z = np.asarray(z_diff, dtype=float)
        
        # Reverse differencing in opposite order
        for initial_val in reversed(history):
            z = np.cumsum(np.concatenate([[initial_val], z]))
            
        return z
    
    def compute_residuals(self, z, params):
        """
        Compute recursive residuals for a single differenced series.
        
        ARIMA equation: z[t] = c + Σφᵢz[t-i] + Σθⱼe[t-j] + e[t]
        Therefore: e[t] = z[t] - c - Σφᵢz[t-i] - Σθⱼe[t-j]
        
        Parameters:
        -----------
        z : array-like
            Differenced series
        params : array-like
            Parameters [c, φ_1,...,φ_p, θ_1,...,θ_q]
            
        Returns:
        --------
        e : ndarray
            Residuals
        """
        c = params[0]
        phi = params[1:1+self.p] if self.p > 0 else np.array([])
        theta = params[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
        
        T = len(z)
        e = np.zeros(T)
        
        for t in range(T):
            # AR component: use past observations
            ar_term = 0.0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_term += phi[i] * z[t - i - 1]
            
            # MA component: use past residuals
            ma_term = 0.0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_term += theta[j] * e[t - j - 1]
            
            # Compute residual
            e[t] = z[t] - c - ar_term - ma_term
            
        return e
    
    def check_stationarity(self, phi):
        """
        Check if AR parameters satisfy stationarity condition.
        For AR(p), roots of 1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0 must be outside unit circle.
        """
        if self.p == 0:
            return True
            
        # Create polynomial coefficients [1, -φ₁, -φ₂, ..., -φₚ]
        poly_coeffs = np.concatenate([[1], -phi])
        roots = np.roots(poly_coeffs)
        
        # Check if all roots are outside unit circle
        return np.all(np.abs(roots) > 1.0)
    
    def check_invertibility(self, theta):
        """
        Check if MA parameters satisfy invertibility condition.
        For MA(q), roots of 1 + θ₁z + θ₂z² + ... + θₑzᵍ = 0 must be outside unit circle.
        """
        if self.q == 0:
            return True
            
        # Create polynomial coefficients [1, θ₁, θ₂, ..., θₑ]
        poly_coeffs = np.concatenate([[1], theta])
        roots = np.roots(poly_coeffs)
        
        # Check if all roots are outside unit circle
        return np.all(np.abs(roots) > 1.0)
    
    def neg_log_likelihood(self, params, series_list):
        """
        Compute negative log-likelihood across multiple series.
        
        Parameters:
        -----------
        params : array-like
            [c, φ_1,...,φ_p, θ_1,...,θ_q, log(σ²)]
        series_list : list of arrays
            Multiple time series
            
        Returns:
        --------
        nll : float
            Negative log-likelihood (to minimize)
        """
        # Extract variance (use log parameterization for stability)
        log_sigma2 = params[-1]
        sigma2 = np.exp(log_sigma2)
        
        # Ensure positive variance
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return 1e10
        
        theta = params[:-1]
        total_ll = 0.0
        
        for series in series_list:
            # Difference the series
            z = self.difference(series)
            
            if len(z) < max(self.p, self.q):
                warnings.warn("Series too short after differencing for given p, q")
                return 1e10
            
            # Compute residuals
            e = self.compute_residuals(z, theta)
            
            # Log-likelihood (assuming Gaussian errors)
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (e**2) / sigma2)
            
            if not np.isfinite(ll):
                return 1e10
                
            total_ll += ll
        
        return -total_ll  # Return negative for minimization
    
    def fit(self, series_list, init_params=None, maxiter=1000):
        """
        Fit the model to multiple series using Maximum Likelihood Estimation.
        
        Parameters:
        -----------
        series_list : list of arrays
            Multiple time series (each can be different length)
        init_params : array-like, optional
            Initial parameter guess [c, φ_1,...,φ_p, θ_1,...,θ_q, log(σ²)]
        maxiter : int
            Maximum iterations for optimizer
            
        Returns:
        --------
        self : MultiSeriesARIMA
            Fitted model
        """
        if len(series_list) == 0:
            raise ValueError("series_list must contain at least one series")
        
        # Validate series
        for i, series in enumerate(series_list):
            series_array = np.asarray(series, dtype=float)
            if len(series_array) <= self.d + max(self.p, self.q):
                raise ValueError(f"Series {i} too short for ARIMA({self.p},{self.d},{self.q})")
        
        # Store original series for inverse differencing
        self.series_history_ = []
        for series in series_list:
            _, history = self.difference(series, return_history=True)
            self.series_history_.append(history)
        
        # Initialize parameters
        n_params = 1 + self.p + self.q + 1  # c + AR + MA + log(σ²)
        
        if init_params is None:
            # Smarter initialization
            init_params = np.zeros(n_params)
            
            # Initialize constant as mean of differenced series
            all_diff = np.concatenate([self.difference(s) for s in series_list])
            init_params[0] = np.mean(all_diff)
            
            # Initialize AR coefficients small positive
            init_params[1:1+self.p] = 0.1
            
            # Initialize MA coefficients small positive
            init_params[1+self.p:1+self.p+self.q] = 0.1
            
            # Initialize log(σ²) from sample variance
            init_params[-1] = np.log(np.var(all_diff) + 1e-6)
        
        # Optimize
        result = minimize(
            self.neg_log_likelihood,
            init_params,
            args=(series_list,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
        
        self.params_ = result.x
        self.fitted_ = True
        
        # Check stationarity and invertibility
        phi = self.params_[1:1+self.p] if self.p > 0 else np.array([])
        theta = self.params_[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
        
        if not self.check_stationarity(phi):
            warnings.warn("AR parameters suggest non-stationary process")
        
        if not self.check_invertibility(theta):
            warnings.warn("MA parameters suggest non-invertible process")
        
        return self
    
    def forecast(self, series, steps=1, return_differenced=False):
        """
        Forecast future values for a given series in ORIGINAL scale.
        
        Parameters:
        -----------
        series : array-like
            Historical time series (must be one of the series used in fit)
        steps : int
            Number of steps ahead to forecast
        return_differenced : bool
            If True, return forecasts in differenced space instead of original
            
        Returns:
        --------
        forecasts : ndarray
            Forecasted values (in original scale unless return_differenced=True)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before forecasting")
        
        if steps < 1:
            raise ValueError("steps must be at least 1")
        
        series = np.asarray(series, dtype=float)
        
        # Get differenced series and history
        z, history = self.difference(series, return_history=True)
        
        # Compute residuals for the known series
        c = self.params_[0]
        phi = self.params_[1:1+self.p] if self.p > 0 else np.array([])
        theta = self.params_[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([])
        
        e = self.compute_residuals(z, self.params_[:-1])
        
        # Prepare for forecasting
        z_list = list(z)
        e_list = list(e)
        forecasts_diff = []
        
        for step in range(steps):
            # AR component
            ar_term = 0.0
            for i in range(self.p):
                idx = len(z_list) - i - 1
                if idx >= 0:
                    ar_term += phi[i] * z_list[idx]
            
            # MA component (expected future errors = 0)
            ma_term = 0.0
            for j in range(self.q):
                idx = len(e_list) - j - 1
                if idx >= 0:
                    ma_term += theta[j] * e_list[idx]
            
            # Forecast
            z_next = c + ar_term + ma_term
            e_next = 0.0  # Expected error is zero
            
            z_list.append(z_next)
            e_list.append(e_next)
            forecasts_diff.append(z_next)
        
        forecasts_diff = np.array(forecasts_diff)
        
        if return_differenced:
            return forecasts_diff
        
        # Inverse difference to get original scale
        # Combine last actual values with forecasts
        combined = np.concatenate([z, forecasts_diff])
        combined_original = self.inverse_difference(combined, history)
        
        # Return only the forecasted portion
        forecasts_original = combined_original[len(series):]
        
        return forecasts_original
    
    def get_params(self):
        """
        Get fitted parameters as a dictionary.
        
        Returns:
        --------
        params : dict
            Dictionary with 'constant', 'ar_coefs', 'ma_coefs', 'sigma2'
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
        
        return {
            'constant': self.params_[0],
            'ar_coefs': self.params_[1:1+self.p] if self.p > 0 else np.array([]),
            'ma_coefs': self.params_[1+self.p:1+self.p+self.q] if self.q > 0 else np.array([]),
            'sigma2': np.exp(self.params_[-1])
        }
    
    def summary(self):
        """
        Print a summary of the fitted model.
        """
        if not self.fitted_:
            print("Model not fitted yet.")
            return
        
        params = self.get_params()
        
        print(f"Multi-Series ARIMA({self.p},{self.d},{self.q}) Model")
        print("=" * 50)
        print(f"Constant: {params['constant']:.6f}")
        
        if self.p > 0:
            print(f"\nAR Coefficients:")
            for i, coef in enumerate(params['ar_coefs'], 1):
                print(f"  φ_{i}: {coef:.6f}")
            stationary = self.check_stationarity(params['ar_coefs'])
            print(f"  Stationary: {stationary}")
        
        if self.q > 0:
            print(f"\nMA Coefficients:")
            for i, coef in enumerate(params['ma_coefs'], 1):
                print(f"  θ_{i}: {coef:.6f}")
            invertible = self.check_invertibility(params['ma_coefs'])
            print(f"  Invertible: {invertible}")
        
        print(f"\nError Variance (σ²): {params['sigma2']:.6f}")
        print("=" * 50)