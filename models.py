"""
Path Loss Propagation Models
Implements empirical and optimized models for radio propagation prediction
"""

import numpy as np
from typing import Union, Tuple, Callable
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for path loss model"""
    f_mhz: Union[np.ndarray, float]
    h_b_m: Union[np.ndarray, float]
    h_m_m: Union[np.ndarray, float]
    d_km: Union[np.ndarray, float]


class COST231HataUrban:
    """
    COST-231 Hata Model for Urban Environments
    
    Valid Range:
        - Frequency: 1500-2000 MHz
        - Base station height: 30-200 m
        - Mobile height: 1-10 m
        - Distance: 1-20 km
    """
    
    def __init__(self):
        self.name = "COST-231 Hata Urban"
        self.params = {'A': 46.3, 'B': 33.9, 'C': 13.82, 'D': 1.1, 'E': 0.7, 'F': 1.56}
    
    def predict(self, f_mhz: np.ndarray, h_b_m: np.ndarray, 
                h_m_m: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """
        Calculate path loss using COST-231 Hata Urban model
        
        Args:
            f_mhz: Frequency in MHz
            h_b_m: Base station height in meters
            h_m_m: Mobile station height in meters
            d_km: Distance in kilometers
            
        Returns:
            Path loss in dB
        """
        # Safety: prevent log(0)
        d_km_safe = np.asarray(d_km) + 1e-9
        h_b_m_safe = np.asarray(h_b_m) + 1e-9
        f_mhz_arr = np.asarray(f_mhz)
        h_m_m_arr = np.asarray(h_m_m)
        
        # Mobile station correction factor
        a_hm_f = (1.1 * np.log10(f_mhz_arr) - 0.7) * h_m_m_arr - \
                 (1.56 * np.log10(f_mhz_arr) - 0.8)
        
        # Urban path loss
        L_urban = (46.3 + 33.9 * np.log10(f_mhz_arr) - 
                   13.82 * np.log10(h_b_m_safe) - a_hm_f + 
                   (44.9 - 6.55 * np.log10(h_b_m_safe)) * np.log10(d_km_safe))
        
        return L_urban


class COST231HataSuburban(COST231HataUrban):
    """
    COST-231 Hata Model for Suburban Environments
    Urban model with -5 dB suburban correction
    """
    
    def __init__(self):
        super().__init__()
        self.name = "COST-231 Hata Suburban"
    
    def predict(self, f_mhz: np.ndarray, h_b_m: np.ndarray, 
                h_m_m: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """Suburban model: Urban - 5 dB correction"""
        urban_loss = super().predict(f_mhz, h_b_m, h_m_m, d_km)
        return urban_loss - 5.0


class OkumuraHataSuburban:
    """
    Okumura-Hata Model for Suburban Areas
    
    Formula: L = L_urban + Cs
    Where Cs = -2*(log10(f/28))^2 - 5.4
    
    Valid Range:
        - Frequency: 150-1500 MHz (model), 1800 MHz extrapolated
        - Distance: 1-20 km
    """
    
    def __init__(self):
        self.name = "Okumura-Hata Suburban"
    
    def predict(self, f_mhz: np.ndarray, h_b_m: np.ndarray, 
                h_m_m: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """Calculate path loss using Okumura-Hata Suburban model"""
        
        d_km_safe = np.asarray(d_km) + 1e-9
        h_b_m_safe = np.asarray(h_b_m) + 1e-9
        f_mhz_arr = np.asarray(f_mhz)
        h_m_m_arr = np.asarray(h_m_m)
        
        # Mobile station correction factor
        a_hm_f = (1.1 * np.log10(f_mhz_arr) - 0.7) * h_m_m_arr - \
                 (1.56 * np.log10(f_mhz_arr) - 0.8)
        
        # Urban base path loss
        L_urban = (69.55 + 26.16 * np.log10(f_mhz_arr) - 
                   13.82 * np.log10(h_b_m_safe) - a_hm_f + 
                   (44.9 - 6.55 * np.log10(h_b_m_safe)) * np.log10(d_km_safe))
        
        # Suburban correction factor
        Cs = -2 * (np.log10(f_mhz_arr / 28.0))**2 - 5.4
        
        return L_urban + Cs


class EgliModel:
    """
    Egli Model for Point-to-Point Rural/Suburban Communication
    
    Formula: L = 117 + 40*log10(d) + 20*log10(f) - 20*log10(hb) - 20*log10(hm)
    """
    
    def __init__(self):
        self.name = "Egli Model"
    
    def predict(self, f_mhz: np.ndarray, h_b_m: np.ndarray, 
                h_m_m: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """Calculate path loss using Egli model"""
        
        d_km_safe = np.asarray(d_km) + 1e-9
        h_b_m_safe = np.asarray(h_b_m) + 1e-9
        h_m_m_safe = np.asarray(h_m_m) + 1e-9
        f_mhz_arr = np.asarray(f_mhz)
        
        L_egli = (117 + 40 * np.log10(d_km_safe) + 
                  20 * np.log10(f_mhz_arr) - 
                  20 * np.log10(h_b_m_safe) - 
                  20 * np.log10(h_m_m_safe))
        
        return L_egli


class OptimizedCOST231(COST231HataUrban):
    """
    Optimized COST-231 Hata Model using Metaheuristic Algorithms
    
    Supports 9 optimizable parameters instead of fixed coefficients
    params = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    """
    
    def __init__(self, optimized_params: np.ndarray = None):
        super().__init__()
        self.optimized_params = optimized_params
        self.name = "COST-231 Optimized"
    
    def predict(self, f_mhz: np.ndarray, h_b_m: np.ndarray, 
                h_m_m: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """Calculate path loss using optimized parameters"""
        
        if self.optimized_params is None:
            raise ValueError("Optimized parameters not set")
        
        d_km_safe = np.asarray(d_km) + 1e-9
        h_b_m_safe = np.asarray(h_b_m) + 1e-9
        f_mhz_arr = np.asarray(f_mhz)
        h_m_m_arr = np.asarray(h_m_m)
        
        # Unpack parameters
        c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.optimized_params
        
        # Calculate mobile height correction
        a_hm_f = (c4 * np.log10(f_mhz_arr) - c5) * h_m_m_arr - \
                 (c6 * np.log10(f_mhz_arr) - c7)
        
        # Optimized path loss formula
        L_optimized = (c1 + c2 * np.log10(f_mhz_arr) - 
                       c3 * np.log10(h_b_m_safe) - a_hm_f + 
                       (c8 - c9 * np.log10(h_b_m_safe)) * np.log10(d_km_safe))
        
        return L_optimized


class ANNModel:
    """
    Artificial Neural Network model wrapper for path loss prediction
    Uses TensorFlow/Keras
    """
    
    def __init__(self, model=None):
        self.model = model
        self.name = "Neural Network (ANN)"
        self.scaler = None
    
    def set_scaler(self, scaler):
        """Set feature scaler for predictions"""
        self.scaler = scaler
    
    def predict(self, f_mhz: np.ndarray, h_b_m: np.ndarray, 
                h_m_m: np.ndarray, d_km: np.ndarray) -> np.ndarray:
        """
        Make predictions using ANN
        
        Args:
            f_mhz, h_b_m, h_m_m, d_km: Input features
            
        Returns:
            Path loss predictions in dB
        """
        if self.model is None:
            raise ValueError("ANN model not initialized")
        
        # Stack features
        features = np.column_stack([
            np.asarray(f_mhz),
            np.asarray(h_b_m),
            np.asarray(h_m_m),
            np.asarray(d_km)
        ])
        
        # Scale if scaler available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features, verbose=0).flatten()
        return predictions


class ModelFactory:
    """Factory for creating path loss models"""
    
    @staticmethod
    def create_model(model_name: str, **kwargs) -> Callable:
        """
        Create a model instance by name
        
        Args:
            model_name: Name of the model ('cost231_urban', 'cost231_suburban', 
                       'okumura_suburban', 'egli', 'optimized_cost231', 'ann')
            **kwargs: Model-specific arguments
            
        Returns:
            Model instance with predict method
        """
        models = {
            'cost231_urban': COST231HataUrban(),
            'cost231_suburban': COST231HataSuburban(),
            'okumura_suburban': OkumuraHataSuburban(),
            'egli': EgliModel(),
            'optimized_cost231': OptimizedCOST231(kwargs.get('params')),
            'ann': ANNModel(kwargs.get('model'))
        }
        
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
        
        return models[model_name]
    
    @staticmethod
    def get_all_models() -> dict:
        """Get all available models"""
        return {
            'cost231_urban': COST231HataUrban(),
            'cost231_suburban': COST231HataSuburban(),
            'okumura_suburban': OkumuraHataSuburban(),
            'egli': EgliModel(),
        }
