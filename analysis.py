"""
Statistical Analysis and Evaluation Functions
Includes cross-validation, significance testing, and diagnostics
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive error metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        residuals = y_true - y_pred
        
        return {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'Mean_Residual': np.mean(residuals),
            'Std_Residual': np.std(residuals),
            'Min_Error': np.min(np.abs(residuals)),
            'Max_Error': np.max(np.abs(residuals))
        }
    
    @staticmethod
    def cross_validate_model(model, X: np.ndarray, y: np.ndarray, 
                            cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform K-fold cross-validation
        
        Args:
            model: Model with fit and predict methods
            X: Feature matrix
            y: Target values
            cv_folds: Number of folds
            
        Returns:
            Dictionary with cross-validation metrics
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            rmse_scores.append(rmse)
            mae_scores.append(mae)
            r2_scores.append(r2)
        
        return {
            'RMSE_mean': np.mean(rmse_scores),
            'RMSE_std': np.std(rmse_scores),
            'MAE_mean': np.mean(mae_scores),
            'MAE_std': np.std(mae_scores),
            'R2_mean': np.mean(r2_scores),
            'R2_std': np.std(r2_scores),
            'fold_rmses': rmse_scores
        }
    
    @staticmethod
    def statistical_significance_test(scores1: np.ndarray, scores2: np.ndarray,
                                      alpha: float = 0.05) -> Dict:
        """
        Perform t-test for statistical significance
        
        Args:
            scores1: Error scores from model 1
            scores2: Error scores from model 2
            alpha: Significance level
            
        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_ind(scores1, scores2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_diff': np.mean(scores1) - np.mean(scores2),
            'ci_lower': np.mean(scores1) - 1.96 * np.std(scores1) / np.sqrt(len(scores1)),
            'ci_upper': np.mean(scores1) + 1.96 * np.std(scores1) / np.sqrt(len(scores1))
        }
    
    @staticmethod
    def residual_diagnostics(residuals: np.ndarray) -> Dict:
        """
        Perform residual diagnostics
        
        Args:
            residuals: Model residuals
            
        Returns:
            Dictionary with diagnostic results
        """
        # Normality tests
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(np.mean(residuals), np.std(residuals)))
        
        # Autocorrelation (Durbin-Watson)
        try:
            from statsmodels.stats.stattools import durbin_watson
            dw_stat = durbin_watson(residuals)
        except ImportError:
            dw_stat = None
        
        return {
            'Shapiro_Wilk_stat': shapiro_stat,
            'Shapiro_Wilk_p': shapiro_p,
            'KS_stat': ks_stat,
            'KS_p': ks_p,
            'Durbin_Watson': dw_stat,
            'Mean': np.mean(residuals),
            'Std': np.std(residuals),
            'Skewness': stats.skew(residuals),
            'Kurtosis': stats.kurtosis(residuals)
        }


class SensitivityAnalyzer:
    """Parameter sensitivity analysis"""
    
    @staticmethod
    def parameter_sensitivity(model, params: np.ndarray, 
                             evaluate_func: callable,
                             perturbation: float = 0.10) -> Dict[str, float]:
        """
        Analyze sensitivity of model to parameter changes
        
        Args:
            model: Model to evaluate
            params: Current parameters
            evaluate_func: Function that evaluates model (returns error)
            perturbation: Percentage perturbation (default 10%)
            
        Returns:
            Dictionary of parameter sensitivities
        """
        baseline_error = evaluate_func(model, params)
        sensitivities = {}
        
        for i, param in enumerate(params):
            # Perturb parameter upward
            perturbed_params = params.copy()
            perturbed_params[i] = param * (1 + perturbation)
            
            # Evaluate perturbed model
            perturbed_error = evaluate_func(model, perturbed_params)
            
            # Calculate sensitivity (% change in error per % change in parameter)
            sensitivity = ((perturbed_error - baseline_error) / baseline_error) / perturbation * 100
            sensitivities[f'param_{i}'] = sensitivity
        
        return sensitivities


class ComparisonReport:
    """Generate comprehensive comparison report"""
    
    @staticmethod
    def generate_comparison_table(models_dict: Dict, y_true: np.ndarray) -> pd.DataFrame:
        """
        Generate comparison table for multiple models
        
        Args:
            models_dict: Dictionary of {model_name: (y_pred, model_object)}
            y_true: True values
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_name, (y_pred, _) in models_dict.items():
            metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
            metrics['Model'] = model_name
            results.append(metrics)
        
        df = pd.DataFrame(results)
        # Reorder columns
        cols = ['Model'] + [c for c in df.columns if c != 'Model']
        return df[cols].sort_values('RMSE')
    
    @staticmethod
    def rank_models(metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank models by multiple criteria
        
        Args:
            metrics_df: Metrics DataFrame
            
        Returns:
            DataFrame with rankings
        """
        ranking_df = metrics_df.copy()
        
        # Lower is better for RMSE, MAE, MAPE
        for col in ['RMSE', 'MAE', 'MAPE']:
            if col in ranking_df.columns:
                ranking_df[f'{col}_Rank'] = ranking_df[col].rank()
        
        # Higher is better for R2
        if 'R2' in ranking_df.columns:
            ranking_df['R2_Rank'] = ranking_df['R2'].rank(ascending=False)
        
        # Overall rank (average of all ranks)
        rank_cols = [c for c in ranking_df.columns if 'Rank' in c]
        if rank_cols:
            ranking_df['Overall_Rank'] = ranking_df[rank_cols].mean(axis=1)
        
        return ranking_df.sort_values('Overall_Rank')


class DistanceStratifiedAnalysis:
    """Analyze model performance at different distances"""
    
    @staticmethod
    def stratified_metrics(distances: np.ndarray, y_true: np.ndarray, 
                          y_pred: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
        """
        Calculate metrics stratified by distance
        
        Args:
            distances: Distance values
            y_true: True path loss
            y_pred: Predicted path loss
            n_bins: Number of distance bins
            
        Returns:
            DataFrame with stratified metrics
        """
        # Create distance bins
        bins = pd.qcut(distances, q=n_bins, duplicates='drop')
        
        results = []
        for bin_label in bins.unique():
            mask = bins == bin_label
            
            if mask.sum() > 0:
                metrics = ModelEvaluator.calculate_metrics(y_true[mask], y_pred[mask])
                metrics['Distance_Range'] = f"{bin_label.left:.1f}-{bin_label.right:.1f} km"
                metrics['Sample_Count'] = mask.sum()
                results.append(metrics)
        
        return pd.DataFrame(results)


def create_summary_report(results_dict: Dict) -> str:
    """
    Create text summary report
    
    Args:
        results_dict: Dictionary of results
        
    Returns:
        Formatted report string
    """
    report = "=" * 70 + "\n"
    report += "PATH LOSS MODEL OPTIMIZATION REPORT\n"
    report += "=" * 70 + "\n\n"
    
    for key, value in results_dict.items():
        report += f"{key}:\n"
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, float):
                    report += f"  {k}: {v:.4f}\n"
                else:
                    report += f"  {k}: {v}\n"
        else:
            report += f"  {value}\n"
        report += "\n"
    
    return report
