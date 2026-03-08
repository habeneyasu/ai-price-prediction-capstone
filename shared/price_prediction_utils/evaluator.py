"""
Evaluation framework for price prediction models.

This module provides comprehensive evaluation capabilities including metrics,
visualizations, and detailed error analysis.
"""

import re
import logging
import math
from typing import List, Callable, Optional, Dict, Any
from itertools import accumulate
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from shared.price_prediction_utils.item import Item

logger = logging.getLogger(__name__)

# ANSI color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"green": GREEN, "orange": YELLOW, "red": RED}


class PriceEvaluator:
    """
    Comprehensive evaluator for price prediction models.
    
    Provides metrics, visualizations, and detailed analysis of model performance.
    """
    
    def __init__(
        self,
        predictor: Callable[[Item], float],
        test_data: List[Item],
        name: Optional[str] = None,
        max_samples: Optional[int] = None,
        num_workers: int = 5,
    ):
        """
        Initialize the evaluator.
        
        Args:
            predictor: Function that takes an Item and returns predicted price
            test_data: List of test items with ground truth prices
            name: Name for this evaluation run
            max_samples: Maximum number of samples to evaluate (None = all)
            num_workers: Number of parallel workers for evaluation
        """
        self.predictor = predictor
        self.test_data = test_data[:max_samples] if max_samples else test_data
        self.name = name or self._get_predictor_name(predictor)
        self.num_workers = num_workers
        
        self.predictions: List[float] = []
        self.actuals: List[float] = []
        self.errors: List[float] = []
        self.items: List[Item] = []
        self.error_colors: List[str] = []
    
    @staticmethod
    def _get_predictor_name(predictor: Callable) -> str:
        """Extract a readable name from the predictor function."""
        if hasattr(predictor, '__name__'):
            name = predictor.__name__
            return name.replace("_", " ").title()
        elif hasattr(predictor, '__class__'):
            return predictor.__class__.__name__
        else:
            return "Unknown Predictor"
    
    @staticmethod
    def _extract_price(value: Any) -> float:
        """
        Extract numeric price from various input formats.
        
        Handles strings with currency symbols, commas, etc.
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove currency symbols and commas
            text = value.replace("$", "").replace(",", "").strip()
            # Find first number
            match = re.search(r"[-+]?\d*\.?\d+", text)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass
        
        logger.warning(f"Could not extract price from: {value}")
        return 0.0
    
    def _get_error_color(self, error: float, actual: float) -> str:
        """Determine color category for error visualization."""
        if error < 40 or (actual > 0 and error / actual < 0.2):
            return "green"
        elif error < 80 or (actual > 0 and error / actual < 0.4):
            return "orange"
        else:
            return "red"
    
    def _evaluate_single_item(self, item: Item) -> tuple:
        """Evaluate a single item and return results."""
        try:
            predicted = self.predictor(item)
            predicted = self._extract_price(predicted)
            actual = item.price
            error = abs(predicted - actual)
            color = self._get_error_color(error, actual)
            
            return item, predicted, actual, error, color
        except Exception as e:
            logger.error(f"Error evaluating item {item.item_id}: {e}")
            return item, 0.0, item.price, item.price, "red"
    
    def evaluate(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run evaluation on test data.
        
        Args:
            verbose: Whether to print progress and results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if verbose:
            logger.info(f"Evaluating {self.name} on {len(self.test_data)} items...")
        
        # Parallel evaluation
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(self._evaluate_single_item, self.test_data),
                    total=len(self.test_data),
                    desc=f"Evaluating {self.name}",
                    disable=not verbose,
                )
            )
        
        # Collect results
        for item, predicted, actual, error, color in results:
            self.items.append(item)
            self.predictions.append(predicted)
            self.actuals.append(actual)
            self.errors.append(error)
            self.error_colors.append(color)
            
            if verbose:
                print(f"{COLOR_MAP[color]}${error:.0f}{RESET} ", end="", flush=True)
        
        if verbose:
            print()  # New line after progress
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(self.actuals, self.predictions)
        mse = mean_squared_error(self.actuals, self.predictions)
        rmse = math.sqrt(mse)
        r2 = r2_score(self.actuals, self.predictions)
        
        # Additional metrics
        median_error = np.median(self.errors)
        max_error = max(self.errors) if self.errors else 0
        min_error = min(self.errors) if self.errors else 0
        
        # Percentage errors
        percentage_errors = [
            abs(p - a) / a * 100 if a > 0 else 0
            for p, a in zip(self.predictions, self.actuals)
        ]
        mape = np.mean(percentage_errors) if percentage_errors else 0
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "r2_percentage": r2 * 100,
            "median_error": median_error,
            "max_error": max_error,
            "min_error": min_error,
            "mape": mape,
            "num_samples": len(self.test_data),
        }
    
    def _print_metrics(self, metrics: Dict[str, Any]):
        """Print evaluation metrics to console."""
        print(f"\n{'='*60}")
        print(f"Evaluation Results: {self.name}")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE):     ${metrics['mae']:,.2f}")
        print(f"Root Mean Squared Error (RMSE): ${metrics['rmse']:,.2f}")
        print(f"R² Score:                      {metrics['r2_percentage']:.2f}%")
        print(f"Median Error:                  ${metrics['median_error']:,.2f}")
        print(f"Mean Absolute % Error (MAPE):   {metrics['mape']:.2f}%")
        print(f"Max Error:                     ${metrics['max_error']:,.2f}")
        print(f"Min Error:                     ${metrics['min_error']:,.2f}")
        print(f"Number of Samples:             {metrics['num_samples']}")
        print(f"{'='*60}\n")
    
    def plot_scatter(self, show: bool = True) -> go.Figure:
        """
        Create scatter plot of predicted vs actual prices.
        
        Args:
            show: Whether to display the plot immediately
            
        Returns:
            Plotly figure object
        """
        df = pd.DataFrame({
            "actual": self.actuals,
            "predicted": self.predictions,
            "error": self.errors,
            "color": self.error_colors,
            "title": [item.title[:50] + "..." if len(item.title) > 50 else item.title
                      for item in self.items],
        })
        
        # Create hover text
        df["hover"] = [
            f"{title}<br>Predicted: ${pred:,.2f}<br>Actual: ${act:,.2f}<br>Error: ${err:,.2f}"
            for title, pred, act, err in zip(df["title"], df["predicted"], df["actual"], df["error"])
        ]
        
        max_val = float(max(df["actual"].max(), df["predicted"].max()))
        
        fig = px.scatter(
            df,
            x="actual",
            y="predicted",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=f"{self.name} - Predicted vs Actual Prices",
            labels={"actual": "Actual Price ($)", "predicted": "Predicted Price ($)"},
            width=1000,
            height=800,
        )
        
        # Add hover information
        for trace in fig.data:
            mask = df["color"] == trace.name
            trace.customdata = df.loc[mask, ["hover"]].to_numpy()
            trace.hovertemplate = "%{customdata[0]}<extra></extra>"
            trace.marker.update(size=6)
        
        # Add reference line (y = x)
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="Perfect Prediction",
                hoverinfo="skip",
                showlegend=False,
            )
        )
        
        fig.update_xaxes(range=[0, max_val], title="Actual Price ($)")
        fig.update_yaxes(range=[0, max_val], title="Predicted Price ($)")
        fig.update_layout(
            showlegend=False,
            template="plotly_white",
        )
        
        if show:
            fig.show()
        
        return fig
    
    def plot_error_trend(self, show: bool = True) -> go.Figure:
        """
        Create error trend chart showing cumulative average error.
        
        Args:
            show: Whether to display the plot immediately
            
        Returns:
            Plotly figure object
        """
        n = len(self.errors)
        
        # Calculate running statistics
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]
        
        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]
        
        # 95% confidence interval
        ci = [
            1.96 * (sd / math.sqrt(i)) if i > 1 else 0
            for i, sd in zip(x, running_stds)
        ]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]
        
        # Create plot
        fig = go.Figure()
        
        # Confidence interval band
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )
        
        # Main trend line
        final_mean = running_means[-1]
        final_ci = ci[-1]
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Average Error",
                customdata=list(zip(ci)),
                hovertemplate=(
                    "Sample: %{x}<br>"
                    "Avg Error: $%{y:,.2f}<br>"
                    "±95% CI: $%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )
        
        title = f"{self.name} Error Trend: ${final_mean:,.2f} ± ${final_ci:,.2f}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Number of Samples",
            yaxis_title="Average Absolute Error ($)",
            width=1000,
            height=400,
            template="plotly_white",
            showlegend=False,
        )
        
        if show:
            fig.show()
        
        return fig
    
    def generate_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            save_path: Optional path to save report as JSON
            
        Returns:
            Dictionary with full report data
        """
        metrics = self._calculate_metrics()
        
        report = {
            "model_name": self.name,
            "metrics": metrics,
            "model_info": getattr(self.predictor, 'get_stats', lambda: {})()
            if hasattr(self.predictor, 'get_stats') else {},
        }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Report saved to {save_path}")
        
        return report


def evaluate(
    predictor: Callable[[Item], float],
    test_data: List[Item],
    name: Optional[str] = None,
    max_samples: Optional[int] = None,
    num_workers: int = 5,
    plot: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for quick evaluation.
    
    Args:
        predictor: Function that predicts price from Item
        test_data: List of test items
        name: Optional name for evaluation
        max_samples: Maximum samples to evaluate
        num_workers: Number of parallel workers
        plot: Whether to generate plots
        
    Returns:
        Dictionary with evaluation metrics
    """
    evaluator = PriceEvaluator(
        predictor=predictor,
        test_data=test_data,
        name=name,
        max_samples=max_samples,
        num_workers=num_workers,
    )
    
    metrics = evaluator.evaluate()
    
    if plot:
        evaluator.plot_error_trend()
        evaluator.plot_scatter()
    
    return metrics
