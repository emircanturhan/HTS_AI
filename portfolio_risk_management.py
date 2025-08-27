import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
import cvxpy as cp
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# ============================================
# PORTFOLIO OPTIMIZER - MODERN PORTFOLIO TEORİSİ
# ============================================

@dataclass
class PortfolioMetrics:
    """Portfolio metrikleri"""
    expected_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk %95
    cvar_95: float  # Conditional VaR
    calmar_ratio: float
    beta: float
    alpha: float
    information_ratio: float

class ModernPortfolioOptimizer:
    """Modern Portfolio Theory (MPT) implementasyonu"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_history = []
        
    def calculate_portfolio_metrics(self, returns: pd.DataFrame, 
                                   weights: np.array) -> PortfolioMetrics:
        """Portfolio metriklerini hesapla"""
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Expected return and volatility
        expected_return = np.mean(portfolio_returns) * 252  # Annualized
        volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252)
        sortino_ratio = (expected_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum Drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) - %95 confidence
        var_95 = np.percentile(portfolio_returns, 5) * np.sqrt(252)
        
        # Conditional Value at Risk (CVaR)
        var_threshold = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_threshold].mean() * np.sqrt(252)
        
        # Calmar Ratio
        calmar_ratio = expected_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Beta (vs market - using BTC as market proxy)
        if 'BTC' in returns.columns:
            market_returns = returns['BTC']
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance > 0 else 1
        else:
            beta = 1.0
        
        # Alpha (Jensen's Alpha)
        alpha = expected_return - (self.risk_free_rate + beta * (0.5 - self.risk_free_rate))
        
        # Information Ratio
        tracking_error = np.std(portfolio_returns - market_returns if 'BTC' in returns.columns else portfolio_returns)
        information_ratio = alpha / (tracking_error * np.sqrt(252)) if tracking_error > 0 else 0
        
        return PortfolioMetrics(
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            calmar_ratio=calmar_ratio,
            beta=beta,
            alpha=alpha,
            information_ratio=information_ratio
        )
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                          optimization_method: str = 'max_sharpe',
                          constraints: Optional[Dict] = None) -> Dict:
        """Portfolio optimizasyonu"""
        
        n_assets = len(returns.columns)
        
        # Default constraints
        if constraints is None:
            constraints = {
                'min_weight': 0.0,
                'max_weight': 0.4,  # Max %40 per asset
                'target_return': None,
                'max_volatility': None
            }
        
        # Optimization based on method
        if optimization_method == 'max_sharpe':
            weights = self._optimize_max_sharpe(returns, constraints)
        elif optimization_method == 'min_variance':
            weights = self._optimize_min_variance(returns, constraints)
        elif optimization_method == 'max_diversification':
            weights = self._optimize_max_diversification(returns, constraints)
        elif optimization_method == 'risk_parity':
            weights = self._optimize_risk_parity(returns)
        elif optimization_method == 'black_litterman':
            weights = self._black_litterman_optimization(returns, constraints)
        elif optimization_method == 'hierarchical_risk_parity':
            weights = self._hierarchical_risk_parity(returns)
        else:
            weights = np.array([1/n_assets] * n_assets)  # Equal weight
        
        # Calculate metrics
        metrics = self.calculate_portfolio_metrics(returns, weights)
        
        # Create result
        result = {
            'weights': dict(zip(returns.columns, weights)),
            'metrics': metrics,
            'optimization_method': optimization_method,
            'timestamp': datetime.now()
        }
        
        self.optimization_history.append(result)
        
        return result
    
    def _optimize_max_sharpe(self, returns: pd.DataFrame, 
                            constraints: Dict) -> np.array:
        """Maximum Sharpe Ratio optimizasyonu"""
        
        n_assets = len(returns.columns)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -(portfolio_return - self.risk_free_rate/252) / portfolio_std
        
        # Constraints
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds
        bounds = tuple(
            (constraints['min_weight'], constraints['max_weight']) 
            for _ in range(n_assets)
        )
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _optimize_min_variance(self, returns: pd.DataFrame, 
                               constraints: Dict) -> np.array:
        """Minimum Variance optimizasyonu"""
        
        n_assets = len(returns.columns)
        cov_matrix = returns.cov()
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(
            (constraints['min_weight'], constraints['max_weight']) 
            for _ in range(n_assets)
        )
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _optimize_max_diversification(self, returns: pd.DataFrame, 
                                     constraints: Dict) -> np.array:
        """Maximum Diversification Ratio optimizasyonu"""
        
        n_assets = len(returns.columns)
        cov_matrix = returns.cov()
        std_devs = np.sqrt(np.diag(cov_matrix))
        
        def negative_diversification_ratio(weights):
            weighted_avg_volatility = np.dot(weights, std_devs)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -weighted_avg_volatility / portfolio_volatility
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(
            (constraints['min_weight'], constraints['max_weight']) 
            for _ in range(n_assets)
        )
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            negative_diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _optimize_risk_parity(self, returns: pd.DataFrame) -> np.array:
        """Risk Parity optimizasyonu - eşit risk katkısı"""
        
        n_assets = len(returns.columns)
        cov_matrix = returns.cov().values
        
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            avg_contrib = np.mean(contrib)
            return np.sum((contrib - avg_contrib)**2)
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x
    
    def _black_litterman_optimization(self, returns: pd.DataFrame, 
                                     constraints: Dict,
                                     views: Optional[Dict] = None) -> np.array:
        """Black-Litterman Model optimizasyonu"""
        
        n_assets = len(returns.columns)
        
        # Market equilibrium weights (market cap weighted - simplified)
        market_weights = np.array([1/n_assets] * n_assets)
        
        # Prior (equilibrium) returns
        cov_matrix = returns.cov()
        lam = 2.5  # Risk aversion parameter
        equilibrium_returns = lam * np.dot(cov_matrix, market_weights)
        
        if views is None:
            # No views, return market weights
            return market_weights
        
        # Views matrix P and Q
        P = views.get('P', np.eye(n_assets))
        Q = views.get('Q', equilibrium_returns)
        omega = views.get('omega', np.eye(len(Q)) * 0.025)
        
        # Black-Litterman formula
        tau = 0.05
        M_inverse = np.linalg.inv(tau * cov_matrix)
        
        # Posterior returns
        posterior_returns = np.linalg.inv(
            M_inverse + np.dot(np.dot(P.T, np.linalg.inv(omega)), P)
        ).dot(
            np.dot(M_inverse, equilibrium_returns) + 
            np.dot(np.dot(P.T, np.linalg.inv(omega)), Q)
        )
        
        # Optimize with posterior returns
        def objective(weights):
            return -np.dot(weights, posterior_returns) + \
                   0.5 * lam * np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(
            (constraints['min_weight'], constraints['max_weight']) 
            for _ in range(n_assets)
        )
        x0 = market_weights
        
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return result.x
    
    def _hierarchical_risk_parity(self, returns: pd.DataFrame) -> np.array:
        """Hierarchical Risk Parity (HRP) - Marcos López de Prado"""
        
        import scipy.cluster.hierarchy as sch
        from scipy.spatial.distance import squareform
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        # Distance matrix
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        
        # Hierarchical clustering
        link = sch.linkage(squareform(dist_matrix), 'single')
        
        # Quasi-diagonalization
        def get_quasi_diag(link):
            link = link.astype(int)
            sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
            num_items = link[-1, 3]
            
            while sort_ix.max() >= num_items:
                sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
                df0 = sort_ix[sort_ix >= num_items]
                i = df0.index
                j = df0.values - num_items
                sort_ix[i] = link[j, 0]
                df0 = pd.Series(link[j, 1], index=i + 1)
                sort_ix = pd.concat([sort_ix, df0])
                sort_ix = sort_ix.sort_index()
                sort_ix.index = range(sort_ix.shape[0])
            
            return sort_ix.tolist()
        
        sort_ix = get_quasi_diag(link)
        
        # Recursive bisection
        def get_recursive_bisection(cov, sort_ix):
            w = pd.Series(1, index=sort_ix)
            c_items = [sort_ix]
            
            while len(c_items) > 0:
                c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
                
                for i in range(0, len(c_items), 2):
                    c_items0 = c_items[i]
                    c_items1 = c_items[i + 1] if i + 1 < len(c_items) else []
                    
                    if len(c_items1) > 0:
                        c_var0 = self._get_cluster_var(cov, c_items0)
                        c_var1 = self._get_cluster_var(cov, c_items1)
                        alpha = 1 - c_var0 / (c_var0 + c_var1)
                        
                        w[c_items0] *= alpha
                        w[c_items1] *= 1 - alpha
            
            return w
        
        cov = returns.cov()
        weights = get_recursive_bisection(cov, sort_ix)
        
        # Reorder to match original columns
        final_weights = np.zeros(len(returns.columns))
        for i, col in enumerate(returns.columns):
            if col in weights.index:
                idx = returns.columns.get_loc(col)
                final_weights[idx] = weights[col]
        
        return final_weights / final_weights.sum()
    
    def _get_cluster_var(self, cov, c_items):
        """Cluster variance hesapla"""
        cov_slice = cov.iloc[c_items, c_items]
        w = pd.Series(1, index=cov_slice.index) / len(c_items)
        c_var = np.dot(w, np.dot(cov_slice, w))
        return c_var
    
    def efficient_frontier(self, returns: pd.DataFrame, 
                          num_portfolios: int = 100) -> pd.DataFrame:
        """Efficient Frontier hesapla"""
        
        results = []
        
        # Generate target returns
        min_ret = returns.mean().min() * 252
        max_ret = returns.mean().max() * 252
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        for target in target_returns:
            constraints = {
                'min_weight': 0,
                'max_weight': 1,
                'target_return': target
            }
            
            # Optimize for minimum variance given target return
            weights = self._optimize_for_target_return(returns, target)
            metrics = self.calculate_portfolio_metrics(returns, weights)
            
            results.append({
                'return': metrics.expected_return,
                'volatility': metrics.volatility,
                'sharpe': metrics.sharpe_ratio
            })
        
        return pd.DataFrame(results)
    
    def _optimize_for_target_return(self, returns: pd.DataFrame, 
                                   target_return: float) -> np.array:
        """Belirli getiri hedefi için minimum varyans"""
        
        n_assets = len(returns.columns)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}
        ]
        
        bounds = tuple((0, 1) for _ in range(n_assets))
        x0 = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        return result.x

# ============================================
# RISK MANAGEMENT SYSTEM
# ============================================

class RiskManagementSystem:
    """Kapsamlı risk yönetimi sistemi"""
    
    def __init__(self):
        self.risk_limits = {
            'max_position_size': 0.1,  # %10
            'max_sector_exposure': 0.3,  # %30
            'max_correlation': 0.7,
            'max_var_95': -0.05,  # -%5
            'max_leverage': 2.0,
            'min_liquidity_ratio': 0.2
        }
        
        self.risk_metrics = {}
        self.alerts = []
        
    def calculate_var_cvar(self, returns: pd.Series, 
                          confidence_level: float = 0.95,
                          method: str = 'historical') -> Tuple[float, float]:
        """Value at Risk ve Conditional VaR hesaplama"""
        
        if method == 'historical':
            # Historical VaR
            var = np.percentile(returns, (1 - confidence_level) * 100)
            # CVaR (Expected Shortfall)
            cvar = returns[returns <= var].mean()
            
        elif method == 'parametric':
            # Parametric VaR (assuming normal distribution)
            mu = returns.mean()
            sigma = returns.std()
            var = norm.ppf(1 - confidence_level, mu, sigma)
            # CVaR for normal distribution
            cvar = mu - sigma * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            simulations = self._monte_carlo_simulation(returns, n_simulations=10000)
            var = np.percentile(simulations, (1 - confidence_level) * 100)
            cvar = simulations[simulations <= var].mean()
            
        else:  # Cornish-Fisher expansion (accounting for skewness and kurtosis)
            mu = returns.mean()
            sigma = returns.std()
            s = skew(returns)
            k = kurtosis(returns)
            
            z = norm.ppf(1 - confidence_level)
            z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * (k - 3) / 24 - (2*z**3 - 5*z) * s**2 / 36
            
            var = mu + sigma * z_cf
            cvar = mu + sigma * (-norm.pdf(z) / (1 - confidence_level))
        
        return var, cvar
    
    def _monte_carlo_simulation(self, returns: pd.Series, 
                               n_simulations: int = 10000) -> np.array:
        """Monte Carlo simülasyonu"""
        
        mu = returns.mean()
        sigma = returns.std()
        
        # GBM simulation
        dt = 1/252
        simulated_returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_simulations)
        
        return simulated_returns
    
    def calculate_risk_metrics(self, portfolio: pd.DataFrame, 
                              weights: np.array) -> Dict:
        """Tüm risk metriklerini hesapla"""
        
        portfolio_returns = (portfolio * weights).sum(axis=1)
        
        # Basic metrics
        metrics = {
            'daily_var_95': self.calculate_var_cvar(portfolio_returns, 0.95, 'historical')[0],
            'daily_cvar_95': self.calculate_var_cvar(portfolio_returns, 0.95, 'historical')[1],
            'daily_var_99': self.calculate_var_cvar(portfolio_returns, 0.99, 'historical')[0],
            'daily_cvar_99': self.calculate_var_cvar(portfolio_returns, 0.99, 'historical')[1],
            'annual_volatility': portfolio_returns.std() * np.sqrt(252),
            'downside_deviation': portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'skewness': skew(portfolio_returns),
            'kurtosis': kurtosis(portfolio_returns),
            'tail_ratio': abs(np.percentile(portfolio_returns, 95) / np.percentile(portfolio_returns, 5))
        }
        
        # Stress testing
        stress_results = self.stress_testing(portfolio, weights)
        metrics['stress_test'] = stress_results
        
        # Liquidity risk
        metrics['liquidity_score'] = self._calculate_liquidity_risk(portfolio, weights)
        
        # Correlation risk
        metrics['correlation_matrix'] = portfolio.corr()
        metrics['max_correlation'] = self._get_max_correlation(portfolio.corr())
        
        self.risk_metrics = metrics
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Maximum drawdown hesapla"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_liquidity_risk(self, portfolio: pd.DataFrame, 
                                 weights: np.array) -> float:
        """Likidite riski skoru"""
        # Simplified liquidity score based on volume
        # In production, would use actual volume data
        liquidity_score = 1.0  # Placeholder
        return liquidity_score
    
    def _get_max_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Maximum correlation (excluding diagonal)"""
        np.fill_diagonal(corr_matrix.values, 0)
        return corr_matrix.abs().max().max()
    
    def stress_testing(self, portfolio: pd.DataFrame, 
                      weights: np.array) -> Dict:
        """Stress testing scenarios"""
        
        scenarios = {
            'market_crash': -0.20,  # -%20 market crash
            'flash_crash': -0.10,   # -%10 flash crash
            'black_swan': -0.30,    # -%30 black swan event
            'correlation_breakdown': 1.0,  # All correlations go to 1
            'liquidity_crisis': 0.5  # Liquidity dries up
        }
        
        results = {}
        
        for scenario, impact in scenarios.items():
            if scenario == 'correlation_breakdown':
                # All assets move together
                stressed_returns = portfolio.mean(axis=1) * impact
            else:
                # Apply shock
                stressed_returns = (portfolio * weights).sum(axis=1) * (1 + impact)
            
            results[scenario] = {
                'portfolio_impact': stressed_returns.sum(),
                'var_95': self.calculate_var_cvar(stressed_returns, 0.95)[0],
                'cvar_95': self.calculate_var_cvar(stressed_returns, 0.95)[1]
            }
        
        return results
    
    def position_sizing_kelly(self, win_probability: float, 
                            win_loss_ratio: float,
                            confidence_adjustment: float = 0.25) -> float:
        """Kelly Criterion for position sizing"""
        
        # Kelly formula: f = (p * b - q) / b
        # where f = fraction to bet, p = win prob, q = lose prob, b = win/loss ratio
        
        q = 1 - win_probability
        kelly_fraction = (win_probability * win_loss_ratio - q) / win_loss_ratio
        
        # Apply confidence adjustment (fractional Kelly)
        adjusted_fraction = kelly_fraction * confidence_adjustment
        
        # Cap at maximum position size
        return min(adjusted_fraction, self.risk_limits['max_position_size'])
    
    def check_risk_limits(self, portfolio_metrics: Dict) -> List[Dict]:
        """Risk limitlerini kontrol et"""
        
        violations = []
        
        # VaR limit check
        if portfolio_metrics['daily_var_95'] < self.risk_limits['max_var_95']:
            violations.append({
                'type': 'VaR_LIMIT_BREACH',
                'current': portfolio_metrics['daily_var_95'],
                'limit': self.risk_limits['max_var_95'],
                'severity': 'HIGH',
                'action': 'REDUCE_POSITION'
            })
        
        # Correlation limit check
        if portfolio_metrics['max_correlation'] > self.risk_limits['max_correlation']:
            violations.append({
                'type': 'CORRELATION_LIMIT_BREACH',
                'current': portfolio_metrics['max_correlation'],
                'limit': self.risk_limits['max_correlation'],
                'severity': 'MEDIUM',
                'action': 'DIVERSIFY'
            })
        
        # Liquidity check
        if portfolio_metrics['liquidity_score'] < self.risk_limits['min_liquidity_ratio']:
            violations.append({
                'type': 'LIQUIDITY_WARNING',
                'current': portfolio_metrics['liquidity_score'],
                'limit': self.risk_limits['min_liquidity_ratio'],
                'severity': 'MEDIUM',
                'action': 'INCREASE_LIQUIDITY'
            })
        
        return violations
    
    def generate_risk_report(self) -> str:
        """Risk raporu oluştur"""
        
        report = "📊 RİSK YÖNETİMİ RAPORU\n"
        report += "=" * 50 + "\n\n"
        
        if self.risk_metrics:
            report += "📈 TEMEL RİSK METRİKLERİ:\n"
            report += f"  • VaR (95%): {self.risk_metrics['daily_var_95']:.2%}\n"
            report += f"  • CVaR (95%): {self.risk_metrics['daily_cvar_95']:.2%}\n"
            report += f"  • Volatilite: {self.risk_metrics['annual_volatility']:.2%}\n"
            report += f"  • Max Drawdown: {self.risk_metrics['max_drawdown']:.2%}\n"
            report += f"  • Skewness: {self.risk_metrics['skewness']:.3f}\n"
            report += f"  • Kurtosis: {self.risk_metrics['kurtosis']:.3f}\n\n"
            
            report += "🔥 STRES TESTİ SONUÇLARI:\n"
            for scenario, result in self.risk_metrics.get('stress_test', {}).items():
                report += f"  • {scenario}: {result['portfolio_impact']:.2%} etki\n"
            
            report += "\n⚠️ RİSK UYARILARI:\n"
            violations = self.check_risk_limits(self.risk_metrics)
            if violations:
                for violation in violations:
                    report += f"  • {violation['type']}: {violation['action']}\n"
            else:
                report += "  • Tüm risk limitleri normal sınırlar içinde\n"
        
        return report