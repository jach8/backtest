"""Configuration dataclasses for backtesting parameters."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import numpy as np
from datetime import datetime

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    
    # Required Parameters
    initial_capital: float = field(default=100_000)
    shares: int = field(default=100)  # Default shares per trade
    commission: float = field(default=5.95)  # Commission per trade
    impact: float = field(default=0.005)  # Market impact
    
    # Optional Parameters
    use_random_impact: bool = field(default=False)
    random_impact_std: float = field(default=0.002)
    
    # Risk Management (optional)
    max_position_pct: float = field(default=0.2)  # Max position size as pct of portfolio
    stop_loss_pct: float = field(default=0.05)  # Stop loss percentage
    take_profit_pct: float = field(default=0.1)  # Take profit percentage
    
    # Trading Rules (optional)
    entry_threshold: float = field(default=0.003)  # Min threshold for entry
    exit_threshold: float = field(default=0.003)  # Min threshold for exit
    max_holding_days: int = field(default=5)  # Max days to hold position
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        
        if self.shares <= 0:
            raise ValueError("Shares must be positive")
            
        if self.commission < 0:
            raise ValueError("Commission cannot be negative")
            
        if not 0 <= self.impact < 1:
            raise ValueError("Market impact must be between 0 and 1")
            
        if not 0 <= self.max_position_pct <= 1:
            raise ValueError("Max position size must be between 0 and 1")
            
        if not 0 <= self.stop_loss_pct < 1:
            raise ValueError("Stop loss must be between 0 and 1")
            
        if not 0 < self.take_profit_pct:
            raise ValueError("Take profit must be positive")
    
    def get_impact(self) -> float:
        """Get total market impact including random component if enabled."""
        total_impact = self.impact
        if self.use_random_impact:
            total_impact += abs(np.random.normal(0, self.random_impact_std))
        return total_impact

    def calculate_shares(self, available_capital: float, price: float) -> int:
        """Calculate number of shares based on position limits.
        
        Args:
            available_capital (float): Current available capital
            price (float): Current asset price
            
        Returns:
            int: Number of shares to trade
        """
        max_shares = int((available_capital * self.max_position_pct) / price)
        return min(self.shares, max_shares)

    def should_exit_position(
        self, 
        entry_price: float, 
        current_price: float,
        days_held: int,
        position_type: str
    ) -> bool:
        """Check if position should be exited based on configured rules.
        
        Args:
            entry_price (float): Price at which position was entered
            current_price (float): Current price
            days_held (int): Number of days position has been held
            position_type (str): Either 'LONG' or 'SHORT'
            
        Returns:
            bool: True if position should be exited
        """
        if days_held >= self.max_holding_days:
            return True
            
        price_change = (current_price - entry_price) / entry_price
        
        if position_type == 'LONG':
            if price_change <= -self.stop_loss_pct or price_change >= self.take_profit_pct:
                return True
        elif position_type == 'SHORT':
            if price_change >= self.stop_loss_pct or price_change <= -self.take_profit_pct:
                return True
                
        return False

    @classmethod
    def default_config(cls) -> 'BacktestConfig':
        """Create a BacktestConfig with default parameters."""
        return cls()

    def to_dict(self) -> Dict[str, Union[float, int, bool]]:
        """Convert config to dictionary format."""
        return {
            'initial_capital': self.initial_capital,
            'shares': self.shares,
            'commission': self.commission,
            'impact': self.impact,
            'use_random_impact': self.use_random_impact,
            'random_impact_std': self.random_impact_std,
            'max_position_pct': self.max_position_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'entry_threshold': self.entry_threshold,
            'exit_threshold': self.exit_threshold,
            'max_holding_days': self.max_holding_days
        }
    
def main():
    # Example usage
    config = BacktestConfig(
        initial_capital=100_000,
        shares=100,
        commission=5.95,
        impact = 0.005,
        use_random_impact=True,
        random_impact_std=0.002,
        max_position_pct=0.2,
        stop_loss_pct=0.05,
        take_profit_pct=0.1,
        entry_threshold=0.003,
        exit_threshold=0.003,
        max_holding_days=5
    )
    print(config.to_dict())


if __name__ == "__main__":
    main()