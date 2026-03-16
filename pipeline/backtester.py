"""
pipeline/backtester.py
──────────────────────
Binance spot backtester for BTC trading signals.

Simulates spot-only (no shorting) trading with realistic fees
and slippage on weekly signals from the prediction model.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    btc_amount: float
    pnl_usdt: float
    pnl_pct: float
    side: str = "LONG"


@dataclass
class BacktestResult:
    """Full backtest results."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[np.ndarray] = None
    dates: Optional[np.ndarray] = None
    prices: Optional[np.ndarray] = None
    total_return_pct: float = 0.0
    annualized_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration_days: int = 0
    n_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    calmar_ratio: float = 0.0
    buy_hold_return_pct: float = 0.0
    final_equity: float = 0.0
    initial_capital: float = 0.0


class BinanceSpotBacktester:
    """
    Simulates Binance spot trading with weekly signals.

    Parameters
    ----------
    initial_capital : float
        Starting USDT balance.
    taker_fee : float
        Taker fee rate (e.g., 0.001 for 0.1%).
    slippage_pct : float
        Slippage as a fraction (e.g., 0.0005 for 0.05%).
    position_size_pct : float
        Fraction of capital to deploy per trade.
    confidence_threshold : float
        Minimum p_up to trigger a BUY.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        taker_fee: float = 0.001,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 0.95,
        confidence_threshold: float = 0.55,
    ):
        self.initial_capital = initial_capital
        self.taker_fee = taker_fee
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        self.confidence_threshold = confidence_threshold

    def run(
        self,
        dates: np.ndarray,
        prices: np.ndarray,
        p_up: np.ndarray,
    ) -> BacktestResult:
        """
        Run the backtest.

        Parameters
        ----------
        dates : array of datetime-like
        prices : array of BTC close prices
        p_up : array of predicted probability of price going up

        Returns
        -------
        BacktestResult
        """
        n = len(dates)
        if n == 0:
            return BacktestResult(initial_capital=self.initial_capital)

        usdt_balance = self.initial_capital
        btc_balance = 0.0
        in_position = False
        entry_price = 0.0
        entry_date = ""
        entry_usdt = 0.0

        trades: List[Trade] = []
        equity = np.zeros(n)

        for i in range(n):
            price = float(prices[i])
            prob = float(p_up[i])
            date_str = str(dates[i])[:10]

            if not in_position and prob >= self.confidence_threshold:
                # BUY
                cost_fraction = self.taker_fee + self.slippage_pct
                invest_usdt = usdt_balance * self.position_size_pct
                effective_price = price * (1 + cost_fraction)
                btc_amount = invest_usdt / effective_price
                usdt_balance -= invest_usdt
                btc_balance = btc_amount
                in_position = True
                entry_price = price
                entry_date = date_str
                entry_usdt = invest_usdt

            elif in_position and prob < (1 - self.confidence_threshold):
                # SELL
                cost_fraction = self.taker_fee + self.slippage_pct
                effective_price = price * (1 - cost_fraction)
                proceeds = btc_balance * effective_price
                pnl = proceeds - entry_usdt
                pnl_pct = pnl / entry_usdt if entry_usdt > 0 else 0.0

                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date_str,
                    entry_price=entry_price,
                    exit_price=price,
                    btc_amount=btc_balance,
                    pnl_usdt=pnl,
                    pnl_pct=pnl_pct,
                ))

                usdt_balance += proceeds
                btc_balance = 0.0
                in_position = False

            # Portfolio value
            portfolio_value = usdt_balance + btc_balance * price
            equity[i] = portfolio_value

        # Close open position at end
        if in_position and n > 0:
            price = float(prices[-1])
            cost_fraction = self.taker_fee + self.slippage_pct
            effective_price = price * (1 - cost_fraction)
            proceeds = btc_balance * effective_price
            pnl = proceeds - entry_usdt
            pnl_pct = pnl / entry_usdt if entry_usdt > 0 else 0.0

            trades.append(Trade(
                entry_date=entry_date,
                exit_date=str(dates[-1])[:10],
                entry_price=entry_price,
                exit_price=price,
                btc_amount=btc_balance,
                pnl_usdt=pnl,
                pnl_pct=pnl_pct,
            ))
            usdt_balance += proceeds
            btc_balance = 0.0
            equity[-1] = usdt_balance

        result = self._compute_metrics(trades, equity, dates, prices)
        return result

    def _compute_metrics(
        self,
        trades: List[Trade],
        equity: np.ndarray,
        dates: np.ndarray,
        prices: np.ndarray,
    ) -> BacktestResult:
        """Compute all backtest metrics."""
        n = len(equity)
        if n == 0:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_equity=self.initial_capital,
            )

        final_equity = equity[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        # Annualized return
        n_days = max(n, 1)
        ann_return = (1 + total_return) ** (365.25 / n_days) - 1 if n_days > 0 else 0

        # Daily returns from equity curve
        daily_returns = np.diff(equity) / equity[:-1] if n > 1 else np.array([0.0])
        daily_returns = np.nan_to_num(daily_returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Sharpe ratio (annualized, risk-free = 0)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(365.25)
        else:
            sharpe = 0.0

        # Sortino ratio
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and np.std(downside) > 0:
            sortino = np.mean(daily_returns) / np.std(downside) * np.sqrt(365.25)
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        # Max drawdown duration
        dd_duration = 0
        max_dd_duration = 0
        for i in range(len(drawdown)):
            if drawdown[i] < 0:
                dd_duration += 1
                max_dd_duration = max(max_dd_duration, dd_duration)
            else:
                dd_duration = 0

        # Trade stats
        n_trades = len(trades)
        wins = [t for t in trades if t.pnl_usdt > 0]
        losses = [t for t in trades if t.pnl_usdt <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win = np.mean([t.pnl_usdt for t in wins]) if wins else 0.0
        avg_loss = np.mean([abs(t.pnl_usdt) for t in losses]) if losses else 0.0
        gross_profit = sum(t.pnl_usdt for t in wins) if wins else 0.0
        gross_loss = sum(abs(t.pnl_usdt) for t in losses) if losses else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calmar ratio
        calmar = ann_return / abs(max_dd) if abs(max_dd) > 0 else 0.0

        # Buy and hold
        if len(prices) > 1:
            bh_return = (float(prices[-1]) - float(prices[0])) / float(prices[0])
        else:
            bh_return = 0.0

        return BacktestResult(
            trades=trades,
            equity_curve=equity,
            dates=dates,
            prices=prices,
            total_return_pct=total_return * 100,
            annualized_return_pct=ann_return * 100,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd * 100,
            max_drawdown_duration_days=max_dd_duration,
            n_trades=n_trades,
            win_rate=win_rate * 100,
            avg_win=float(avg_win),
            avg_loss=float(avg_loss),
            profit_factor=profit_factor,
            calmar_ratio=calmar,
            buy_hold_return_pct=bh_return * 100,
            final_equity=final_equity,
            initial_capital=self.initial_capital,
        )


def run_backtest(
    dates: np.ndarray,
    prices: np.ndarray,
    p_up: np.ndarray,
    config: dict,
) -> BacktestResult:
    """Convenience function to run a backtest from config."""
    bt = BinanceSpotBacktester(
        initial_capital=config.get("bt_initial_capital", 10000),
        taker_fee=config.get("bt_taker_fee", 0.001),
        slippage_pct=config.get("bt_slippage_pct", 0.0005),
        position_size_pct=config.get("bt_position_size_pct", 0.95),
        confidence_threshold=config.get("bt_confidence_threshold", 0.55),
    )
    return bt.run(dates, prices, p_up)


def save_backtest_outputs(result: BacktestResult, config: dict) -> None:
    """Save all backtest outputs (trade log, charts, summary JSON)."""
    out_dir = config.get("output_dir", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # Trade log CSV
    if result.trades:
        rows = []
        for t in result.trades:
            rows.append({
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "btc_amount": round(t.btc_amount, 6),
                "pnl_usdt": round(t.pnl_usdt, 2),
                "pnl_pct": round(t.pnl_pct * 100, 2),
            })
        df_trades = pd.DataFrame(rows)
        path = os.path.join(out_dir, "trade_log.csv")
        df_trades.to_csv(path, index=False)
        logger.info("Saved trade log: %s (%d trades)", path, len(rows))

    # Summary JSON
    summary = {
        "total_return_pct": round(result.total_return_pct, 2),
        "annualized_return_pct": round(result.annualized_return_pct, 2),
        "sharpe_ratio": round(result.sharpe_ratio, 3),
        "sortino_ratio": round(result.sortino_ratio, 3),
        "max_drawdown_pct": round(result.max_drawdown_pct, 2),
        "max_drawdown_duration_days": result.max_drawdown_duration_days,
        "n_trades": result.n_trades,
        "win_rate_pct": round(result.win_rate, 1),
        "avg_win_usdt": round(result.avg_win, 2),
        "avg_loss_usdt": round(result.avg_loss, 2),
        "profit_factor": round(result.profit_factor, 3) if result.profit_factor != float("inf") else "inf",
        "calmar_ratio": round(result.calmar_ratio, 3),
        "buy_hold_return_pct": round(result.buy_hold_return_pct, 2),
        "final_equity": round(result.final_equity, 2),
        "initial_capital": result.initial_capital,
    }
    path = os.path.join(out_dir, "backtest_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved backtest summary: %s", path)

    # Equity curve plot
    if _HAS_MPL and result.equity_curve is not None and result.prices is not None:
        _plot_equity_curve(result, out_dir)
        _plot_drawdown(result, out_dir)
        _plot_monthly_returns(result, out_dir)


def _plot_equity_curve(result: BacktestResult, out_dir: str) -> None:
    """Plot equity curve: strategy vs buy-and-hold."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(result.equity_curve, label="Strategy", linewidth=1.5)

    # Buy and hold equity
    prices = result.prices.astype(float)
    bh_equity = result.initial_capital * prices / prices[0]
    ax.plot(bh_equity, label="Buy & Hold", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Day")
    ax.set_ylabel("Portfolio Value (USDT)")
    ax.set_title("Equity Curve: Strategy vs Buy & Hold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "equity_curve_backtest.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved equity curve: %s", path)


def _plot_drawdown(result: BacktestResult, out_dir: str) -> None:
    """Plot drawdown chart."""
    equity = result.equity_curve
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.4, color="red")
    ax.plot(drawdown, color="red", linewidth=0.8)
    ax.set_xlabel("Day")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Portfolio Drawdown")
    ax.grid(True, alpha=0.3)

    path = os.path.join(out_dir, "drawdown_chart.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    logger.info("Saved drawdown chart: %s", path)


def _plot_monthly_returns(result: BacktestResult, out_dir: str) -> None:
    """Plot monthly returns heatmap."""
    if result.dates is None or result.equity_curve is None:
        return

    try:
        dates = pd.to_datetime(result.dates)
        equity = result.equity_curve

        df = pd.DataFrame({"date": dates, "equity": equity})
        df = df.set_index("date")
        df["daily_ret"] = df["equity"].pct_change()

        monthly = df["daily_ret"].resample("M").apply(lambda x: (1 + x).prod() - 1)
        monthly_df = pd.DataFrame({
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values * 100,
        })

        pivot = monthly_df.pivot_table(values="return", index="year", columns="month", aggfunc="first")
        pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=(12, max(3, len(pivot) * 0.6)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-20, vmax=20)
        fig.colorbar(im, ax=ax, label="Monthly Return (%)")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Monthly Returns Heatmap (%)")

        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=8, color="black")

        path = os.path.join(out_dir, "monthly_returns_heatmap.png")
        fig.savefig(path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        logger.info("Saved monthly returns heatmap: %s", path)
    except Exception as e:
        logger.warning("Monthly returns heatmap failed: %s", e)


def print_backtest_summary(result: BacktestResult) -> None:
    """Print a formatted backtest summary to console."""
    print("\n" + "=" * 60)
    print("         BACKTEST RESULTS (Binance Spot)")
    print("=" * 60)
    print(f"  Initial Capital:       ${result.initial_capital:,.2f}")
    print(f"  Final Equity:          ${result.final_equity:,.2f}")
    print(f"  Total Return:          {result.total_return_pct:+.2f}%")
    print(f"  Annualized Return:     {result.annualized_return_pct:+.2f}%")
    print(f"  Buy & Hold Return:     {result.buy_hold_return_pct:+.2f}%")
    print("-" * 60)
    print(f"  Sharpe Ratio:          {result.sharpe_ratio:.3f}")
    print(f"  Sortino Ratio:         {result.sortino_ratio:.3f}")
    print(f"  Calmar Ratio:          {result.calmar_ratio:.3f}")
    print(f"  Max Drawdown:          {result.max_drawdown_pct:.2f}%")
    print(f"  Max DD Duration:       {result.max_drawdown_duration_days} days")
    print("-" * 60)
    print(f"  Number of Trades:      {result.n_trades}")
    print(f"  Win Rate:              {result.win_rate:.1f}%")
    print(f"  Avg Win:               ${result.avg_win:.2f}")
    print(f"  Avg Loss:              ${result.avg_loss:.2f}")
    pf = f"{result.profit_factor:.3f}" if result.profit_factor != float("inf") else "inf"
    print(f"  Profit Factor:         {pf}")
    print("=" * 60 + "\n")
