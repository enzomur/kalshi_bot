"""Report generation for backtest results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from kalshi_bot.backtesting.engine import BacktestResult
from kalshi_bot.backtesting.metrics import BacktestMetrics, format_metrics_summary
from kalshi_bot.backtesting.position_tracker import PositionTracker
from kalshi_bot.utils.logging import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generates reports from backtest results.

    Supports multiple output formats:
    - Text summary (console output)
    - JSON (programmatic access)
    - Markdown (documentation)
    - HTML (visual reports)
    """

    def __init__(self, result: BacktestResult) -> None:
        """
        Initialize report generator.

        Args:
            result: The backtest result to report on
        """
        self.result = result

    def generate_text_report(self) -> str:
        """Generate a text summary report."""
        lines = [
            "",
            format_metrics_summary(self.result.metrics),
            "",
            "Backtest Configuration",
            "-" * 40,
            f"  Initial Balance:    ${self.result.config.initial_balance:,.2f}",
            f"  Max Position/Market: {self.result.config.max_position_per_market}",
            f"  Max Position %:     {self.result.config.max_position_pct:.0%}",
            f"  Min Profit Thresh:  ${self.result.config.min_profit_threshold:.2f}",
            "",
            "Execution Summary",
            "-" * 40,
            f"  Time Steps:         {self.result.progress.total_steps}",
            f"  Opportunities Found: {self.result.progress.opportunities_found}",
            f"  Trades Executed:    {self.result.progress.trades_executed}",
            f"  Positions Settled:  {self.result.progress.positions_settled}",
            f"  Execution Time:     {self.result.duration_seconds:.1f}s",
            "",
        ]

        if self.result.error_message:
            lines.extend([
                "Errors",
                "-" * 40,
                f"  {self.result.error_message}",
                "",
            ])

        return "\n".join(lines)

    def generate_json_report(self) -> dict[str, Any]:
        """Generate a JSON-serializable report."""
        return {
            "success": self.result.success,
            "error_message": self.result.error_message,
            "execution": {
                "start_time": self.result.start_time.isoformat(),
                "end_time": self.result.end_time.isoformat() if self.result.end_time else None,
                "duration_seconds": self.result.duration_seconds,
            },
            "config": {
                "initial_balance": self.result.config.initial_balance,
                "max_position_per_market": self.result.config.max_position_per_market,
                "max_position_pct": self.result.config.max_position_pct,
                "min_profit_threshold": self.result.config.min_profit_threshold,
                "enable_single_market": self.result.config.enable_single_market,
                "enable_multi_outcome": self.result.config.enable_multi_outcome,
                "start_date": self.result.config.start_date.isoformat() if self.result.config.start_date else None,
                "end_date": self.result.config.end_date.isoformat() if self.result.config.end_date else None,
            },
            "progress": {
                "total_steps": self.result.progress.total_steps,
                "opportunities_found": self.result.progress.opportunities_found,
                "trades_executed": self.result.progress.trades_executed,
                "positions_settled": self.result.progress.positions_settled,
            },
            "metrics": self.result.metrics.to_dict(),
        }

    def generate_markdown_report(self) -> str:
        """Generate a Markdown report."""
        m = self.result.metrics
        p = self.result.progress

        lines = [
            "# Backtest Report",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Initial Balance | ${m.initial_balance:,.2f} |",
            f"| Final Balance | ${m.final_balance:,.2f} |",
            f"| Total P&L | ${m.total_pnl:,.2f} |",
            f"| Total Return | {m.total_return_pct:.2%} |",
            "",
            "## Trade Statistics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {m.total_trades} |",
            f"| Win Rate | {m.win_rate:.2%} |",
            f"| Profit Factor | {m.profit_factor:.2f} |",
            f"| Expectancy | ${m.expectancy:,.2f} |",
            "",
            "## Risk Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Max Drawdown | {m.max_drawdown_pct:.2%} |",
            f"| Sharpe Ratio | {m.sharpe_ratio:.2f} |",
            f"| Sortino Ratio | {m.sortino_ratio:.2f} |",
            f"| Calmar Ratio | {m.calmar_ratio:.2f} |",
            "",
        ]

        if m.strategy_pnl:
            lines.extend([
                "## Strategy Breakdown",
                "",
                "| Strategy | P&L | Trades |",
                "|----------|-----|--------|",
            ])
            for strategy, pnl in sorted(m.strategy_pnl.items()):
                trades = m.strategy_trades.get(strategy, 0)
                lines.append(f"| {strategy} | ${pnl:,.2f} | {trades} |")
            lines.append("")

        lines.extend([
            "## Execution Details",
            "",
            f"- **Time Steps Processed:** {p.total_steps}",
            f"- **Opportunities Found:** {p.opportunities_found}",
            f"- **Trades Executed:** {p.trades_executed}",
            f"- **Positions Settled:** {p.positions_settled}",
            f"- **Execution Time:** {self.result.duration_seconds:.1f} seconds",
            "",
        ])

        return "\n".join(lines)

    def generate_html_report(self) -> str:
        """Generate an HTML report with embedded styles."""
        m = self.result.metrics

        # Determine P&L color
        pnl_color = "#22c55e" if m.total_pnl >= 0 else "#ef4444"

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f8fafc;
            color: #1e293b;
        }}
        h1, h2 {{ color: #0f172a; }}
        .header {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .card-label {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: 600;
        }}
        .card-value.positive {{ color: #22c55e; }}
        .card-value.negative {{ color: #ef4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f1f5f9;
            font-weight: 600;
            color: #475569;
        }}
        tr:last-child td {{ border-bottom: none; }}
        .timestamp {{
            text-align: center;
            color: #64748b;
            font-size: 14px;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Backtest Report</h1>
        <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    </div>

    <div class="summary-grid">
        <div class="card">
            <div class="card-label">Total P&L</div>
            <div class="card-value" style="color: {pnl_color}">${m.total_pnl:,.2f}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Return</div>
            <div class="card-value" style="color: {pnl_color}">{m.total_return_pct:.2%}</div>
        </div>
        <div class="card">
            <div class="card-label">Win Rate</div>
            <div class="card-value">{m.win_rate:.1%}</div>
        </div>
        <div class="card">
            <div class="card-label">Sharpe Ratio</div>
            <div class="card-value">{m.sharpe_ratio:.2f}</div>
        </div>
        <div class="card">
            <div class="card-label">Max Drawdown</div>
            <div class="card-value negative">{m.max_drawdown_pct:.2%}</div>
        </div>
        <div class="card">
            <div class="card-label">Total Trades</div>
            <div class="card-value">{m.total_trades}</div>
        </div>
    </div>

    <h2>Performance Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Initial Balance</td><td>${m.initial_balance:,.2f}</td></tr>
        <tr><td>Final Balance</td><td>${m.final_balance:,.2f}</td></tr>
        <tr><td>Profit Factor</td><td>{m.profit_factor:.2f}</td></tr>
        <tr><td>Expectancy</td><td>${m.expectancy:,.2f}</td></tr>
        <tr><td>Average Win</td><td>${m.avg_win:,.2f}</td></tr>
        <tr><td>Average Loss</td><td>${m.avg_loss:,.2f}</td></tr>
        <tr><td>Largest Win</td><td>${m.largest_win:,.2f}</td></tr>
        <tr><td>Largest Loss</td><td>${m.largest_loss:,.2f}</td></tr>
        <tr><td>Total Fees</td><td>${m.total_fees:,.2f}</td></tr>
    </table>

    <h2>Risk Analysis</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Max Drawdown ($)</td><td>${m.max_drawdown:,.2f}</td></tr>
        <tr><td>Max Drawdown (%)</td><td>{m.max_drawdown_pct:.2%}</td></tr>
        <tr><td>Sharpe Ratio</td><td>{m.sharpe_ratio:.2f}</td></tr>
        <tr><td>Sortino Ratio</td><td>{m.sortino_ratio:.2f}</td></tr>
        <tr><td>Calmar Ratio</td><td>{m.calmar_ratio:.2f}</td></tr>
        <tr><td>Annualized Volatility</td><td>{m.volatility:.2%}</td></tr>
    </table>
"""

        if m.strategy_pnl:
            html += """
    <h2>Strategy Breakdown</h2>
    <table>
        <tr><th>Strategy</th><th>P&L</th><th>Trades</th></tr>
"""
            for strategy, pnl in sorted(m.strategy_pnl.items()):
                trades = m.strategy_trades.get(strategy, 0)
                color = "#22c55e" if pnl >= 0 else "#ef4444"
                html += f'        <tr><td>{strategy}</td><td style="color: {color}">${pnl:,.2f}</td><td>{trades}</td></tr>\n'
            html += "    </table>\n"

        html += f"""
    <p class="timestamp">
        Backtest Duration: {self.result.duration_seconds:.1f}s |
        Time Steps: {self.result.progress.total_steps} |
        Opportunities: {self.result.progress.opportunities_found}
    </p>
</body>
</html>"""

        return html

    def save_report(
        self,
        path: str | Path,
        format: str = "auto",
    ) -> Path:
        """
        Save report to file.

        Args:
            path: Output file path
            format: Format (auto, text, json, markdown, html)

        Returns:
            Path to saved file
        """
        path = Path(path)

        # Auto-detect format from extension
        if format == "auto":
            ext = path.suffix.lower()
            format_map = {
                ".txt": "text",
                ".json": "json",
                ".md": "markdown",
                ".html": "html",
            }
            format = format_map.get(ext, "text")

        # Generate content
        if format == "json":
            content = json.dumps(self.generate_json_report(), indent=2)
        elif format == "markdown":
            content = self.generate_markdown_report()
        elif format == "html":
            content = self.generate_html_report()
        else:
            content = self.generate_text_report()

        # Write file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        logger.info(f"Report saved to {path}")
        return path

    def print_report(self) -> None:
        """Print text report to console."""
        print(self.generate_text_report())
