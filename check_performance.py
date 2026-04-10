#!/usr/bin/env python3
"""Daily performance summary for paper trading."""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "kalshi_bot.db"


def get_performance_summary():
    """Get paper trading performance metrics."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\n" + "=" * 60)
    print(f"  BOT PERFORMANCE - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Get trades with settlement outcomes to calculate P&L
    cur = conn.execute("""
        SELECT
            t.trade_id,
            t.market_ticker,
            t.side,
            t.price,
            t.quantity,
            t.total_cost,
            t.fee,
            t.executed_at,
            s.outcome,
            CASE
                WHEN s.outcome IS NULL THEN NULL
                WHEN (t.side = 'yes' AND s.outcome = 'yes') OR (t.side = 'no' AND s.outcome = 'no')
                THEN (t.quantity * 1.0) - t.total_cost
                ELSE -t.total_cost
            END as pnl
        FROM trades t
        LEFT JOIN market_settlements s ON t.market_ticker = s.ticker
        ORDER BY t.executed_at DESC
    """)
    trades = cur.fetchall()

    if not trades:
        print("\n  No trades yet. Bot is running and will execute")
        print("  trades when opportunities arise.")
        check_ml_status(conn)
        print("\n" + "=" * 60)
        conn.close()
        return

    # Calculate stats
    settled = [t for t in trades if t['pnl'] is not None]
    pending = [t for t in trades if t['pnl'] is None]
    wins = [t for t in settled if t['pnl'] > 0]
    losses = [t for t in settled if t['pnl'] < 0]

    total_pnl = sum(t['pnl'] for t in settled)
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.01
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    win_rate = (len(wins) / len(settled) * 100) if settled else 0

    print(f"""
  TRADE STATS
  -----------
  Total Trades:     {len(trades)}
  Settled:          {len(settled)}
  Pending:          {len(pending)}

  Wins / Losses:    {len(wins)} / {len(losses)}
  Win Rate:         {win_rate:.1f}%
  Total P&L:        ${total_pnl:.2f}
  Profit Factor:    {profit_factor:.2f}
""")

    if wins:
        print(f"  Best Trade:       ${max(t['pnl'] for t in wins):.2f}")
    if losses:
        print(f"  Worst Trade:      ${min(t['pnl'] for t in losses):.2f}")

    # Recent settled trades
    recent_settled = [t for t in settled][:5]
    if recent_settled:
        print("\n  RECENT SETTLED TRADES")
        print("  ---------------------")
        for t in recent_settled:
            pnl = t['pnl']
            emoji = "W" if pnl > 0 else "L"
            print(f"  [{emoji}] {t['market_ticker'][:20]:<20} {t['side']:>3} ${pnl:>+7.2f}")

    # ML Predictions performance
    check_ml_status(conn)

    # GO LIVE recommendation
    print("\n  " + "-" * 40)
    if len(settled) >= 50 and win_rate >= 55 and profit_factor >= 1.3:
        print("  READY TO GO LIVE")
        print(f"    {len(settled)} settled trades, {win_rate:.0f}% WR, {profit_factor:.1f} PF")
    else:
        missing = []
        if len(settled) < 50:
            missing.append(f"need {50-len(settled)} more settled trades")
        if win_rate < 55 and len(settled) > 0:
            missing.append(f"win rate {win_rate:.0f}% < 55%")
        if profit_factor < 1.3 and len(settled) > 0:
            missing.append(f"profit factor {profit_factor:.1f} < 1.3")
        print("  NOT READY YET - " + ", ".join(missing))

    print("\n" + "=" * 60 + "\n")
    conn.close()


def check_ml_status(conn):
    """Check ML prediction performance."""
    cur = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
            SUM(CASE WHEN actual_outcome IS NOT NULL THEN 1 ELSE 0 END) as settled
        FROM ml_predictions
    """)
    row = cur.fetchone()

    if row and row['total'] > 0:
        print(f"""
  ML PREDICTIONS
  --------------
  Total Made:       {row['total']}
  Settled:          {row['settled']}
  Correct:          {row['correct'] or 0}
  Accuracy:         {(row['correct'] / row['settled'] * 100) if row['settled'] else 0:.1f}%
""")
    else:
        print("""
  ML PREDICTIONS
  --------------
  No predictions yet - ML just enabled, will start predicting soon
""")


if __name__ == "__main__":
    get_performance_summary()
