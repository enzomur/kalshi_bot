"""Entry point for the Kalshi Arbitrage Bot."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from typing import Any

from dotenv import load_dotenv

# Load .env file before any other imports that might read environment variables
load_dotenv()

from kalshi_bot.bot import KalshiArbitrageBot
from kalshi_bot.config.settings import Settings, get_settings
from kalshi_bot.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


class BotRunner:
    """Handles bot lifecycle and signal handling."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bot: KalshiArbitrageBot | None = None
        self._shutdown_requested = False

    async def run(self) -> int:
        """Run the bot with proper lifecycle management."""
        self.bot = KalshiArbitrageBot(self.settings)

        loop = asyncio.get_running_loop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s)),
            )

        try:
            await self.bot.initialize()

            if self.settings.dashboard.enabled:
                dashboard_task = asyncio.create_task(self._run_dashboard())
            else:
                dashboard_task = None

            bot_task = asyncio.create_task(self.bot.start())

            tasks = [bot_task]
            if dashboard_task:
                tasks.append(dashboard_task)

            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            return 0

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            return 1

        finally:
            if self.bot:
                await self.bot.shutdown()

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        if self._shutdown_requested:
            logger.warning("Forced shutdown requested")
            sys.exit(1)

        self._shutdown_requested = True
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown...")

        if self.bot:
            await self.bot.stop()

    async def _run_dashboard(self) -> None:
        """Run the FastAPI dashboard."""
        try:
            from kalshi_bot.dashboard.app import create_app
            import uvicorn

            app = create_app(self.bot)
            config = uvicorn.Config(
                app,
                host=self.settings.dashboard.host,
                port=self.settings.dashboard.port,
                log_level="warning",
            )
            server = uvicorn.Server(config)

            logger.info(
                f"Dashboard starting at http://{self.settings.dashboard.host}:{self.settings.dashboard.port}"
            )

            await server.serve()

        except ImportError:
            logger.warning("Dashboard dependencies not available, skipping")
        except Exception as e:
            logger.error(f"Dashboard error: {e}")


def validate_configuration(settings: Settings) -> bool:
    """Validate configuration before starting."""
    errors: list[str] = []

    if not settings.api_key_id:
        errors.append("KALSHI_API_KEY_ID is required")

    if not settings.private_key_path:
        errors.append("KALSHI_PRIVATE_KEY_PATH is required")

    if errors:
        logger.error("Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False

    return True


async def backfill_history(settings: Settings, days: int, include_candlesticks: bool, include_historical: bool) -> int:
    """Run historical data backfill for ML training."""
    from kalshi_bot.api.client import KalshiAPIClient
    from kalshi_bot.persistence.database import Database
    from kalshi_bot.ml.historical_backfill import HistoricalBackfiller

    logger.info("Starting historical data backfill...")

    db = Database(settings.database_path)
    api_client = KalshiAPIClient(settings)

    try:
        await db.initialize()
        await api_client.connect()

        backfiller = HistoricalBackfiller(db, api_client)

        # Show current status
        status = await backfiller.get_backfill_status()
        print(f"\nCurrent ML Data Status:")
        print(f"  Settlements: {status['total_settlements']}")
        print(f"  Snapshots: {status['total_snapshots']}")
        print(f"  Unique markets with snapshots: {status['unique_tickers_with_snapshots']}")
        print(f"  Settlements with sufficient data: {status['settlements_with_sufficient_data']}")
        print()

        # Run backfill
        result = await backfiller.run_full_backfill(
            days_back=days,
            include_candlesticks=include_candlesticks,
            include_historical=include_historical,
        )

        # Show results
        print(f"\nBackfill Results:")
        print(f"  Settled markets added: {result.settled_markets_added}")
        print(f"  Snapshots added: {result.snapshots_added}")
        print(f"  Duration: {result.duration_seconds:.1f} seconds")
        print(f"  Success: {'Yes' if result.success else 'No'}")

        if result.errors:
            print(f"\nErrors encountered:")
            for error in result.errors:
                print(f"  - {error}")

        # Show updated status
        status = await backfiller.get_backfill_status()
        print(f"\nUpdated ML Data Status:")
        print(f"  Total settlements: {status['total_settlements']}")
        print(f"  Total snapshots: {status['total_snapshots']}")
        print(f"  Outcome distribution: {status['outcome_distribution']}")

        # Check if ready for training
        min_settlements = settings.ml.min_settlements_for_training
        if status['total_settlements'] >= min_settlements:
            print(f"\nReady for ML training! (>= {min_settlements} settlements)")
            print("Run: python -m kalshi_bot.main --train-model")
        else:
            needed = min_settlements - status['total_settlements']
            print(f"\nNeed {needed} more settlements before ML training")
            print("Continue running the bot to collect more data, or backfill more history.")

        return 0 if result.success else 1

    except Exception as e:
        logger.error(f"Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await api_client.close()
        await db.close()


async def train_ml_model(settings: Settings, model_type: str) -> int:
    """Train an ML model."""
    from kalshi_bot.persistence.database import Database
    from kalshi_bot.ml.training.trainer import ModelTrainer

    logger.info(f"Training {model_type} model...")

    db = Database(settings.database_path)
    try:
        await db.initialize()

        trainer = ModelTrainer(db)

        # Check training status
        status = await trainer.get_training_status()
        print(f"\nML Training Status:")
        print(f"  Settlements available: {status['settlement_count']}")
        print(f"  Minimum required: {status['min_settlements_required']}")
        print(f"  Ready for training: {status['ready_for_training']}")

        if not status['ready_for_training']:
            print(f"\nNot enough data for training. Need {status['min_settlements_required']} settlements.")
            print("Run the bot to collect more market data.")
            return 1

        # Train model
        print(f"\nTraining {model_type} model...")
        result = await trainer.train_model(model_type, run_cv=True)

        if not result.success:
            print(f"\nTraining failed: {result.error_message}")
            return 1

        print(f"\nTraining successful!")
        print(f"  Model ID: {result.model_id}")
        print(f"  Training samples: {result.training_samples}")
        print(f"  CV Accuracy: {result.cv_accuracy:.3f} (+/- {result.cv_std:.3f})")
        if result.metrics:
            print(f"  Final Accuracy: {result.metrics.accuracy:.3f}")
            print(f"  AUC: {result.metrics.auc:.3f}")
            print(f"  Brier Score: {result.metrics.brier_score:.3f}")
        print(f"  Model saved to: {result.model_path}")

        # Compare with current model
        comparison = await trainer.compare_models(result.model_id)
        print(f"\nComparison: {comparison.reason}")

        if comparison.should_replace:
            print("Activating new model...")
            await trainer.activate_model(result.model_id)
            print("New model activated!")
        else:
            print("Keeping current model.")

        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

    finally:
        await db.close()


async def show_ml_status(settings: Settings) -> int:
    """Show ML system status."""
    from kalshi_bot.persistence.database import Database
    from kalshi_bot.ml.training.trainer import ModelTrainer

    db = Database(settings.database_path)
    try:
        await db.initialize()

        trainer = ModelTrainer(db)
        status = await trainer.get_training_status()

        print("\n" + "=" * 60)
        print("ML System Status")
        print("=" * 60)

        print(f"\nData Collection:")
        print(f"  Snapshots collected: {status['snapshot_count']}")
        print(f"  Markets settled: {status['settlement_count']}")
        print(f"  Min settlements for training: {status['min_settlements_required']}")
        print(f"  Ready for training: {'Yes' if status['ready_for_training'] else 'No'}")

        print(f"\nActive Models:")
        if status['active_models']:
            for model in status['active_models']:
                print(f"  - {model['model_id']} ({model['model_type']})")
                if model['metrics']:
                    print(f"    Accuracy: {model['metrics'].get('accuracy', 'N/A'):.3f}")
                print(f"    Trained: {model['trained_at']}")
        else:
            print("  No active models")

        print(f"\nRecent Training:")
        if status['recent_training']:
            for model in status['recent_training'][:3]:
                print(f"  - {model['model_id']}: {model['status']}")
        else:
            print("  No training history")

        # Get prediction accuracy if available
        predictions = await db.fetch_one(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
            FROM ml_predictions
            WHERE actual_outcome IS NOT NULL
            """
        )

        if predictions and predictions['total'] > 0:
            accuracy = predictions['correct'] / predictions['total']
            print(f"\nPrediction Performance:")
            print(f"  Total predictions (settled): {predictions['total']}")
            print(f"  Correct predictions: {predictions['correct']}")
            print(f"  Accuracy: {accuracy:.1%}")

        print("\n" + "=" * 60)
        return 0

    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return 1

    finally:
        await db.close()


async def run_backtest(
    settings: Settings,
    initial_balance: float,
    start_date: str | None,
    end_date: str | None,
    output_path: str | None,
    verbose: bool,
) -> int:
    """Run a backtest on historical data."""
    from datetime import datetime
    from kalshi_bot.persistence.database import Database
    from kalshi_bot.backtesting.engine import BacktestEngine, BacktestConfig
    from kalshi_bot.backtesting.report import ReportGenerator

    logger.info("Starting backtest...")

    db = Database(settings.database_path)
    try:
        await db.initialize()

        engine = BacktestEngine(db, settings)

        # Get data summary first
        summary = await engine.get_data_summary()
        print(f"\nHistorical Data Summary:")
        print(f"  Total snapshots: {summary.get('total_snapshots', 0)}")
        print(f"  Unique markets: {summary.get('unique_tickers', 0)}")
        print(f"  Date range: {summary.get('earliest_date', 'N/A')} to {summary.get('latest_date', 'N/A')}")
        print(f"  Total settlements: {summary.get('total_settlements', 0)}")

        if summary.get('total_snapshots', 0) == 0:
            print("\nNo historical data available. Run the bot to collect data first.")
            return 1

        # Parse dates
        config_start = None
        config_end = None
        if start_date:
            config_start = datetime.fromisoformat(start_date)
        if end_date:
            config_end = datetime.fromisoformat(end_date)

        # Create config
        config = BacktestConfig(
            start_date=config_start,
            end_date=config_end,
            initial_balance=initial_balance,
            verbose=verbose,
        )

        print(f"\nRunning backtest with ${initial_balance:,.2f} initial balance...")

        # Progress callback
        def on_progress(progress):
            if verbose and progress.current_step % 50 == 0:
                print(f"  Step {progress.current_step}/{progress.total_steps}")

        # Run backtest
        result = await engine.run(config, progress_callback=on_progress)

        if not result.success:
            print(f"\nBacktest failed: {result.error_message}")
            return 1

        # Generate and display report
        reporter = ReportGenerator(result)
        reporter.print_report()

        # Save report if output path specified
        if output_path:
            reporter.save_report(output_path)
            print(f"\nReport saved to: {output_path}")

        return 0

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await db.close()


async def export_opportunities(settings: Settings, output_path: str, max_markets: int) -> int:
    """Export market opportunities for Claude Code analysis."""
    from kalshi_bot.analysis.opportunity_exporter import OpportunityExporter
    from kalshi_bot.api.client import KalshiAPIClient

    logger.info("Exporting market opportunities for analysis...")

    api_client = KalshiAPIClient(settings)
    try:
        await api_client.connect()

        # Fetch markets
        logger.info("Fetching markets from Kalshi...")
        markets = await api_client.get_all_markets(
            status="open",
            min_volume=0,
            min_open_interest=0,
            require_orderbook=False,
            max_pages=100,
        )
        logger.info(f"Fetched {len(markets)} markets")

        # Export with relaxed filters to find candidates
        exporter = OpportunityExporter(
            min_volume=50,  # Lower volume threshold
            max_spread_cents=15,  # Allow wider spreads
            max_days_to_expiry=90,  # 3 months instead of 1
            price_range=(10, 90),  # Wider price range
        )

        output_file = exporter.export_to_file(markets, output_path, max_markets)
        logger.info(f"Opportunities exported to: {output_file}")

        # Print summary
        candidates = exporter.filter_candidates(markets)
        print(f"\nExported {min(len(candidates), max_markets)} market opportunities to {output_path}")
        print(f"Total candidates matching filters: {len(candidates)}")
        print(f"\nShare this file with Claude Code for analysis.")

        return 0

    except Exception as e:
        logger.error(f"Failed to export opportunities: {e}")
        return 1

    finally:
        await api_client.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Kalshi Arbitrage Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--export-opportunities",
        action="store_true",
        help="Export market opportunities for Claude Code analysis instead of running the bot",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="opportunities.md",
        help="Output file path for --export-opportunities (default: opportunities.md)",
    )

    parser.add_argument(
        "--max-markets",
        type=int,
        default=20,
        help="Maximum number of markets to export (default: 20)",
    )

    parser.add_argument(
        "--paper-trading",
        action="store_true",
        help="Force paper trading mode (overrides settings)",
    )

    # ML commands
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train ML model instead of running the bot",
    )

    parser.add_argument(
        "--model-type",
        choices=["logistic", "gradient_boost"],
        default="logistic",
        help="Model type to train (default: logistic)",
    )

    parser.add_argument(
        "--ml-status",
        action="store_true",
        help="Show ML system status and exit",
    )

    # Historical data backfill
    parser.add_argument(
        "--backfill-history",
        action="store_true",
        help="Backfill historical data for ML training",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of history to backfill (default: 90)",
    )

    parser.add_argument(
        "--no-candlesticks",
        action="store_true",
        help="Skip candlestick backfill (only backfill settlements)",
    )

    parser.add_argument(
        "--include-historical",
        action="store_true",
        help="Include archived/historical data (>3 months old)",
    )

    # Backtesting
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run a backtest on historical data",
    )

    parser.add_argument(
        "--backtest-balance",
        type=float,
        default=10000.0,
        help="Initial balance for backtest (default: 10000)",
    )

    parser.add_argument(
        "--backtest-start",
        type=str,
        default=None,
        help="Backtest start date (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--backtest-end",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD format)",
    )

    parser.add_argument(
        "--backtest-output",
        type=str,
        default=None,
        help="Output file for backtest report (supports .txt, .json, .md, .html)",
    )

    parser.add_argument(
        "--backtest-verbose",
        action="store_true",
        help="Enable verbose backtest output",
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    try:
        settings = get_settings()
    except Exception as e:
        print(f"Failed to load settings: {e}")
        return 1

    setup_logging(settings)

    # Handle export-opportunities command
    if args.export_opportunities:
        return asyncio.run(export_opportunities(settings, args.output, args.max_markets))

    # Handle ML commands
    if args.train_model:
        return asyncio.run(train_ml_model(settings, args.model_type))

    if args.ml_status:
        return asyncio.run(show_ml_status(settings))

    # Handle backfill command
    if args.backfill_history:
        return asyncio.run(backfill_history(
            settings,
            days=args.days,
            include_candlesticks=not args.no_candlesticks,
            include_historical=args.include_historical,
        ))

    # Handle backtest command
    if args.backtest:
        return asyncio.run(run_backtest(
            settings,
            initial_balance=args.backtest_balance,
            start_date=args.backtest_start,
            end_date=args.backtest_end,
            output_path=args.backtest_output,
            verbose=args.backtest_verbose,
        ))

    logger.info("=" * 60)
    logger.info("Kalshi Arbitrage Trading Bot")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.environment.value}")
    logger.info(f"API Base URL: {settings.api_base_url}")

    # Show paper trading mode prominently
    paper_mode = getattr(settings.trading, 'paper_trading_mode', True)
    if paper_mode:
        logger.info("")
        logger.info("*** PAPER TRADING MODE ENABLED ***")
        logger.info("*** No real trades will be executed ***")
        logger.info("")
    else:
        logger.warning("")
        logger.warning("!!! LIVE TRADING MODE - REAL MONEY !!!")
        logger.warning("")

    logger.info("=" * 60)

    if not validate_configuration(settings):
        return 1

    runner = BotRunner(settings)

    try:
        return asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 0


if __name__ == "__main__":
    sys.exit(main())
