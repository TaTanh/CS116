"""Command-line interface for the recommender system."""

from datetime import datetime
from pathlib import Path

import click
import polars as pl

from src.recommender.candidates import generate_candidates
from src.recommender.data import load_items, load_transactions, load_users
from src.recommender.features import build_feature_label_table
from src.recommender.metrics import evaluate_recommendations
from src.recommender.train import (
    load_model,
    predict_and_rank,
    save_model,
    train_model,
    get_feature_importance,
)


@click.group()
def cli():
    """Time-based purchase prediction recommender system."""
    pass


@cli.command()
@click.option(
    "--transactions-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to transactions parquet file",
)
@click.option(
    "--items-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to items parquet file",
)
@click.option(
    "--users-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to users parquet file",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path to save feature table",
)
@click.option(
    "--observation-date",
    type=str,
    default=None,
    help="Observation date (YYYY-MM-DD), defaults to today",
)
@click.option(
    "--lookback-days",
    type=int,
    default=90,
    help="Number of days to look back for features",
)
@click.option(
    "--prediction-days",
    type=int,
    default=30,
    help="Number of days forward for labels",
)
@click.option(
    "--candidate-strategy",
    type=click.Choice(["user_history", "popular_items", "category_based", "hybrid"]),
    default="hybrid",
    help="Candidate generation strategy",
)
def build_features(
    transactions_path: str,
    items_path: str,
    users_path: str,
    output_path: str,
    observation_date: str,
    lookback_days: int,
    prediction_days: int,
    candidate_strategy: str,
):
    """Build feature and label table for training."""
    click.echo("Building features...")
    
    # Parse observation date
    obs_date = (
        datetime.strptime(observation_date, "%Y-%m-%d")
        if observation_date
        else datetime.now()
    )
    click.echo(f"Observation date: {obs_date}")
    
    # Load data
    click.echo("Loading data...")
    transactions = load_transactions(transactions_path)
    items = load_items(items_path)
    users = load_users(users_path)
    
    # Generate candidates
    click.echo(f"Generating candidates using '{candidate_strategy}' strategy...")
    candidates = generate_candidates(
        transactions, items, users, obs_date, strategy=candidate_strategy
    )
    
    # Build features
    click.echo("Building feature table...")
    features = build_feature_label_table(
        transactions,
        items,
        users,
        obs_date,
        lookback_days=lookback_days,
        prediction_days=prediction_days,
    )
    
    # Save to parquet
    click.echo(f"Saving features to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.collect().write_parquet(output_path)
    
    click.echo("✓ Features built successfully!")


@cli.command()
@click.option(
    "--features-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to features parquet file",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path to save trained model",
)
@click.option(
    "--validation-split",
    type=float,
    default=0.2,
    help="Fraction of data for validation",
)
@click.option(
    "--show-importance",
    is_flag=True,
    help="Display feature importance after training",
)
def train(
    features_path: str,
    output_path: str,
    validation_split: float,
    show_importance: bool,
):
    """Train a purchase prediction model."""
    click.echo("Training model...")
    
    # Load features
    click.echo(f"Loading features from {features_path}...")
    features_df = pl.scan_parquet(features_path)
    
    # Define feature columns (exclude ID columns and label)
    all_columns = features_df.columns
    exclude_cols = {"customer_id", "item_id", "label", "order_id"}
    feature_columns = [col for col in all_columns if col not in exclude_cols]
    
    click.echo(f"Using {len(feature_columns)} features for training")
    
    # Train model
    model = train_model(
        features_df,
        feature_columns,
        validation_split=validation_split,
    )
    
    # Save model
    save_model(model, output_path)
    
    # Show feature importance
    if show_importance:
        click.echo("\nTop 10 most important features:")
        importance_df = get_feature_importance(model, feature_columns)
        for row in importance_df.head(10).iter_rows(named=True):
            click.echo(f"  {row['feature']}: {row['importance']:.2f}")
    
    click.echo("✓ Model trained successfully!")


@cli.command()
@click.option(
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model",
)
@click.option(
    "--features-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to features parquet file",
)
@click.option(
    "--output-path",
    type=click.Path(),
    required=True,
    help="Path to save predictions",
)
@click.option(
    "--top-k",
    type=int,
    default=20,
    help="Number of top recommendations per user",
)
def predict(
    model_path: str,
    features_path: str,
    output_path: str,
    top_k: int,
):
    """Generate predictions using trained model."""
    click.echo("Generating predictions...")
    
    # Load model
    model = load_model(model_path)
    
    # Load features
    click.echo(f"Loading features from {features_path}...")
    features_df = pl.scan_parquet(features_path)
    
    # Define feature columns
    all_columns = features_df.columns
    exclude_cols = {"customer_id", "item_id", "label", "order_id"}
    feature_columns = [col for col in all_columns if col not in exclude_cols]
    
    # Predict and rank
    predictions = predict_and_rank(
        model,
        features_df,
        feature_columns,
        top_k=top_k,
    )
    
    # Save predictions
    click.echo(f"Saving predictions to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output_path)
    
    click.echo(f"✓ Predictions generated for {predictions.height} user-item pairs!")


@cli.command()
@click.option(
    "--predictions-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to predictions parquet file",
)
@click.option(
    "--ground-truth-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to ground truth parquet file",
)
@click.option(
    "--k-values",
    type=str,
    default="5,10,20",
    help="Comma-separated K values for evaluation",
)
def evaluate(
    predictions_path: str,
    ground_truth_path: str,
    k_values: str,
):
    """Evaluate recommendation quality."""
    click.echo("Evaluating recommendations...")
    
    # Parse K values
    k_list = [int(k.strip()) for k in k_values.split(",")]
    
    # Load data
    predictions = pl.read_parquet(predictions_path)
    ground_truth = pl.read_parquet(ground_truth_path)
    
    # Evaluate
    results = evaluate_recommendations(predictions, ground_truth, k_values=k_list)
    
    # Display results
    click.echo("\nEvaluation Results:")
    click.echo("=" * 50)
    for metric_name, metric_values in results.items():
        click.echo(f"\n{metric_name.upper()}:")
        for k, value in metric_values.items():
            click.echo(f"  @{k}: {value:.4f}")
    
    click.echo("\n✓ Evaluation complete!")


if __name__ == "__main__":
    cli()
