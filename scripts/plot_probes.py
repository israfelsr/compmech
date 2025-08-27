#!/usr/bin/env python3
"""
Simple script to plot probe results across layers.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re


def load_results(results_dir):
    """Load all probe result files."""
    results = {}

    # Look for probe result files
    pattern = str(Path(results_dir) / "probe_results_logistic_*.json")
    files = glob.glob(pattern)

    for file_path in files:
        filename = Path(file_path).name

        # Extract layer from filename
        if "last" in filename:
            layer = "last"
        else:
            # Try to find number in filename
            numbers = re.findall(r"\d+", filename)
            if numbers:
                layer = int(numbers[-1])  # Take the last number
            else:
                continue  # Skip if can't find layer

        with open(file_path, "r") as f:
            data = json.load(f)

        results[layer] = data
        print(f"Loaded {filename} -> layer {layer}")

    return results


def load_taxonomy(taxonomy_file):
    with open(taxonomy_file, "r") as f:
        return json.load(f)


def extract_layer_performance(results, metric="f1"):
    """Extract performance metrics for each layer."""
    layer_data = []

    for layer, data in results.items():
        individual_results = data["individual_results"]

        # Get all scores for this metric
        scores = [r[f"mean_{metric}"] for r in individual_results]

        layer_data.append(
            {
                "layer": layer,
                "mean": np.mean(scores),
                "std": np.std(scores),
                "median": np.median(scores),
                "min": np.min(scores),
                "max": np.max(scores),
                "n_attributes": len(scores),
            }
        )

    # Sort by layer (numeric first, then 'last')
    def sort_key(item):
        if item["layer"] == "last":
            return 999
        else:
            return item["layer"]

    layer_data.sort(key=sort_key)
    return layer_data


def attr_performance_distribution(results, layer, metric="f1", save_path=None):
    """
    Plot histogram/boxplot of F1 scores for all attributes within a single layer.
    Shows spread of performance - are most attributes learned equally well?
    """
    if layer not in results:
        print(f"Layer {layer} not found in results")
        return

    # Get all attribute scores for this layer
    individual_results = results[layer]["individual_results"]
    scores = [r[f"mean_{metric}"] for r in individual_results]
    attributes = [r["attribute"] for r in individual_results]

    # Print some insights
    print(f"\nLayer {layer} Performance Distribution:")
    print(f"- {len(scores)} attributes tested")
    print(f"- Mean {metric}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")
    print(f"- Range: {np.min(scores):.3f} to {np.max(scores):.3f}")

    # Identify best and worst performers
    sorted_results = sorted(zip(attributes, scores), key=lambda x: x[1], reverse=True)
    print(f"\nTop 5 attributes:")
    for attr, score in sorted_results[:5]:
        print(f"  {attr}: {score:.3f}")

    print(f"\nBottom 5 attributes:")
    for attr, score in sorted_results[-5:]:
        print(f"  {attr}: {score:.3f}")


def category_breakdown(results, layer, taxonomy, metric="f1", save_path=None):
    """
    Plot pie chart or stacked bar showing F1 performance by semantic category.
    Which types of concepts does this layer capture best?
    """
    if layer not in results:
        print(f"Layer {layer} not found in results")
        return

    # Get all attribute scores for this layer
    individual_results = results[layer]["individual_results"]

    # Group scores and baselines by category
    category_scores = {}
    category_baselines = {}
    category_counts = {}
    for result in individual_results:
        attr = result["attribute"]
        score = result[f"mean_{metric}"]
        baseline = result[f"baseline_mean_{metric}"]
        # Get category from taxonomy
        category = taxonomy.get(attr, "unknown")
        if category not in category_scores:
            category_scores[category] = []
            category_baselines[category] = []
            category_counts[category] = 0
        category_scores[category].append(score)
        category_baselines[category].append(baseline)
        category_counts[category] += 1
    # Calculate mean score and baseline per category
    category_means = {}
    category_stds = {}
    category_baseline_means = {}
    for cat, scores in category_scores.items():
        category_means[cat] = np.mean(scores)
        category_stds[cat] = np.std(scores)
        category_baseline_means[cat] = np.mean(category_baselines[cat])
    # Sort categories by performance
    sorted_categories = sorted(category_means.items(), key=lambda x: x[1], reverse=True)
    categories = [cat for cat, _ in sorted_categories]
    means = [score for _, score in sorted_categories]
    stds = [category_stds[cat] for cat, _ in sorted_categories]
    counts = [category_counts[cat] for cat, _ in sorted_categories]
    baselines = [category_baseline_means[cat] for cat, _ in sorted_categories]
    # Single bar chart
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    x_pos = range(len(categories))

    bars = plt.bar(
        x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black"
    )

    # Add red baseline markers for each category
    for i, baseline in enumerate(baselines):
        plt.plot(
            [i - 0.4, i + 0.4],
            [baseline, baseline],
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
        )

        # Add baseline percentage text in red for each category
        plt.text(
            i,
            baseline - 0.02,
            f"{baseline:.2f}%",
            color="red",
            fontweight="bold",
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", edgecolor="red", alpha=0.8
            ),
        )

    plt.xlabel("Category")
    plt.ylabel(f"Mean {metric.upper()} Score")
    plt.title(f"{metric.upper()} Performance by Category - Layer {layer}")
    plt.xticks(x_pos, categories, rotation=45, ha="right")
    plt.grid(True, alpha=0.3, axis="y")

    # Add performance value and count labels on bars
    for i, (bar, count, mean_val) in enumerate(zip(bars, counts, means)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + stds[i] + 0.01,
            f"{mean_val:.2f}%\nn={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    plt.show()
    # Print detailed breakdown
    print(f"\nCategory Breakdown for Layer {layer}:")
    print("=" * 50)
    for cat, mean_score in sorted_categories:
        std_score = category_stds[cat]
        baseline_score = category_baseline_means[cat]
        count = category_counts[cat]
        print(
            f"{cat:15s}: {mean_score:.3f} ± {std_score:.3f} (baseline: {baseline_score:.3f})"
        )
    return category_means, category_counts


def overview_performance(results, metric="f1", save_path=None):
    """
    Plot line plot with mean performance at each layer.
    Shows overall probe performance trend across layers.
    """
    # Extract performance data for all layers
    layer_data = extract_layer_performance(results, metric)

    if not layer_data:
        print("No layer data found!")
        return

    # Extract data for plotting
    layers = [d["layer"] for d in layer_data]
    means = [d["mean"] for d in layer_data]
    stds = [d["std"] for d in layer_data]
    n_attributes = [d["n_attributes"] for d in layer_data]

    # Create the plot
    plt.figure(figsize=(12, 6))
    x_pos = range(len(layers))

    # Main line plot
    plt.plot(
        x_pos,
        means,
        "o-",
        linewidth=3,
        markersize=8,
        color="steelblue",
        label="Mean Performance",
    )

    # Error bars (shaded area)
    plt.fill_between(
        x_pos,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.3,
        color="steelblue",
        label="± STD",
    )

    # Formatting
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(f"Mean {metric.upper()} Score", fontsize=12)
    plt.title(
        f"Probe Performance Overview - {metric.upper()} Across Layers",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set x-axis labels
    plt.xticks(x_pos, [str(l) for l in layers])

    # Add annotations for best and worst layers
    best_idx = np.argmax(means)
    worst_idx = np.argmin(means)

    plt.annotate(
        f"Best: Layer {layers[best_idx]}\n{means[best_idx]:.3f}",
        xy=(best_idx, means[best_idx]),
        xytext=(10, 15),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        fontsize=10,
    )

    plt.annotate(
        f"Worst: Layer {layers[worst_idx]}\n{means[worst_idx]:.3f}",
        xy=(worst_idx, means[worst_idx]),
        xytext=(10, -25),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        fontsize=10,
    )

    # Add text box with summary stats
    summary_text = f"""Summary:
    Layers: {len(layers)}
    Best: Layer {layers[best_idx]} ({means[best_idx]:.3f})
    Worst: Layer {layers[worst_idx]} ({means[worst_idx]:.3f})
    Range: {max(means) - min(means):.3f}
    Avg attributes/layer: {np.mean(n_attributes):.0f}"""

    plt.text(
        0.02,
        0.98,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()

    # Print summary to console
    print(f"\nOverview Performance Summary ({metric.upper()}):")
    print("=" * 40)
    for i, (layer, mean, std, n_attr) in enumerate(
        zip(layers, means, stds, n_attributes)
    ):
        status = ""
        if i == best_idx:
            status = " ⭐ BEST"
        elif i == worst_idx:
            status = " ⚠️  WORST"
        print(f"Layer {layer:>4}: {mean:.3f} ± {std:.3f} (n={n_attr:3d}){status}")

    print(
        f"\nPerformance range: {min(means):.3f} - {max(means):.3f} (Δ={max(means)-min(means):.3f})"
    )

    return layer_data


def attribute_breakdown(
    results, layer, taxonomy, metric="f1", save_path=None, top_n_per_category=10
):
    """
    Plot individual attributes as bars, color-coded by category.
    Shows both individual performance and category grouping in one visualization.
    Creates two plots: all attributes and top N per category.
    """
    if layer not in results:
        print(f"Layer {layer} not found in results")
        return

    # Get all attribute scores for this layer
    individual_results = results[layer]["individual_results"]

    # Create list of (attribute, score, baseline, category) tuples
    attr_data = []
    for result in individual_results:
        attr = result["attribute"]
        score = result[f"mean_{metric}"]
        baseline = result[f"baseline_mean_{metric}"]
        category = taxonomy.get(attr, "unknown")
        attr_data.append((attr, score, baseline, category))

    # Group by category and sort within each category by score descending
    category_groups = {}
    for attr, score, baseline, category in attr_data:
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append((attr, score, baseline))

    # Sort categories by their mean performance (best categories first)
    category_means = {
        cat: np.mean([score for _, score, _ in attrs])
        for cat, attrs in category_groups.items()
    }
    sorted_categories = sorted(category_means.items(), key=lambda x: x[1], reverse=True)
    category_order = [cat for cat, _ in sorted_categories]

    # Sort attributes within each category by score (best first)
    for cat in category_groups:
        category_groups[cat].sort(key=lambda x: x[1], reverse=True)

    # Get unique categories and assign colors
    unique_categories = category_order
    category_colors = dict(
        zip(unique_categories, plt.cm.Set3(np.linspace(0, 1, len(unique_categories))))
    )

    def create_plot(category_groups_plot, title_suffix, figsize=(18, 8)):
        """Helper function to create a plot with given category groups"""
        # Flatten data in category-grouped order with spacing
        attributes = []
        scores = []
        baselines = []
        categories = []
        x_positions = []  # Custom x positions with spacing
        category_positions = (
            {}
        )  # Track where each category starts for visual separators

        current_pos = 0
        bar_spacing = 2  # Space between bars (1.0 = no extra space, >1.0 = more space)

        for cat in category_order:
            if cat not in category_groups_plot:
                continue
            category_positions[cat] = current_pos
            for attr, score, baseline in category_groups_plot[cat]:
                attributes.append(attr)
                scores.append(score)
                baselines.append(baseline)
                categories.append(cat)
                x_positions.append(current_pos)
                current_pos += bar_spacing

        if not attributes:
            return None

        # Assign colors to each bar based on category
        bar_colors = [category_colors[cat] for cat in categories]

        # Create the plot
        fig = plt.figure(figsize=figsize)

        # Create bars with custom spacing
        bars = plt.bar(
            x_positions,
            scores,
            color=bar_colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            width=1.2,  # Bar width
        )

        # Add baseline markers
        for i, baseline in enumerate(baselines):
            x_pos = x_positions[i]
            plt.plot(
                [x_pos - 0.4, x_pos + 0.4],
                [baseline, baseline],
                color="red",
                linestyle="--",
                linewidth=1.5,
                alpha=0.8,
            )

        # Add vertical separators between categories
        y_max = max(scores) * 1.1
        current_cat = None
        for i, cat in enumerate(categories):
            if current_cat is not None and cat != current_cat:
                separator_pos = (x_positions[i - 1] + x_positions[i]) / 2
                plt.axvline(
                    x=separator_pos, color="gray", linestyle="-", linewidth=2, alpha=0.6
                )
            current_cat = cat

        # Add category labels at the top
        for cat in category_order:
            if cat not in category_positions:
                continue
            cat_attrs = [attr for attr, c in zip(attributes, categories) if c == cat]
            if not cat_attrs:
                continue

            cat_positions = [x for x, c in zip(x_positions, categories) if c == cat]
            start_x = min(cat_positions)
            end_x = max(cat_positions)
            center_pos = (start_x + end_x) / 2

            plt.text(
                center_pos,
                y_max * 0.95,
                cat,
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=10,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=category_colors[cat],
                    alpha=0.3,
                    edgecolor="black",
                ),
            )

        # Formatting
        plt.xlabel("Attributes (grouped by category)", fontsize=12)
        plt.ylabel(f"{metric.upper()} Score", fontsize=12)
        plt.title(
            f"Individual Attribute Performance (Layer {layer}) {title_suffix}",
            fontsize=14,
        )
        plt.grid(True, alpha=0.3, axis="y")

        # Adjust y-axis to accommodate category labels
        plt.ylim(0, y_max)

        # Rotate x-axis labels for readability with better spacing
        plt.xticks(x_positions, attributes, rotation=90, ha="right", fontsize=9)

        plt.margins(x=0.01, tight=True)
        plt.tight_layout()

        return fig

    # Create first plot - all attributes
    fig1 = create_plot(category_groups, "- All Attributes")

    if save_path:
        base_path = save_path.replace(".png", "")
        fig1.savefig(f"{base_path}_all.png", dpi=300, bbox_inches="tight")
        print(f"Saved all attributes plot to: {base_path}_all.png")

    plt.show()

    # Create second plot - top N per category
    category_groups_top = {}
    for cat in category_order:
        if cat in category_groups:
            # Take top N attributes per category
            category_groups_top[cat] = category_groups[cat][:top_n_per_category]

    fig2 = create_plot(category_groups_top, f"- Top {top_n_per_category} per Category")

    if save_path:
        base_path = save_path.replace(".png", "")
        fig2.savefig(
            f"{base_path}_top{top_n_per_category}.png", dpi=300, bbox_inches="tight"
        )
        print(
            f"Saved top {top_n_per_category} per category plot to: {base_path}_top{top_n_per_category}.png"
        )

    plt.show()

    # Print summary by category
    print(f"\nAttribute Performance Summary for Layer {layer}:")
    print("=" * 60)

    category_stats = {}
    for cat in unique_categories:
        cat_scores = [
            score for attr, score, _, category in attr_data if category == cat
        ]
        cat_baselines = [
            baseline for attr, _, baseline, category in attr_data if category == cat
        ]
        category_stats[cat] = {
            "count": len(cat_scores),
            "mean_score": np.mean(cat_scores),
            "std_score": np.std(cat_scores),
            "mean_baseline": np.mean(cat_baselines),
            "best_attr": max(
                [
                    (attr, score)
                    for attr, score, _, category in attr_data
                    if category == cat
                ],
                key=lambda x: x[1],
            ),
            "worst_attr": min(
                [
                    (attr, score)
                    for attr, score, _, category in attr_data
                    if category == cat
                ],
                key=lambda x: x[1],
            ),
        }

    # Sort categories by mean performance
    sorted_cats = sorted(
        category_stats.items(), key=lambda x: x[1]["mean_score"], reverse=True
    )

    for cat, stats in sorted_cats:
        print(f"\n{cat} (n={stats['count']}):")
        print(f"  Mean: {stats['mean_score']:.3f} ± {stats['std_score']:.3f}")
        print(f"  Baseline: {stats['mean_baseline']:.3f}")
        print(f"  Best: {stats['best_attr'][0]} ({stats['best_attr'][1]:.3f})")
        print(f"  Worst: {stats['worst_attr'][0]} ({stats['worst_attr'][1]:.3f})")

    print(f"\nTop {top_n_per_category} attributes per category:")
    for cat in category_order:
        if cat in category_groups_top and category_groups_top[cat]:
            print(f"\n{cat}:")
            for i, (attr, score, baseline) in enumerate(category_groups_top[cat], 1):
                print(f"  {i:2d}. {attr:<30}: {score:.3f} (baseline: {baseline:.3f})")

    return attr_data, category_stats


def performance_curves(results, taxonomy, categories=None, metric="f1", save_path=None):
    """
    Plot multiple line plots, one per category, across layers.
    Do colors emerge earlier than shapes? Animals vs. objects?
    """
    if not categories:
        # Get all unique categories from taxonomy
        categories = list(set(taxonomy.values()))
        categories.sort()

    # Initialize data structure for each category
    category_data = {cat: [] for cat in categories}

    # Sort layers properly
    sorted_layers = sorted(results.keys(), key=lambda x: 999 if x == "last" else x)

    # For each layer, calculate performance by category
    for layer in sorted_layers:
        if layer not in results:
            continue

        individual_results = results[layer]["individual_results"]

        # Group scores by category for this layer
        layer_category_scores = {cat: [] for cat in categories}

        for result in individual_results:
            attr = result["attribute"]
            score = result[f"mean_{metric}"]
            category = taxonomy.get(attr, "unknown")

            if category in layer_category_scores:
                layer_category_scores[category].append(score)

        # Calculate mean for each category in this layer
        for cat in categories:
            if layer_category_scores[cat]:  # If category has attributes in this layer
                mean_score = np.mean(layer_category_scores[cat])
                std_score = np.std(layer_category_scores[cat])
                n_attrs = len(layer_category_scores[cat])

                category_data[cat].append(
                    {
                        "layer": layer,
                        "mean": mean_score,
                        "std": std_score,
                        "n_attributes": n_attrs,
                    }
                )

    # Filter categories that have data across layers
    categories_with_data = []
    for cat in categories:
        if len(category_data[cat]) >= 3:  # At least 3 layers with data
            categories_with_data.append(cat)

    if not categories_with_data:
        print("No categories with sufficient data across layers!")
        return

    # Create the plot
    plt.figure(figsize=(14, 8))

    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories_with_data)))

    # Plot each category
    for i, cat in enumerate(categories_with_data):
        data = category_data[cat]
        layers = [d["layer"] for d in data]
        means = [d["mean"] for d in data]
        stds = [d["std"] for d in data]

        # Convert layer names to x positions
        x_positions = []
        for layer in layers:
            if layer == "last":
                x_positions.append(len(sorted_layers) - 1)
            else:
                x_positions.append(sorted_layers.index(layer))

        # Plot line with error bars
        plt.plot(
            x_positions,
            means,
            "o-",
            linewidth=2.5,
            markersize=6,
            color=colors[i],
            label=f"{cat}",
            alpha=0.8,
        )

        # Add error bars (optional - can be commented out if too cluttered)
        plt.fill_between(
            x_positions,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15,
            color=colors[i],
        )

    # Formatting
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel(f"Mean {metric.upper()} Score", fontsize=12)
    plt.title(
        f"Performance Curves by Category - {metric.upper()} Across Layers",
        fontsize=14,
    )
    plt.grid(True, alpha=0.3)

    # Set x-axis
    x_labels = [str(layer) for layer in sorted_layers]
    plt.xticks(range(len(sorted_layers)), x_labels)

    # Legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()

    # Print analysis
    print(f"\nPerformance Curves Analysis ({metric.upper()}):")
    print("=" * 50)

    # Find peak layer for each category
    category_peaks = {}
    for cat in categories_with_data:
        data = category_data[cat]
        if data:
            best_performance = max(data, key=lambda x: x["mean"])
            category_peaks[cat] = {
                "layer": best_performance["layer"],
                "score": best_performance["mean"],
                "n_attrs": best_performance["n_attributes"],
            }

    # Sort categories by their peak performance layer
    sorted_peaks = sorted(
        category_peaks.items(),
        key=lambda x: 999 if x[1]["layer"] == "last" else x[1]["layer"],
    )

    print("Categories ranked by peak performance layer:")
    for cat, peak_info in sorted_peaks:
        print(
            f"  {cat:15s}: Peak at layer {peak_info['layer']:>4} "
            f"({peak_info['score']:.3f}, n={peak_info['n_attrs']})"
        )

    # Category emergence analysis
    print(f"\nCategory Emergence Analysis:")
    print("(Which types of concepts emerge earlier vs. later?)")

    early_layers = [
        cat
        for cat, peak in sorted_peaks
        if (isinstance(peak["layer"], int) and peak["layer"] <= 6)
    ]
    late_layers = [
        cat
        for cat, peak in sorted_peaks
        if (isinstance(peak["layer"], int) and peak["layer"] > 6)
        or peak["layer"] == "last"
    ]

    if early_layers:
        print(f"Early emerging (layers 0-6): {', '.join(early_layers)}")
    if late_layers:
        print(f"Late emerging (layers 7+): {', '.join(late_layers)}")

    return category_data, category_peaks


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Plot probe results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Directory with probe results",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1"],
        choices=["f1", "accuracy", "precision", "recall"],
        help="Metrics to plot",
    )
    parser.add_argument("--taxonomy_file", type=str, help="Taxonomy for attributes")
    parser.add_argument("--save", type=str, help="Save plot to file")

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    if args.taxonomy_file:
        taxonomy = load_taxonomy(args.taxonomy_file)

    if not results:
        print("No results found!")
        return
    print(f"Found {len(results)} layers: {results.keys()}")

    if args.save:
        attr_performance_distribution(
            results,
            layer="last",
            metric="f1",
            save_path=f"{args.save}/attr_performance_distribution.png",
        )
        layer_data = overview_performance(results, save_path=f"{args.save}/overview_performance.png")
        
        # Find the best performing layer
        best_layer = None
        if layer_data:
            best_idx = max(range(len(layer_data)), key=lambda i: layer_data[i]['mean'])
            best_layer = layer_data[best_idx]['layer']
            print(f"Best performing layer: {best_layer} (F1: {layer_data[best_idx]['mean']:.3f})")
        
        if taxonomy:
            category_breakdown(
                results,
                "last",
                taxonomy,
                save_path=f"{args.save}/category_breakdown_last.png",
            )
            if best_layer is not None:
                category_breakdown(
                    results,
                    best_layer,
                    taxonomy,
                    save_path=f"{args.save}/category_breakdown_best.png",
                )
                attribute_breakdown(
                    results,
                    best_layer,
                    taxonomy,
                    save_path=f"{args.save}/attribute_breakdown_best.png",
                )
            performance_curves(
                results,
                taxonomy,
                save_path=f"{args.save}/performance_curves.png",
            )


if __name__ == "__main__":
    main()
