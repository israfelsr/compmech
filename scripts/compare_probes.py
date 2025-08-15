#!/usr/bin/env python3
"""
Script to compare probe results between two different experiments.
Compares performance across all layers and specific layers.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import re
import argparse


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
        elif "proj" in filename:
            layer = "proj"
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

    # Sort by layer (numeric first, then 'proj', then 'last')
    def sort_key(item):
        if item["layer"] == "last":
            return 9999
        elif item["layer"] == "proj":
            return 9998
        else:
            return item["layer"]

    layer_data.sort(key=sort_key)
    return layer_data


def compare_overview_performance(
    results1, results2, labels, metric="f1", save_path=None
):
    """Compare performance overview between two experiments."""
    layer_data1 = extract_layer_performance(results1, metric)
    layer_data2 = extract_layer_performance(results2, metric)

    if not layer_data1 or not layer_data2:
        print("No layer data found!")
        return

    # Create aligned layer data
    layers1 = [d["layer"] for d in layer_data1]
    layers2 = [d["layer"] for d in layer_data2]
    common_layers = set(layers1) & set(layers2)

    if not common_layers:
        print("No common layers found between experiments!")
        return

    # Filter to common layers and sort
    def sort_key(layer):
        if layer == "last":
            return 9999
        elif layer == "proj":
            return 9998
        else:
            return layer

    common_layers = sorted(common_layers, key=sort_key)

    # Extract data for common layers
    means1 = []
    means2 = []
    stds1 = []
    stds2 = []

    for layer in common_layers:
        data1 = next(d for d in layer_data1 if d["layer"] == layer)
        data2 = next(d for d in layer_data2 if d["layer"] == layer)
        means1.append(data1["mean"])
        means2.append(data2["mean"])
        stds1.append(data1["std"])
        stds2.append(data2["std"])

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    x_pos = range(len(common_layers))

    # Top plot: Side-by-side comparison
    ax1.plot(
        x_pos,
        means1,
        "o-",
        linewidth=3,
        markersize=8,
        color="steelblue",
        label=labels[0],
        alpha=0.8,
    )
    ax1.plot(
        x_pos,
        means2,
        "s-",
        linewidth=3,
        markersize=8,
        color="coral",
        label=labels[1],
        alpha=0.8,
    )

    # Add error bars
    ax1.fill_between(
        x_pos,
        [m - s for m, s in zip(means1, stds1)],
        [m + s for m, s in zip(means1, stds1)],
        alpha=0.2,
        color="steelblue",
    )
    ax1.fill_between(
        x_pos,
        [m - s for m, s in zip(means2, stds2)],
        [m + s for m, s in zip(means2, stds2)],
        alpha=0.2,
        color="coral",
    )

    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel(f"Mean {metric.upper()} Score", fontsize=12)
    ax1.set_title(f"Performance Comparison: {labels[0]} vs {labels[1]}", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(l) for l in common_layers])

    # Bottom plot: Difference plot
    differences = [m2 - m1 for m1, m2 in zip(means1, means2)]
    colors = ["green" if d > 0 else "red" for d in differences]

    bars = ax2.bar(x_pos, differences, color=colors, alpha=0.7, edgecolor="black")
    ax2.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel(
        f"Difference in {metric.upper()} ({labels[1]} - {labels[0]})", fontsize=12
    )
    ax2.set_title(f"Performance Difference by Layer", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(l) for l in common_layers])

    # Add difference values on bars
    for i, (bar, diff) in enumerate(zip(bars, differences)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.005 if height > 0 else -0.005),
            f"{diff:+.3f}",
            ha="center",
            va="bottom" if height > 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()

    # Print summary
    print(f"\nComparison Summary ({metric.upper()}):")
    print("=" * 60)
    print(f"{'Layer':<8} {'Exp1':<10} {'Exp2':<10} {'Difference':<12} {'Winner':<10}")
    print("-" * 60)

    for i, layer in enumerate(common_layers):
        diff = differences[i]
        winner = labels[1] if diff > 0 else labels[0]
        if abs(diff) < 0.001:
            winner = "Tie"

        print(
            f"{layer:<8} {means1[i]:<10.3f} {means2[i]:<10.3f} {diff:<+12.3f} {winner:<10}"
        )

    avg_diff = np.mean(differences)
    print("-" * 60)
    print(
        f"{'Average':<8} {np.mean(means1):<10.3f} {np.mean(means2):<10.3f} {avg_diff:<+12.3f} "
        f"{labels[1] if avg_diff > 0 else labels[0]:<10}"
    )

    return layer_data1, layer_data2, differences


def compare_specific_layers(
    results1, results2, labels, layers_to_compare, metric="f1", save_path=None
):
    """Compare performance for specific layers in detail."""
    fig, axes = plt.subplots(
        1, len(layers_to_compare), figsize=(6 * len(layers_to_compare), 6)
    )
    if len(layers_to_compare) == 1:
        axes = [axes]

    comparison_data = {}

    for idx, layer in enumerate(layers_to_compare):
        if layer not in results1 or layer not in results2:
            print(f"Layer {layer} not found in one or both experiments")
            continue

        # Get individual attribute scores
        scores1 = [r[f"mean_{metric}"] for r in results1[layer]["individual_results"]]
        scores2 = [r[f"mean_{metric}"] for r in results2[layer]["individual_results"]]
        attrs1 = [r["attribute"] for r in results1[layer]["individual_results"]]
        attrs2 = [r["attribute"] for r in results2[layer]["individual_results"]]

        # Find common attributes
        common_attrs = set(attrs1) & set(attrs2)
        if not common_attrs:
            print(f"No common attributes found for layer {layer}")
            continue

        # Create aligned data for common attributes
        aligned_scores1 = []
        aligned_scores2 = []
        aligned_attrs = []

        for attr in common_attrs:
            idx1 = attrs1.index(attr)
            idx2 = attrs2.index(attr)
            aligned_scores1.append(scores1[idx1])
            aligned_scores2.append(scores2[idx2])
            aligned_attrs.append(attr)

        # Sort by difference (biggest improvements first)
        differences = [s2 - s1 for s1, s2 in zip(aligned_scores1, aligned_scores2)]
        sorted_data = sorted(
            zip(aligned_attrs, aligned_scores1, aligned_scores2, differences),
            key=lambda x: x[3],
            reverse=True,
        )

        # Extract sorted data
        sorted_attrs, sorted_scores1, sorted_scores2, sorted_diffs = zip(*sorted_data)

        # Scatter plot
        ax = axes[idx]
        ax.scatter(sorted_scores1, sorted_scores2, alpha=0.6, s=50)

        # Perfect correlation line
        min_val = min(min(sorted_scores1), min(sorted_scores2))
        max_val = max(max(sorted_scores1), max(sorted_scores2))
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            alpha=0.8,
            label="Perfect correlation",
        )

        ax.set_xlabel(f"{labels[0]} {metric.upper()} Score", fontsize=11)
        ax.set_ylabel(f"{labels[1]} {metric.upper()} Score", fontsize=11)
        ax.set_title(
            f"Layer {layer} Comparison\n(n={len(common_attrs)} attributes)", fontsize=12
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Add correlation coefficient
        correlation = np.corrcoef(sorted_scores1, sorted_scores2)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {correlation:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        # Store comparison data
        comparison_data[layer] = {
            "attributes": sorted_attrs,
            "scores1": sorted_scores1,
            "scores2": sorted_scores2,
            "differences": sorted_diffs,
            "correlation": correlation,
            "mean_diff": np.mean(sorted_diffs),
        }

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    plt.show()

    # Print detailed comparison
    for layer in layers_to_compare:
        if layer in comparison_data:
            data = comparison_data[layer]
            print(f"\nLayer {layer} Detailed Comparison:")
            print("=" * 50)
            print(f"Correlation: {data['correlation']:.3f}")
            print(f"Mean difference: {data['mean_diff']:+.3f}")
            print(f"\nTop 5 improvements ({labels[1]} vs {labels[0]}):")
            for i in range(min(5, len(data["attributes"]))):
                attr = data["attributes"][i]
                s1, s2, diff = (
                    data["scores1"][i],
                    data["scores2"][i],
                    data["differences"][i],
                )
                print(f"  {attr}: {s1:.3f} → {s2:.3f} ({diff:+.3f})")

            print(f"\nTop 5 degradations ({labels[1]} vs {labels[0]}):")
            for i in range(
                max(0, len(data["attributes"]) - 5), len(data["attributes"])
            ):
                attr = data["attributes"][i]
                s1, s2, diff = (
                    data["scores1"][i],
                    data["scores2"][i],
                    data["differences"][i],
                )
                print(f"  {attr}: {s1:.3f} → {s2:.3f} ({diff:+.3f})")

    return comparison_data


def analyze_attribute_flipping(
    results1, results2, labels, metric="f1", min_layers=3, save_path=None
):
    """Analyze which attributes flip performance across layers."""

    # Get common layers
    layers1 = set(results1.keys())
    layers2 = set(results2.keys())
    common_layers = layers1 & layers2

    def sort_key(layer):
        if layer == "last":
            return 9999
        elif layer == "proj":
            return 9998
        else:
            return layer

    common_layers = sorted(common_layers, key=sort_key)

    if len(common_layers) < min_layers:
        print(f"Not enough common layers ({len(common_layers)}) for flipping analysis")
        return

    # Collect attribute performance across all layers
    attribute_data = {}  # {attr: [(layer, score1, score2, diff), ...]}

    for layer in common_layers:
        attrs1 = {
            r["attribute"]: r[f"mean_{metric}"]
            for r in results1[layer]["individual_results"]
        }
        attrs2 = {
            r["attribute"]: r[f"mean_{metric}"]
            for r in results2[layer]["individual_results"]
        }

        common_attrs = set(attrs1.keys()) & set(attrs2.keys())

        for attr in common_attrs:
            if attr not in attribute_data:
                attribute_data[attr] = []

            score1 = attrs1[attr]
            score2 = attrs2[attr]
            diff = score2 - score1

            attribute_data[attr].append((layer, score1, score2, diff))

    # Filter attributes that appear in enough layers
    filtered_attrs = {
        attr: data for attr, data in attribute_data.items() if len(data) >= min_layers
    }

    if not filtered_attrs:
        print("No attributes found with sufficient layer coverage")
        return

    # Analyze flipping patterns
    flip_analysis = {}

    for attr, layer_data in filtered_attrs.items():
        differences = [diff for _, _, _, diff in layer_data]
        scores1 = [s1 for _, s1, _, _ in layer_data]
        scores2 = [s2 for _, _, s2, _ in layer_data]

        flip_analysis[attr] = {
            "layer_data": layer_data,
            "mean_diff": np.mean(differences),
            "std_diff": np.std(differences),
            "max_diff": max(differences),
            "min_diff": min(differences),
            "range_diff": max(differences) - min(differences),
            "mean_score1": np.mean(scores1),
            "mean_score2": np.mean(scores2),
            "flips": sum(
                1
                for i in range(len(differences) - 1)
                if differences[i] * differences[i + 1] < 0
            ),  # Sign changes
            "consistency": np.std(differences),  # Lower = more consistent
        }

    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Top flippers by range (biggest swings)
    top_flippers = sorted(
        flip_analysis.items(), key=lambda x: x[1]["range_diff"], reverse=True
    )[:15]

    flip_attrs = [attr for attr, _ in top_flippers]
    flip_ranges = [data["range_diff"] for _, data in top_flippers]
    flip_colors = [
        "red" if data["mean_diff"] < 0 else "green" for _, data in top_flippers
    ]

    bars1 = ax1.barh(range(len(flip_attrs)), flip_ranges, color=flip_colors, alpha=0.7)
    ax1.set_yticks(range(len(flip_attrs)))
    ax1.set_yticklabels(flip_attrs, fontsize=8)
    ax1.set_xlabel(f"Performance Range ({metric.upper()} difference)")
    ax1.set_title("Top 15 Attributes by Performance Swing")
    ax1.grid(True, alpha=0.3)

    # 2. Most inconsistent performers
    most_inconsistent = sorted(
        flip_analysis.items(), key=lambda x: x[1]["consistency"], reverse=True
    )[:15]

    inconsistent_attrs = [attr for attr, _ in most_inconsistent]
    inconsistent_vals = [data["consistency"] for _, data in most_inconsistent]

    ax2.barh(
        range(len(inconsistent_attrs)), inconsistent_vals, color="orange", alpha=0.7
    )
    ax2.set_yticks(range(len(inconsistent_attrs)))
    ax2.set_yticklabels(inconsistent_attrs, fontsize=8)
    ax2.set_xlabel(f"Performance Inconsistency (std of differences)")
    ax2.set_title("Most Inconsistent Attributes Across Layers")
    ax2.grid(True, alpha=0.3)

    # 3. Scatter plot: Overall performance vs consistency
    all_attrs = list(flip_analysis.keys())
    mean_diffs = [flip_analysis[attr]["mean_diff"] for attr in all_attrs]
    consistencies = [flip_analysis[attr]["consistency"] for attr in all_attrs]

    scatter = ax3.scatter(mean_diffs, consistencies, alpha=0.6, s=30)
    ax3.set_xlabel(f"Mean Performance Difference ({labels[1]} - {labels[0]})")
    ax3.set_ylabel("Performance Inconsistency")
    ax3.set_title("Performance Change vs Consistency")
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    # 4. Performance trajectory for top flippers
    top_5_flippers = top_flippers[:5]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_5_flippers)))

    for i, (attr, data) in enumerate(top_5_flippers):
        layer_data = data["layer_data"]
        layers = [str(layer) for layer, _, _, _ in layer_data]
        diffs = [diff for _, _, _, diff in layer_data]

        ax4.plot(
            layers,
            diffs,
            "o-",
            color=colors[i],
            label=attr[:20],
            linewidth=2,
            alpha=0.8,
        )

    ax4.set_xlabel("Layer")
    ax4.set_ylabel(f"Performance Difference ({labels[1]} - {labels[0]})")
    ax4.set_title("Performance Trajectories: Top 5 Flippers")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved flipping analysis to: {save_path}")

    plt.show()

    # Detailed text analysis
    print(f"\nAttribute Flipping Analysis ({metric.upper()}):")
    print("=" * 80)

    print(f"\nTop 10 Biggest Performance Swings:")
    print(
        f"{'Attribute':<25} {'Mean Δ':<8} {'Range':<8} {'Inconsist.':<10} {'Flips':<6}"
    )
    print("-" * 65)

    for attr, data in top_flippers[:10]:
        print(
            f"{attr[:24]:<25} {data['mean_diff']:>+7.3f} {data['range_diff']:>7.3f} "
            f"{data['consistency']:>9.3f} {data['flips']:>5}"
        )

    print(f"\nMost Improved Overall (Average across layers):")
    most_improved = sorted(
        flip_analysis.items(), key=lambda x: x[1]["mean_diff"], reverse=True
    )[:10]
    for attr, data in most_improved:
        print(f"  {attr:<30}: {data['mean_diff']:>+.3f} ± {data['std_diff']:.3f}")

    print(f"\nMost Degraded Overall (Average across layers):")
    most_degraded = sorted(flip_analysis.items(), key=lambda x: x[1]["mean_diff"])[:10]
    for attr, data in most_degraded:
        print(f"  {attr:<30}: {data['mean_diff']:>+.3f} ± {data['std_diff']:.3f}")

    print(f"\nMost Consistent Performers:")
    most_consistent = sorted(flip_analysis.items(), key=lambda x: x[1]["consistency"])[
        :10
    ]
    for attr, data in most_consistent:
        print(
            f"  {attr:<30}: consistency = {data['consistency']:.3f}, mean Δ = {data['mean_diff']:+.3f}"
        )

    return flip_analysis


def detailed_attribute_comparison(
    results1, results2, labels, layer, taxonomy, metric="f1", top_n=20, save_path=None
):
    """Detailed comparison of individual attributes for a specific layer."""

    if layer not in results1 or layer not in results2:
        print(f"Layer {layer} not found in one or both experiments")
        return

    # Get attribute data
    attrs1 = {
        r["attribute"]: r[f"mean_{metric}"]
        for r in results1[layer]["individual_results"]
    }
    attrs2 = {
        r["attribute"]: r[f"mean_{metric}"]
        for r in results2[layer]["individual_results"]
    }

    # Find common attributes
    common_attrs = set(attrs1.keys()) & set(attrs2.keys())
    if not common_attrs:
        print(f"No common attributes found for layer {layer}")
        return

    # Create comparison data with categories
    attr_comparison = []
    for attr in common_attrs:
        score1 = attrs1[attr]
        score2 = attrs2[attr]
        diff = score2 - score1
        pct_change = (diff / score1 * 100) if score1 > 0 else float("inf")
        category = taxonomy.get(attr, "unknown")

        attr_comparison.append(
            {
                "attribute": attr,
                "score1": score1,
                "score2": score2,
                "difference": diff,
                "pct_change": pct_change,
                "abs_diff": abs(diff),
                "category": category,
            }
        )

    # Sort by absolute difference
    attr_comparison.sort(key=lambda x: x["abs_diff"], reverse=True)

    # Get unique categories and assign colors
    unique_categories = list(set(item["category"] for item in attr_comparison))
    unique_categories.sort()
    category_colors = dict(
        zip(unique_categories, plt.cm.Set3(np.linspace(0, 1, len(unique_categories))))
    )

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Top improvements and degradations with category colors
    top_changes = attr_comparison[:top_n]

    attrs = [item["attribute"] for item in top_changes]
    diffs = [item["difference"] for item in top_changes]
    categories = [item["category"] for item in top_changes]

    # Color bars by category
    bar_colors = [category_colors[cat] for cat in categories]

    y_pos = range(len(attrs))
    bars = ax1.barh(
        y_pos, diffs, color=bar_colors, alpha=0.7, edgecolor="black", linewidth=0.5
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        [f"{attr[:25]} [{cat[:8]}]" for attr, cat in zip(attrs, categories)], fontsize=8
    )
    ax1.set_xlabel(f"Performance Difference ({labels[1]} - {labels[0]})")
    ax1.set_title(f"Top {top_n} Attribute Changes - Layer {layer} (by Category)")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.8)

    # Add values on bars
    for bar, diff in zip(bars, diffs):
        width = bar.get_width()
        ax1.text(
            width + (0.01 if width > 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{diff:+.3f}",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=7,
        )

    # Create category legend for bar plot
    # legend_elements = [
    #    plt.Rectangle((0, 0), 1, 1, facecolor=category_colors[cat], alpha=0.7,
    #                 edgecolor='black', label=cat)
    #    for cat in unique_categories if cat in categories
    # ]
    # ax1.legend(handles=legend_elements, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    # Before vs After scatter plot colored by category
    scores1 = [item["score1"] for item in attr_comparison]
    scores2 = [item["score2"] for item in attr_comparison]
    scatter_categories = [item["category"] for item in attr_comparison]
    scatter_colors = [category_colors[cat] for cat in scatter_categories]

    # Create scatter plot
    scatter = ax2.scatter(
        scores1,
        scores2,
        c=scatter_colors,
        alpha=0.6,
        s=30,
        edgecolor="black",
        linewidth=0.3,
    )

    # Perfect correlation line
    min_val = min(min(scores1), min(scores2))
    max_val = max(max(scores1), max(scores2))
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.8,
        label="No change",
        linewidth=2,
    )

    # Highlight top changers
    top_attrs = set(item["attribute"] for item in top_changes[:5])
    highlight_idx = [
        i for i, item in enumerate(attr_comparison) if item["attribute"] in top_attrs
    ]

    if highlight_idx:
        highlight_s1 = [scores1[i] for i in highlight_idx]
        highlight_s2 = [scores2[i] for i in highlight_idx]
        highlight_colors = [scatter_colors[i] for i in highlight_idx]

        ax2.scatter(
            highlight_s1,
            highlight_s2,
            c=highlight_colors,
            s=100,
            alpha=0.9,
            edgecolor="red",
            linewidth=2,
            label="Top 5 changes",
        )

        # Annotate top changers
        for i in highlight_idx[:5]:
            attr = attr_comparison[i]["attribute"]
            ax2.annotate(
                f"{attr[:15]}",
                (scores1[i], scores2[i]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax2.set_xlabel(f"{labels[0]} {metric.upper()} Score")
    ax2.set_ylabel(f"{labels[1]} {metric.upper()} Score")
    ax2.set_title(f"Before vs After Performance - Layer {layer} (by Category)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add correlation
    correlation = np.corrcoef(scores1, scores2)[0, 1]
    ax2.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Create category legend for scatter plot
    scatter_legend_elements = [
        plt.scatter(
            [],
            [],
            c=category_colors[cat],
            alpha=0.7,
            s=50,
            edgecolor="black",
            linewidth=0.3,
            label=cat,
        )
        for cat in unique_categories
    ]
    scatter_legend = ax2.legend(
        handles=scatter_legend_elements,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=8,
        title="Categories",
    )
    # Add the main legend back
    ax2.add_artist(ax2.legend(loc="lower right", fontsize=8))
    ax2.add_artist(scatter_legend)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved detailed comparison to: {save_path}")

    plt.show()

    # Print detailed results by category
    print(f"\nDetailed Attribute Comparison - Layer {layer}")
    print("=" * 80)
    print(f"Total common attributes: {len(common_attrs)}")
    print(f"Correlation: {correlation:.3f}")

    # Group by category for analysis
    category_analysis = {}
    for item in attr_comparison:
        cat = item["category"]
        if cat not in category_analysis:
            category_analysis[cat] = {"improvements": [], "degradations": [], "all": []}

        category_analysis[cat]["all"].append(item)
        if item["difference"] > 0:
            category_analysis[cat]["improvements"].append(item)
        else:
            category_analysis[cat]["degradations"].append(item)

    print(f"\nCategory Breakdown:")
    print(
        f"{'Category':<20} {'Count':<6} {'Avg Δ':<8} {'Improvements':<12} {'Degradations':<12}"
    )
    print("-" * 70)

    for cat in sorted(category_analysis.keys()):
        data = category_analysis[cat]
        avg_diff = np.mean([item["difference"] for item in data["all"]])
        n_improvements = len(data["improvements"])
        n_degradations = len(data["degradations"])

        print(
            f"{cat:<20} {len(data['all']):<6} {avg_diff:>+7.3f} {n_improvements:<12} {n_degradations:<12}"
        )

    print(f"\nTop 5 Improvements by Category:")
    for cat in sorted(category_analysis.keys()):
        improvements = category_analysis[cat]["improvements"]
        if improvements:
            improvements.sort(key=lambda x: x["difference"], reverse=True)
            print(f"\n{cat}:")
            for item in improvements[:5]:
                print(
                    f"  {item['attribute']:<30}: {item['score1']:.3f} → {item['score2']:.3f} "
                    f"({item['difference']:+.3f})"
                )

    print(f"\nTop 5 Degradations by Category:")
    for cat in sorted(category_analysis.keys()):
        degradations = category_analysis[cat]["degradations"]
        if degradations:
            degradations.sort(key=lambda x: x["difference"])
            print(f"\n{cat}:")
            for item in degradations[:5]:
                print(
                    f"  {item['attribute']:<30}: {item['score1']:.3f} → {item['score2']:.3f} "
                    f"({item['difference']:+.3f})"
                )

    return attr_comparison


def main():
    parser = argparse.ArgumentParser(
        description="Compare probe results between two experiments"
    )
    parser.add_argument(
        "--results_dir1",
        type=str,
        required=True,
        help="First experiment results directory",
    )
    parser.add_argument(
        "--results_dir2",
        type=str,
        required=True,
        help="Second experiment results directory",
    )
    parser.add_argument(
        "--labels", nargs=2, default=["Exp1", "Exp2"], help="Labels for the experiments"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "accuracy", "precision", "recall"],
        help="Metric to compare",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=str,
        help="Specific layers to compare in detail (e.g., 11 12 last)",
    )
    parser.add_argument("--save", type=str, help="Directory to save plots")
    parser.add_argument(
        "--flipping-analysis",
        action="store_true",
        help="Perform detailed attribute flipping analysis",
    )
    parser.add_argument(
        "--detailed-layer", type=str, help="Layer for detailed attribute comparison"
    )
    parser.add_argument(
        "--taxonomy", type=str, help="Taxonomy file for attribute categories"
    )

    args = parser.parse_args()

    # Load results
    print(f"Loading results from: {args.results_dir1}")
    results1 = load_results(args.results_dir1)
    print(f"Loading results from: {args.results_dir2}")
    results2 = load_results(args.results_dir2)

    if not results1 or not results2:
        print("Could not load results from one or both directories!")
        return

    # Load taxonomy if provided
    taxonomy = {}
    if args.taxonomy:
        try:
            import json

            with open(args.taxonomy, "r") as f:
                taxonomy = json.load(f)
            print(f"Loaded taxonomy with {len(taxonomy)} attributes")
        except Exception as e:
            print(f"Warning: Could not load taxonomy file {args.taxonomy}: {e}")
            print("Proceeding without taxonomy categories")

    # Sort layers properly for display
    def display_sort_key(layer):
        if layer == "last":
            return 9999
        elif layer == "proj":
            return 9998
        else:
            return layer

    sorted_layers1 = sorted(results1.keys(), key=display_sort_key)
    sorted_layers2 = sorted(results2.keys(), key=display_sort_key)

    print(f"\nExp1 layers: {sorted_layers1}")
    print(f"Exp2 layers: {sorted_layers2}")

    # Overall comparison
    if args.save:
        overview_save_path = f"{args.save}/comparison_overview.png"
    else:
        overview_save_path = None

    compare_overview_performance(
        results1, results2, args.labels, args.metric, overview_save_path
    )

    # Specific layer comparison
    if args.layers:
        # Convert layer names to appropriate types
        layers_to_compare = []
        for layer in args.layers:
            if layer in ["last", "proj"]:
                layers_to_compare.append(layer)
            else:
                try:
                    layers_to_compare.append(int(layer))
                except ValueError:
                    print(f"Invalid layer: {layer}")
                    continue

        if layers_to_compare:
            if args.save:
                specific_save_path = f"{args.save}/comparison_specific_layers.png"
            else:
                specific_save_path = None

            compare_specific_layers(
                results1,
                results2,
                args.labels,
                layers_to_compare,
                args.metric,
                specific_save_path,
            )

    # Attribute flipping analysis
    if args.flipping_analysis:
        if args.save:
            flipping_save_path = f"{args.save}/attribute_flipping_analysis.png"
        else:
            flipping_save_path = None

        analyze_attribute_flipping(
            results1, results2, args.labels, args.metric, save_path=flipping_save_path
        )

    # Detailed layer analysis
    if args.detailed_layer:
        # Convert layer name to appropriate type
        if args.detailed_layer in ["last", "proj"]:
            detailed_layer = args.detailed_layer
        else:
            try:
                detailed_layer = int(args.detailed_layer)
            except ValueError:
                print(f"Invalid layer: {args.detailed_layer}")
                return

        if args.save:
            detailed_save_path = f"{args.save}/detailed_layer_{detailed_layer}.png"
        else:
            detailed_save_path = None

        detailed_attribute_comparison(
            results1,
            results2,
            args.labels,
            detailed_layer,
            taxonomy,
            args.metric,
            save_path=detailed_save_path,
        )


if __name__ == "__main__":
    main()
