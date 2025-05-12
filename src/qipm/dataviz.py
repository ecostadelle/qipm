import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from collections import defaultdict

def plot_transition_feat_relevance(
    ipm_id: np.ndarray,
    ipm_ood: np.ndarray,
    labels: list[str],
    *,
    figsize: tuple[float, float] = (12, 8),
    row_gap: float = 0.01,  # Space between feature rows
    node_height: float = 0.015,  # Height of nodes
    min_width: float = 0.0,  # Minimum node width
    max_width: float = 0.25,  # Maximum node width
    flow_color: str = '#7f7f7f',  # Color of connections
    node_color_id: str = '#1f77b4',  # Color of ID nodes
    node_color_ood: str = '#ff7f0e',  # Color of OOD nodes
    flow_alpha: float = 0.5,  # Transparency of connections
    flow_linewidth: float = 1.0,  # Thickness of connections
    feature_fontsize: float = 9,
    title_fontsize: float = 12,
) -> tuple[Figure, Axes]:
    """
    Plot a diagram that represents the transition of feature relevance 
    before and after a domain adaptation strategy. The visualization 
    consists of a connected set of horizontal bars, where each feature 
    is represented by a pair of bars positioned on the left and right 
    columns.

    The left column (blue) corresponds to the feature importance 
    values in the in-distribution dataset, while the right column 
    (orange) represents the out-of-distribution dataset. The width of 
    each bar is proportional to the feature importance, allowing an 
    intuitive comparison of relevance shifts between the two 
    distributions.

    Curved lines connect the corresponding features across both 
    columns, visually indicating changes in ranking and importance 
    due to domain adaptation. This diagram helps to identify 
    features that gain or lose importance when transitioning from 
    the source to the target domain.

    Parameters
    ----------
    ipm_id : np.ndarray
        Feature importance values for the ID dataset.
    ipm_ood : np.ndarray
        Feature importance values for the OOD dataset.
    labels : list[str]
        List of feature labels.
    figsize : tuple[float, float], optional
        Size of the figure (default is (12, 8)).
    row_gap : float, optional
        Vertical spacing between feature nodes (default is 0.01).
    node_height : float, optional
        Height of the rectangular nodes (default is 0.015).
    min_width : float, optional
        Minimum width of the nodes (default is 0.0).
    max_width : float, optional
        Maximum width of the nodes (default is 0.25).
    flow_color : str, optional
        Color of the connection lines (default is '#7f7f7f').
    node_color_id : str, optional
        Color for ID nodes (default is '#1f77b4').
    node_color_ood : str, optional
        Color for OOD nodes (default is '#ff7f0e').
    flow_alpha : float, optional
        Transparency of the connection lines (default is 0.5).
    flow_linewidth : float, optional
        Thickness of the connection lines (default is 1.0).

    Returns
    -------
    tuple
        A tuple containing the Matplotlib figure and axes objects.
    """

    n_features = len(ipm_id)

    # Normalize node widths
    all_values = np.concatenate([ipm_id, ipm_ood])
    delta_width = max_width - min_width
    width_id = ipm_id / all_values.max() * delta_width + min_width
    width_ood = ipm_ood / all_values.max() * delta_width + min_width

    # Sort by feature importance
    rank_id = np.argsort(-ipm_id)
    rank_ood = np.argsort(-ipm_ood)

    # Define feature positions
    y_positions = np.linspace(1 - row_gap, row_gap, n_features)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw connections between ID and OOD
    for i in range(n_features):
        idx_id = np.argmax(rank_id == i)
        idx_ood = np.argmax(rank_ood == i)

        y_src = y_positions[idx_id]
        y_dst = y_positions[idx_ood]

        x_src = 0.2 + width_id[i]  # End of ID node
        x_dst = 0.8 - width_ood[i]  # OOD node expands to the left

        verts = [
            (x_src, y_src),  # Starting point
            (x_src + 0.1, y_src),  # Curve control point
            (x_dst - 0.1, y_dst),  # Curve control point
            (x_dst, y_dst)  # End point where OOD node begins
        ]

        path = Path(verts, [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
        patch = patches.PathPatch(
            path, edgecolor=flow_color, facecolor='none',
            lw=flow_linewidth, alpha=flow_alpha
        )
        ax.add_patch(patch)

        # Draw ID nodes (left)
        y = y_positions[idx_id] - node_height / 2
        ax.add_patch(patches.Rectangle(
            (0.2, y), width_id[i], node_height,
            facecolor=node_color_id
        ))
        ax.text(0.18, y + node_height / 2, labels[i], 
                ha='right', va='center', fontsize=feature_fontsize)

        # Draw OOD nodes (right, expanding left)
        y = y_positions[idx_ood] - node_height / 2
        ax.add_patch(patches.Rectangle(
            (0.8 - width_ood[i], y), width_ood[i], node_height,
            facecolor=node_color_ood
        ))
        ax.text(0.82, y + node_height / 2, labels[i], 
                ha='left', va='center', fontsize=feature_fontsize)

    # Add titles
    ax.text(0.2, 1.02, 'ID Dataset', ha='center', va='bottom', 
            fontsize=title_fontsize, weight='bold', color=node_color_id)
    ax.text(0.8, 1.02, 'OOD Dataset', ha='center', va='bottom', 
            fontsize=title_fontsize, weight='bold', color=node_color_ood)

    return fig, ax

    
def join_onehoted_importance(
    original_feats: np.ndarray, 
    importances: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Aggregates the importance of features that were separated using OneHot Encoding.

    Parameters
    ----------
    features_to_aggregate : ndarray of shape (m,), dtype=str
        List of feature prefixes that should be aggregated.

    importance : ndarray of shape (n_features,), dtype=float64
        Importance values of each feature.

    labels : ndarray of shape (n_features,), dtype=str
        Feature names corresponding to the importance values.

    Returns
    -------
    result : dict
        Dictionary containing the aggregated importance of features.
    """

    result = defaultdict(float)

    # Convert the list of features to aggregate 
    # into a set for efficient lookup
    feature_set = set(original_feats)

    for label, importance in zip(labels, importances):
        # Check if the label starts with any 
        # of the features to be aggregated
        aggregated = False
        for feat in feature_set:
            if label.startswith(feat):
                result[f"{feat}*"] += importance
                aggregated = True
                break  # Stop checking once a match is found

        # If the label does not match any feature in the 
        # aggregation list, store it as is
        if not aggregated:
            result[label] += importance  

    return dict(result)


def id_ood_labels_from_dict(id_agg, ood_agg):
    result = {
        "ipm_id": np.array(list(id_agg.values())), 
        "ipm_ood": np.array([ood_agg[k] for k in id_agg.keys()]), 
        "labels": np.array(list(id_agg.keys())),
    }
    return result

def join_and_plot(
    ipm_id, 
    ipm_ood, 
    labels, 
    onehot_feats,
    *,
    figsize: tuple[float, float] = (12, 8),
    row_gap: float = 0.01,  # Space between feature rows
    node_height: float = 0.015,  # Height of nodes
    min_width: float = 0.0,  # Minimum node width
    max_width: float = 0.25,  # Maximum node width
    flow_color: str = '#7f7f7f',  # Color of connections
    node_color_id: str = '#1f77b4',  # Color of ID nodes
    node_color_ood: str = '#ff7f0e',  # Color of OOD nodes
    flow_alpha: float = 0.5,  # Transparency of connections
    flow_linewidth: float = 1.0,  # Thickness of connections
    feature_fontsize: float = 9,
    title_fontsize: float = 12,
) -> tuple[Figure, Axes]:
    """
    Plot a diagram that represents the transition of feature relevance 
    before and after a domain adaptation strategy. The visualization 
    consists of a connected set of horizontal bars, where each feature 
    is represented by a pair of bars positioned on the left and right 
    columns.

    The left column (blue) corresponds to the feature importance 
    values in the in-distribution dataset, while the right column 
    (orange) represents the out-of-distribution dataset. The width of 
    each bar is proportional to the feature importance, allowing an 
    intuitive comparison of relevance shifts between the two 
    distributions.

    Curved lines connect the corresponding features across both 
    columns, visually indicating changes in ranking and importance 
    due to domain adaptation. This diagram helps to identify 
    features that gain or lose importance when transitioning from 
    the source to the target domain.

    Parameters
    ----------
    ipm_id : np.ndarray
        Feature importance values for the ID dataset.
    ipm_ood : np.ndarray
        Feature importance values for the OOD dataset.
    labels : list[str]
        List of feature labels.
    onehot_feats : list[str]
        List of feature to be joint.
    figsize : tuple[float, float], optional
        Size of the figure (default is (12, 8)).
    row_gap : float, optional
        Vertical spacing between feature nodes (default is 0.01).
    node_height : float, optional
        Height of the rectangular nodes (default is 0.015).
    min_width : float, optional
        Minimum width of the nodes (default is 0.0).
    max_width : float, optional
        Maximum width of the nodes (default is 0.25).
    flow_color : str, optional
        Color of the connection lines (default is '#7f7f7f').
    node_color_id : str, optional
        Color for ID nodes (default is '#1f77b4').
    node_color_ood : str, optional
        Color for OOD nodes (default is '#ff7f0e').
    flow_alpha : float, optional
        Transparency of the connection lines (default is 0.5).
    flow_linewidth : float, optional
        Thickness of the connection lines (default is 1.0).

    Returns
    -------
    tuple
        A tuple containing the Matplotlib figure and axes objects.
    """

    
    id_agg = join_onehoted_importance(onehot_feats, ipm_id, labels)
    ood_agg =  join_onehoted_importance(onehot_feats, ipm_ood, labels)
    
    args = id_ood_labels_from_dict(id_agg, ood_agg)
    
    return plot_transition_feat_relevance(
            **args,
            figsize=figsize,
            row_gap=row_gap,
            node_height=node_height,
            min_width=min_width,
            max_width=max_width,
            flow_color=flow_color,
            node_color_id=node_color_id,
            node_color_ood=node_color_ood,
            flow_alpha=flow_alpha,
            flow_linewidth=flow_linewidth,
            feature_fontsize=feature_fontsize,
            title_fontsize=title_fontsize,
    )
