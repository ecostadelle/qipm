import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_transition_feat_relevance(
    ipm_id: np.ndarray,
    ipm_ood: np.ndarray,
    labels: list[str],
    figsize: tuple[float, float] = (12, 8),
    row_gap: float = 0.01,  # Space between feature rows
    node_height: float = 0.015,  # Height of nodes
    min_width: float = 0.0,  # Minimum node width
    max_width: float = 0.25,  # Maximum node width
    flow_color: str = '#7f7f7f',  # Color of connections
    node_color_id: str = '#1f77b4',  # Color of ID nodes
    node_color_ood: str = '#ff7f0e',  # Color of OOD nodes
    flow_alpha: float = 0.5,  # Transparency of connections
    flow_linewidth: float = 1.0  # Thickness of connections
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
        ax.text(0.18, y + node_height / 2, labels[i], ha='right', va='center', fontsize=9)

        # Draw OOD nodes (right, expanding left)
        y = y_positions[idx_ood] - node_height / 2
        ax.add_patch(patches.Rectangle(
            (0.8 - width_ood[i], y), width_ood[i], node_height,
            facecolor=node_color_ood
        ))
        ax.text(0.82, y + node_height / 2, labels[i], ha='left', va='center', fontsize=9)

    # Add titles
    ax.text(0.2, 1.02, 'ID Dataset', ha='center', va='bottom', fontsize=12, weight='bold', color=node_color_id)
    ax.text(0.8, 1.02, 'OOD Dataset', ha='center', va='bottom', fontsize=12, weight='bold', color=node_color_ood)

    return fig, ax