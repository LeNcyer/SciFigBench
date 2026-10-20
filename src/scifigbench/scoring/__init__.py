"""Approved numerical functions. SciPy is required; no silent greedy fallback."""
try:
    from scipy.optimize import linear_sum_assignment as _linear_sum_assignment
except ImportError as exc:
    raise ImportError("SciFigBench evaluation requires SciPy; install the declared dependencies.") from exc
from ._tasks.dense_content_perception import score as _dense_content_perception
from ._tasks.edge_level_verification import score as _edge_level_verification
from ._tasks.node_level_degree_counting import score as _node_level_degree_counting
from ._tasks.path_level_reachability import score as _path_level_reachability
from ._tasks.information_flow_tracing import score as _information_flow_tracing

SCORERS = {
    'dense_content_perception': _dense_content_perception,
    'edge_level_verification': _edge_level_verification,
    'node_level_degree_counting': _node_level_degree_counting,
    'path_level_reachability': _path_level_reachability,
    'information_flow_tracing': _information_flow_tracing,
}
