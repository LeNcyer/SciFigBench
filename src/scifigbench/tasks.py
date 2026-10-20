"""Public task names and data files; no construction or annotation dependencies."""
from dataclasses import dataclass

@dataclass(frozen=True)
class TaskSpec:
    name: str
    metric: str

    @property
    def filename(self) -> str:
        return f"qa_{self.name}.jsonl"

TASKS = {name: TaskSpec(name, metric) for name, metric in [
    ("dense_content_perception", "Position-wise ANLS"),
    ("edge_level_verification", "Accuracy"),
    ("node_level_degree_counting", "Accuracy"),
    ("path_level_reachability", "Accuracy"),
    ("information_flow_tracing", "Set-based ANLS*"),
]}
COLORS = ["red", "orange", "yellow", "green", "blue", "violet"]

def select_tasks(task: str = "all") -> tuple[str, ...]:
    if task == "all":
        return tuple(TASKS)
    if not isinstance(task, str) or task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; choose 'all' or one of: {', '.join(TASKS)}")
    return (task,)
