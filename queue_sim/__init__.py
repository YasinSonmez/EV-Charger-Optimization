"""Queue-based traffic simulation package (macOS-only due to liblsp.dylib)."""
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)

try:
    from .queue_model_EV import Link, Node, Agent, Simulation
    from .interface import ShortestPath, Graph
    from .runner_EV import Runner
    QUEUE_SIM_AVAILABLE = True
    _QUEUE_SIM_ERROR = None
except (OSError, ImportError) as _e:
    QUEUE_SIM_AVAILABLE = False
    _QUEUE_SIM_ERROR = str(_e)
    Link = Node = Agent = Simulation = None
    ShortestPath = Graph = Runner = None
