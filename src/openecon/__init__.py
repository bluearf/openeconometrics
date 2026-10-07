"""OpenEconometrics's public Python analysis interface.

Resolve analysis exports on first use so the web control plane can import the
package without starting the separate tensor and dataframe runtime.
"""

from importlib import import_module
from typing import TYPE_CHECKING

# Keep the small, dependency-free formatter bound as a function. Otherwise
# loading the openecon.latex submodule would replace the lazy ``oe.latex``
# function export with that module as a side effect of normal Python imports.
from openecon.latex import Latex, latex, regression_table, to_latex

__version__ = "0.3.18a2"

if TYPE_CHECKING:
    from openecon.analysis import AnalysisError, capabilities, fit, ols, logit, probit
    from openecon.data import read, example_frame as example
    from openecon.models import ModelSpec, ResultBundle
    from openecon.console import load_dataset
    from openecon import plotting as plot
    from openecon.frame import DataFrame
    from openecon.dataset import Dataset, scan
    from openecon.script_packages import install
    from openecon.networks import Network, network
    from openecon._network_multi import MultiNetwork, multigraph, read_multigraph
    from openecon._network_dynamic import DynamicNetwork, dynamic_network, read_dynamic_network
    from openecon._network_matching import NetworkMatchingResult
    from openecon._network_multilayer import MultilayerNetwork, multilayer_network, read_multilayer_network
    from openecon._network_signed import SignedNetwork, signed_network
    from openecon._network_io import read_network
    from openecon._network_flow import NetworkFlowResult
    from openecon._network_cut import NetworkCutResult
    from openecon._network_sbm import NetworkBlockResult
    from openecon._network_temporal import NetworkSnapshots, network_snapshots
    from openecon._network_hypergraph import Hypergraph, hypergraph, read_hypergraph

__all__ = ["AnalysisError", "ModelSpec", "ResultBundle", "capabilities", "fit", "ols", "logit", "probit", "read", "scan", "Dataset", "example", "load_dataset", "plot", "DataFrame", "Latex", "latex", "to_latex", "regression_table", "install"]
__all__ += ["test", "testparm", "lincom", "nlcom", "predict", "margins"]
__all__ += ["Network", "NetworkFlowResult", "NetworkCutResult", "network", "read_network"]
__all__ += ["SignedNetwork", "signed_network"]
__all__ += ["NetworkBlockResult", "NetworkSnapshots", "network_snapshots"]
__all__ += ["MultiNetwork", "multigraph", "read_multigraph"]
__all__ += ["DynamicNetwork", "dynamic_network", "read_dynamic_network"]
__all__ += ["ergm", "simulate_ergm"]
__all__ += ["saom", "simulate_saom"]
__all__ += ["gaussian_block_model", "mixed_membership_block_model"]
__all__ += ["network_embedding", "network_gnn", "EmbeddingResult", "GNNResult"]
__all__ += ["MultilayerNetwork", "multilayer_network", "read_multilayer_network"]
__all__ += ["Hypergraph", "hypergraph", "read_hypergraph"]
__all__ += ["NetworkMatchingResult"]

_EXPORTS = {
    "network_embedding": ("openecon._network_learning", "network_embedding"),
    "network_gnn": ("openecon._network_learning", "network_gnn"),
    "EmbeddingResult": ("openecon._network_learning", "EmbeddingResult"),
    "GNNResult": ("openecon._network_learning", "GNNResult"),
    "gaussian_block_model": ("openecon._network_latent_blocks", "gaussian_block_model"),
    "mixed_membership_block_model": ("openecon._network_latent_blocks", "mixed_membership_block_model"),
    "saom": ("openecon._network_saom", "saom"),
    "simulate_saom": ("openecon._network_saom", "simulate_saom"),
    "ergm": ("openecon._network_ergm", "ergm"),
    "simulate_ergm": ("openecon._network_ergm", "simulate_ergm"),
    **{name: ("openecon.analysis", name) for name in ("fit", "ols", "logit", "probit")},
    **{name: ("openecon.analysis_contracts", name) for name in ("AnalysisError", "capabilities")},
    "ModelSpec": ("openecon.models", "ModelSpec"),
    "ResultBundle": ("openecon.models", "ResultBundle"),
    "read": ("openecon.data", "read"),
    "example": ("openecon.data", "example_frame"),
    "load_dataset": ("openecon.console", "load_dataset"),
    "plot": ("openecon.plotting", None),
    "DataFrame": ("openecon.frame", "DataFrame"),
    "Dataset": ("openecon.dataset", "Dataset"),
    "scan": ("openecon.dataset", "scan"),
    "install": ("openecon.script_packages", "install"),
    "SignedNetwork": ("openecon._network_signed", "SignedNetwork"),
    "signed_network": ("openecon._network_signed", "signed_network"),
    "Network": ("openecon.networks", "Network"),
    "Hypergraph": ("openecon._network_hypergraph", "Hypergraph"),
    "hypergraph": ("openecon._network_hypergraph", "hypergraph"),
    "read_hypergraph": ("openecon._network_hypergraph", "read_hypergraph"),
    "NetworkFlowResult": ("openecon._network_flow", "NetworkFlowResult"),
    "NetworkCutResult": ("openecon._network_cut", "NetworkCutResult"),
    "NetworkBlockResult": ("openecon._network_sbm", "NetworkBlockResult"),
    "NetworkSnapshots": ("openecon._network_temporal", "NetworkSnapshots"),
    "network_snapshots": ("openecon._network_temporal", "network_snapshots"),
    "network": ("openecon.networks", "network"),
    "open_network": ("openecon.networks", "open_network"),
    "DiskNetwork": ("openecon._network_store", "DiskNetwork"),
    "read_network": ("openecon._network_io", "read_network"),
    "MultiNetwork": ("openecon._network_multi", "MultiNetwork"),
    "multigraph": ("openecon._network_multi", "multigraph"),
    "read_multigraph": ("openecon._network_multi", "read_multigraph"),
    "DynamicNetwork": ("openecon._network_dynamic", "DynamicNetwork"),
    "dynamic_network": ("openecon._network_dynamic", "dynamic_network"),
    "read_dynamic_network": ("openecon._network_dynamic", "read_dynamic_network"),
    "NetworkMatchingResult": ("openecon._network_matching", "NetworkMatchingResult"),
    "MultilayerNetwork": ("openecon._network_multilayer", "MultilayerNetwork"),
    "multilayer_network": ("openecon._network_multilayer", "multilayer_network"),
    "read_multilayer_network": ("openecon._network_multilayer", "read_multilayer_network"),
    **{name: ("openecon.analysis", name) for name in
       ("test", "testparm", "lincom", "nlcom")},
    **{name: ("openecon.analysis", name) for name in ("predict", "margins")},
    "Latex": ("openecon.latex", "Latex"),
    "latex": ("openecon.latex", "latex"),
    "to_latex": ("openecon.latex", "to_latex"),
    "regression_table": ("openecon.latex", "regression_table"),
}


def _registry_exports() -> dict:
    # Estimator families publish their Stata-style functions (oe.xtreg, ...)
    # through the Torch-free registry; their code loads on first use.
    from openecon.econometrics.registry import public_exports
    return public_exports()


def __getattr__(name: str):
    target = _EXPORTS.get(name) or _registry_exports().get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = import_module(module_name)
    value = module if attribute is None else getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_registry_exports()))
