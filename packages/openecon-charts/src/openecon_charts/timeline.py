"""Versioned, lossless record interning for network timeline transport.

Public PlotSpec data remains expanded. Only HTML, saved views and local plot
artifacts use this representation, and only when it saves bytes. No external
frame files or asynchronous frame fetches are needed offline.
"""
import json

from .network import MAX_EDGES, MAX_NODES, MAX_PAYLOAD_BYTES, validate_network

ENCODING = "timeline-pool-v1"


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def pack(plot):
    graph = plot.get("config", {}).get("network")
    if not graph or "frames" not in graph:
        return plot
    pools, indices = {"nodes": [], "edges": []}, {"nodes": {}, "edges": {}}

    def reference(network):
        result = {key: value for key, value in network.items() if key not in ("nodes", "edges", "frames")}
        for kind in pools:
            refs = []
            for record in network[kind]:
                key = _json(record)
                index = indices[kind].get(key)
                if index is None:
                    index = len(pools[kind])
                    indices[kind][key] = index
                    pools[kind].append(record)
                refs.append(index)
            result[kind] = refs
        return result

    encoded = {"encoding": ENCODING, "base": reference(graph),
               "frames": [{"label": frame["label"], "network": reference(frame["network"])}
                          for frame in graph["frames"]], **pools}
    result = {**plot, "config": {**plot["config"], "network": encoded}}
    return result if len(_json(result)) < len(_json(plot)) else plot


def unpack(plot):
    """Validate resource admission before expanding any reference arrays."""
    config = plot.get("config") if isinstance(plot, dict) else None
    graph = config.get("network") if isinstance(config, dict) else None
    if not isinstance(graph, dict) or "encoding" not in graph:
        return plot
    if set(graph) != {"encoding", "base", "frames", "nodes", "edges"} or graph["encoding"] != ENCODING:
        raise ValueError("Unsupported network timeline encoding.")
    frames = graph["frames"]
    if not isinstance(frames, list) or not 1 <= len(frames) <= 60:
        raise ValueError("Timeline transport requires 1 to 60 frames.")
    for frame in frames:
        if not isinstance(frame, dict) or set(frame) != {"label", "network"}:
            raise ValueError("Invalid timeline transport frame.")
    networks = [graph["base"], *[frame["network"] for frame in frames]]
    used = {"nodes": set(), "edges": set()}
    for kind, cap in (("nodes", MAX_NODES), ("edges", MAX_EDGES)):
        pool = graph[kind]
        if not isinstance(pool, list) or len(pool) > cap:
            raise ValueError("Timeline record pool exceeds the display budget.")
        total = 0
        for network in networks:
            if not isinstance(network, dict) or "frames" in network:
                raise ValueError("Invalid nested timeline transport network.")
            refs = network.get(kind)
            if not isinstance(refs, list):
                raise ValueError("Timeline references must be lists.")
            total += len(refs)
            if total > cap:
                raise ValueError("Timeline references exceed the aggregate display budget.")
            for ref in refs:
                if type(ref) is not int or not 0 <= ref < len(pool):
                    raise ValueError("Invalid timeline record reference.")
                used[kind].add(ref)
        if len(used[kind]) != len(pool):
            raise ValueError("Timeline record pools cannot contain unused records.")
    if len(_json(graph).encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError("Timeline transport exceeds its 128 MiB payload budget.")

    def expand(network):
        return {**network, "nodes": [graph["nodes"][index] for index in network["nodes"]],
                "edges": [graph["edges"][index] for index in network["edges"]]}

    expanded = expand(graph["base"])
    expanded["frames"] = [{"label": frame["label"], "network": expand(frame["network"])} for frame in frames]
    # This checks the conservative expanded workspace budget as well as typed
    # identity, endpoints, labels, attributes, direction and full/shown counts.
    expanded = validate_network(expanded)
    return {**plot, "config": {**plot["config"], "network": expanded}}
