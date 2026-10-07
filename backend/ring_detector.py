from collections import Counter, defaultdict
import networkx as nx


def build_graph(edges):
    g = nx.DiGraph()
    for e in edges:
        s = str(e["sender"])
        r = str(e["receiver"])
        if s != r:
            g.add_edge(s, r, amount=float(e.get("amount", 0.0)))
    return g


def detect_rings(edges, max_cycles=100):
    """
    Lightweight graph intelligence for the prototype.

    Returns structural patterns, not fraud labels. A detected pattern is a
    risk signal and must not be interpreted as proof of laundering.
    """
    g = build_graph(edges)
    findings = []

    # Star / funnel patterns.
    for node in g.nodes:
        out_n = g.out_degree(node)
        in_n = g.in_degree(node)

        if out_n >= 8:
            findings.append({
                "type": "star_out",
                "center": node,
                "degree": out_n,
                "risk_signal": min(1.0, out_n / 25.0),
            })

        if in_n >= 8:
            findings.append({
                "type": "funnel_in",
                "center": node,
                "degree": in_n,
                "risk_signal": min(1.0, in_n / 25.0),
            })

    # Short directed cycles.
    try:
        cycles = nx.simple_cycles(g)
        for i, cycle in enumerate(cycles):
            if i >= max_cycles:
                break
            if 3 <= len(cycle) <= 6:
                findings.append({
                    "type": "cycle",
                    "members": cycle,
                    "length": len(cycle),
                    "risk_signal": 0.85,
                })
    except Exception:
        pass

    # Chains: nodes with one predecessor and one successor.
    for node in g.nodes:
        if g.in_degree(node) == 1 and g.out_degree(node) == 1:
            p = next(iter(g.predecessors(node)))
            q = next(iter(g.successors(node)))
            if p != q:
                findings.append({
                    "type": "chain_link",
                    "members": [p, node, q],
                    "risk_signal": 0.45,
                })

    # Deduplicate coarse findings.
    unique = []
    seen = set()
    for f in findings:
        key = (f["type"], str(f.get("center", "")), str(f.get("members", "")))
        if key not in seen:
            seen.add(key)
            unique.append(f)

    return unique
