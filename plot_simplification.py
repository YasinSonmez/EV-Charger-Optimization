#!/usr/bin/env python3
"""All 6 regions — before/after cleaning comparison."""
import sys, warnings; sys.path.insert(0,'.'); warnings.filterwarnings('ignore')
import osmnx as ox; ox.settings.use_cache = True; ox.settings.log_console = False
import networkx as nx; import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; import numpy as np
from collections import Counter
from src.graph_cache import get_graph
from src.road_network import RoadNet

REGIONS = [
    ("Berkeley",    (-122.30, 37.85, -122.20, 37.90)),
    ("San Francisco",(-122.52, 37.72, -122.35, 37.82)),
    ("Bay Area",    (-122.55, 37.70, -122.10, 37.95)),
    ("College Park",(-77.00, 38.95, -76.90, 39.00)),
    ("DC Area",     (-77.15, 38.82, -76.85, 39.02)),
    ("Manhattan",   (-74.04, 40.70, -73.90, 40.82)),
]
MAJOR = ["motorway","trunk","primary","secondary",
         "motorway_link","trunk_link","primary_link","secondary_link"]

def run_pipeline(G):
    G = G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    rn = RoadNet('t'); rn.graph = G
    rn._merge_degree2_chains(); rn._prune_dead_ends_graph()
    rn._suppress_t_junctions(threshold=200)
    rn._remove_chain_fragments()  # only remove orphan chains, keep intersecting fragments
    rn._merge_degree2_chains(); rn._prune_dead_ends_graph()
    rn._keep_largest_scc()  # safety: guarantee connectivity
    rn._merge_junctions(threshold=80)  # merge close-node clusters at interchanges
    return rn.graph

def plot(ax, G, title, color='green'):
    if G is None or len(G.edges) == 0:
        ax.text(0.5, 0.5, '(empty)', ha='center', va='center', fontsize=10);
        ax.set_title(title, fontsize=7); ax.axis('off'); return
    ox.plot_graph(G, ax=ax, show=False, close=False,
                  node_size=4, node_color=color, edge_color='#333',
                  edge_linewidth=0.2, bgcolor='white')
    ax.set_title(title, fontsize=7); ax.axis('off')

fig, axes = plt.subplots(len(REGIONS), 3, figsize=(15, 3.5*len(REGIONS)))

for row, (name, coords) in enumerate(REGIONS):
    G_all = get_graph(coords, highway_types=None)
    G_maj = get_graph(coords, highway_types=MAJOR)
    G_cln = run_pipeline(G_maj.copy())
    
    n0 = len(G_maj.nodes)
    n1 = len(G_cln.nodes); e1 = len(G_cln.edges)
    ud = Counter()
    for n in G_cln.nodes:
        ud[len(set(G_cln.successors(n))|set(G_cln.predecessors(n)))] += 1
    pct = n1/max(n0,1)*100
    
    plot(axes[row,0], G_all, f'{name}\nAll Roads\n{len(G_all.nodes)}n/{len(G_all.edges)}e', '#d73027')
    plot(axes[row,1], G_maj, f'Major Raw\n{n0}n/{len(G_maj.edges)}e', '#fc8d59')
    plot(axes[row,2], G_cln, f'Cleaned (LSCC+merge+prune+T-simp200m)\n{n1}n/{e1}e ({pct:.1f}%)\nd2={ud[2]} d3={ud[3]} d4={ud[4]}', '#1a9850')

fig.suptitle('Network Simplification — All 6 Regions\n'
             'Left: All roads | Center: Major roads (motorway–secondary) | Right: After full cleaning pipeline',
             fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig('all_regions_simplification.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print('Saved all_regions_simplification.png')
