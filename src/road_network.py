import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from src.graph_cache import get_graph


MAJOR_ROAD_TYPES = [
    'motorway', 'trunk', 'primary', 'secondary',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
]


class RoadNet:
    def __init__(self, name):
        self.name = name
        self.graph = []
        self.nodes = []
        self.edges = []
        self.exit_node_id = []
        self.demand = []
        self.osmid_to_nid_dict = {}
        self.nid_to_osmid_dict = {}
        self.stage_counts = {}  # {stage_name: {'nodes': n, 'edges': e}}

    def get_map(self, no_lat, so_lat, east_long, west_long,
                highway_types=None, prune_dead_ends=False, merge_chains=True,
                suppress_t_junctions=True, cross_threshold=200):
        """Download OSM street graph and apply cleaning pipeline.

        Pipeline: LSCC → merge chains → prune leaves → 
                   T-junction suppression (short cross-streets) → 
                   LSCC re-extraction → merge → prune → rearrange_data

        Args:
            highway_types: If provided, filter OSM to these types. None = all drivable.
            prune_dead_ends: Enable leaf pruning.
            merge_chains: Enable degree-2 chain merging.
            suppress_t_junctions: Enable T-junction suppression (removes cross-streets
                                  shorter than cross_threshold at T-junctions).
            cross_threshold: Max cross-street length (m) to suppress at T-junctions.
        """
        if highway_types:
            self.graph = get_graph(
                (west_long, so_lat, east_long, no_lat), highway_types=highway_types)
            self._keep_largest_scc()
        else:
            self.graph = get_graph(
                (west_long, so_lat, east_long, no_lat))
        if highway_types is not None or prune_dead_ends:
            if merge_chains:
                self._merge_degree2_chains()
            if prune_dead_ends:
                self._prune_dead_ends_graph()
            if suppress_t_junctions:
                self._suppress_t_junctions(threshold=cross_threshold)
                # Remove only orphan chain fragments (no intersections)
                self._remove_chain_fragments()
                # Re-merge and re-prune
                self._merge_degree2_chains()
                self._prune_dead_ends_graph()
            # Safety: guarantee connectivity — keep only largest SCC
            self._keep_largest_scc()
            # Merge close-node clusters at interchanges
            self._merge_junctions(threshold=80)
        self.rearrange_data()

    def _merge_degree2_chains(self):
        """Merge all pass-through nodes (undirected-degree = 2).

        A node with exactly 2 unique neighbors is a pass-through — traffic
        enters from one side and exits the other. There is no routing
        decision to make, so the node can be eliminated.

        Applies 3 iterations to catch nodes that become pass-throughs
        after adjacent merges. Each iteration is bounded.

        Merged link properties:
        - length: sum (exact)
        - maxspeed: length-weighted harmonic mean (exact FFT)
        - lanes: min (bottleneck capacity)
        - geometry: concatenated LineStrings (exact)
        """
        def _num(val, default=25):
            v = val[0] if isinstance(val, list) else val
            try: return int(str(v).split()[0]) if str(v).split()[0].isdigit() else default
            except: return default

        def _ln_num(val, default=1):
            v = val[0] if isinstance(val, list) else val
            try: return int(str(v))
            except: return default

        def _build_merged(in_e, out_e):
            ie = in_e[3]; oe = out_e[3]
            gi = ie.get('geometry'); go = oe.get('geometry')
            geom = LineString(list(gi.coords)+list(go.coords)[1:]) if (gi is not None and go is not None) else (gi or go)
            li = float(ie.get('length',0)); lo = float(oe.get('length',0))
            tot = li + lo
            na = dict(ie)
            na['geometry'] = geom; na['length'] = tot
            if tot > 0 and li >= 0 and lo >= 0:
                si = max(_num(ie.get('maxspeed','25 mph')), 1)
                so = max(_num(oe.get('maxspeed','25 mph')), 1)
                na['maxspeed'] = f'{tot/(li/si + lo/so):.0f} mph' if (li/si+lo/so) > 0 else '25 mph'
            na['lanes'] = str(min(_ln_num(ie.get('lanes','1')), _ln_num(oe.get('lanes','1'))))
            na.pop('name', None); na.pop('ref', None)
            return na

        total_merged = 0
        for iteration in range(3):
            # Find all current pass-through nodes
            to_merge = []
            for n in list(self.graph.nodes):
                nbrs = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                if len(nbrs) == 2:
                    n1, n2 = list(nbrs)
                    if n1 != n and n2 != n:
                        to_merge.append((n, n1, n2))
            if not to_merge:
                break

            iter_merged = 0
            for n, n1, n2 in to_merge:
                if n not in self.graph.nodes:
                    continue

                n1_to_n = [e for e in self.graph.in_edges(n,  data=True, keys=True) if e[0] == n1]
                n_to_n2 = [e for e in self.graph.out_edges(n, data=True, keys=True) if e[1] == n2]
                n2_to_n = [e for e in self.graph.in_edges(n,  data=True, keys=True) if e[0] == n2]
                n_to_n1 = [e for e in self.graph.out_edges(n, data=True, keys=True) if e[1] == n1]

                if not n1_to_n or not n_to_n2:
                    # Try swapped direction: n2→n→n1
                    if n2_to_n and n_to_n1:
                        n1, n2 = n2, n1
                        n1_to_n, n_to_n2 = n2_to_n, n_to_n1
                        n2_to_n = [e for e in self.graph.in_edges(n, data=True, keys=True) if e[0] == n2]
                        n_to_n1 = [e for e in self.graph.out_edges(n, data=True, keys=True) if e[1] == n1]
                        if not n1_to_n or not n_to_n2:
                            continue
                    else:
                        continue

                new_fwd = _build_merged(n1_to_n[0], n_to_n2[0])
                new_rev = _build_merged(n2_to_n[0], n_to_n1[0]) if n2_to_n and n_to_n1 else None

                self.graph.remove_node(n)
                self.graph.add_edge(n1, n2, **new_fwd)
                if new_rev:
                    self.graph.add_edge(n2, n1, **new_rev)
                iter_merged += 1

            total_merged += iter_merged

        if total_merged > 0:
            print(f"  Merged {total_merged} pass-through nodes.")

    def _suppress_t_junctions(self, threshold=200):
        """Suppress short cross-streets at T-junctions, then merge and prune.

        At each T-junction (undirected-degree=3), removes the shortest edge
        if it's under the threshold. This converts T-junctions into pass-through
        nodes, which get merged away. Both endpoints must have degree > 2 to
        ensure connectivity is preserved.
        """
        def _num(val, d=25):
            v = val[0] if isinstance(val, list) else val
            v = str(v)
            try: return int(v.split()[0]) if v.split()[0].isdigit() else d
            except: return d

        removed = 0
        for iteration in range(5):
            to_remove = []
            for n in list(self.graph.nodes):
                nb = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                if len(nb) != 3:
                    continue
                edges = []
                for nbr in nb:
                    for e in self.graph.in_edges(n, data=True, keys=True):
                        if e[0] == nbr:
                            edges.append((float(e[3].get('length', 0)),
                                          (e[0], e[1], e[2]), nbr))
                    for e in self.graph.out_edges(n, data=True, keys=True):
                        if e[1] == nbr:
                            edges.append((float(e[3].get('length', 0)),
                                          (e[0], e[1], e[2]), nbr))
                if len(edges) < 3:
                    continue
                edges.sort(key=lambda x: x[0])
                if edges[0][0] >= threshold:
                    continue
                # Other endpoint must have degree > 2 (not a dead-end)
                other_nbr = edges[0][2]
                other_udeg = len(set(self.graph.successors(other_nbr))
                                 | set(self.graph.predecessors(other_nbr)))
                if other_udeg <= 2:
                    continue
                # Node n's other 2 edges must go to 2 different neighbors
                other_dests = set()
                for ed in edges[1:]:
                    other_dests.add(ed[2])
                if len(other_dests) != 2:
                    continue
                to_remove.append(edges[0][1])

            if not to_remove:
                break

            # Connectivity-aware removal: only suppress if SCC stays intact
            for key in set(to_remove):
                # Save edge data before removal
                saved_data = None
                try:
                    saved_data = dict(self.graph.edges[key])
                except Exception:
                    pass
                scc_before = len(list(nx.strongly_connected_components(self.graph)))
                try:
                    self.graph.remove_edge(key[0], key[1], key=key[2])
                    scc_after = len(list(nx.strongly_connected_components(self.graph)))
                    if scc_after > scc_before:
                        # This edge was a bridge — undo the removal
                        if saved_data:
                            self.graph.add_edge(key[0], key[1], key=key[2], **saved_data)
                        continue
                    removed += 1
                except Exception:
                    pass

            # Re-merge + re-prune after removing edges
            self._merge_degree2_chains()
            self._prune_dead_ends_graph()

        if removed > 0:
            print(f"  Suppressed {removed} short cross-streets "
                  f"(< {threshold}m at T-junctions).")

    def _remove_chain_fragments(self):
        """Remove SCCs that contain only chain nodes (no intersections).

        After T-junction suppression, some small fragments may become disconnected.
        This removes only those fragments that have no degree-3+ nodes (i.e.,
        they're just chains/dead-ends, not genuine sub-networks).
        """
        sccs = list(nx.strongly_connected_components(self.graph))
        if len(sccs) <= 1:
            return
        largest = max(sccs, key=len)
        to_remove = []
        for scc in sccs:
            if scc is largest:
                continue
            sub = self.graph.subgraph(scc)
            has_intersection = any(
                len(set(sub.successors(n)) | set(sub.predecessors(n))) >= 3
                for n in scc
            )
            if not has_intersection:
                to_remove.append(scc)
        if to_remove:
            removed = sum(len(s) for s in to_remove)
            for scc in to_remove:
                self.graph.remove_nodes_from(scc)
            print(f"  Removed {removed} orphan chain nodes "
                  f"({len(to_remove)} fragments).")

    def _prune_dead_ends_graph(self):
        """Iteratively remove degree-1 and undirected-degree-1 nodes from the OSM graph.

        Runs BEFORE rearrange_data so subsequent rearrange naturally gives contiguous IDs.
        """
        removed = 0
        while True:
            to_remove = set()
            for n in self.graph.nodes:
                deg = self.graph.in_degree(n) + self.graph.out_degree(n)
                if deg <= 1:
                    to_remove.add(n)
                    continue
                neighbors = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
                if len(neighbors) == 1:
                    to_remove.add(n)
            if not to_remove:
                break
            self.graph.remove_nodes_from(to_remove)
            removed += len(to_remove)
        if removed > 0:
            print(f"  Pruned {removed} leaf nodes from graph.")

    def _keep_largest_scc(self):
        """Keep only the largest strongly connected component of the OSM graph.

        Filtering to major roads can create one-way segments that are unreachable
        in one direction. This ensures every remaining node is reachable from
        every other node in both directions.
        """
        sccs = list(nx.strongly_connected_components(self.graph))
        if len(sccs) <= 1:
            return
        largest = max(sccs, key=len)
        removed = len(self.graph.nodes) - len(largest)
        if removed == 0:
            return
        self.graph = self.graph.subgraph(largest).copy()
        print(f"  Kept largest SCC: {len(self.graph.nodes)} nodes, "
              f"{len(self.graph.edges)} edges ({removed} nodes removed)")

    def _merge_junctions(self, threshold=80):
        """Merge close-node clusters at interchanges into single junction nodes.

        Finds connected components of edges shorter than threshold. Each
        component with 3+ nodes is a "junction cluster" — ramp connections
        at an interchange that should be a single routing node. Merges each
        cluster to a centroid node, preserving all external edges.

        This is mathematically rigorous: the junction subsumes internal
        navigation (ramp-to-ramp), while external edges retain their
        original physical properties (FFT, capacity, geometry).
        """
        # Build subgraph of short edges
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(self.graph.nodes)
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            length = float(d.get('length', 0))
            if 0 < length < threshold:
                cluster_graph.add_edge(u, v)

        clusters = list(nx.connected_components(cluster_graph))
        junctions = [c for c in clusters if len(c) >= 3]
        if not junctions:
            return

        # Get node positions from nodes_df (built during rearrange_data or from graph)
        node_lonlat = {}
        for n in self.graph.nodes:
            lon = self.graph.nodes[n].get('x') or self.graph.nodes[n].get('lon')
            lat = self.graph.nodes[n].get('y') or self.graph.nodes[n].get('lat')
            if lon is not None and lat is not None:
                node_lonlat[n] = (float(lon), float(lat))

        total_merged = 0
        for jid, cluster in enumerate(junctions):
            nodes = list(cluster)
            lons = [node_lonlat[n][0] for n in nodes if n in node_lonlat]
            lats = [node_lonlat[n][1] for n in nodes if n in node_lonlat]
            if not lons:
                continue
            cx, cy = np.mean(lons), np.mean(lats)
            new_id = f'_J{jid}'
            self.graph.add_node(new_id, x=cx, y=cy, lon=cx, lat=cy)

            for n in nodes:
                if n not in self.graph.nodes:
                    continue
                # Redirect inbound edges from outside
                for u, v, k, d in list(self.graph.in_edges(n, keys=True, data=True)):
                    if u not in cluster:
                        self.graph.add_edge(u, new_id, **dict(d))
                # Redirect outbound edges to outside
                for u, v, k, d in list(self.graph.out_edges(n, keys=True, data=True)):
                    if v not in cluster:
                        self.graph.add_edge(new_id, v, **dict(d))
                self.graph.remove_node(n)
            total_merged += len(nodes)

        # Clean self-loops
        for n in list(self.graph.nodes):
            while self.graph.has_edge(n, n):
                self.graph.remove_edge(n, n)

        if total_merged > 0:
            print(f"  Merged {total_merged} close nodes into {len(junctions)} junctions "
                  f"(< {threshold}m).")
            self._keep_largest_scc()
            self._merge_degree2_chains()
            self._prune_dead_ends_graph()

    def _contract_short_edges(self, threshold=100):
        """Contract edges shorter than threshold, merging endpoints into one node.

        Uses networkx.contracted_edge — preserves all other edges and attributes.
        This is the principled way to eliminate node clusters at complex
        junctions where multiple nodes are connected by very short links.
        """
        contracted = 0
        for _ in range(20):
            short = sorted(
                [(float(d.get('length', 0)), u, v)
                 for u, v, k, d in self.graph.edges(keys=True, data=True)
                 if 0 < float(d.get('length', 0)) < threshold]
            )
            if not short:
                break
            _, u, v = short[0]
            self.graph = nx.contracted_edge(self.graph, (u, v), self_loops=False, copy=True)
            # Remove self-loops created by contraction
            for n in list(self.graph.nodes):
                while self.graph.has_edge(n, n):
                    self.graph.remove_edge(n, n)
            contracted += 1
        if contracted > 0:
            print(f"  Contracted {contracted} short edges (< {threshold}m).")

    def rearrange_data(self):
        nodesOX, edgesOX = ox.graph_to_gdfs(self.graph)

        raw_nodes = nodesOX.copy().reset_index()
        raw_nodes['node_id'] = np.arange(raw_nodes.shape[0])
        raw_nodes['lon'] = raw_nodes['x']
        raw_nodes['lat'] = raw_nodes['y']
        raw_nodes['node_osmid'] = raw_nodes['osmid'].astype(object)
        raw_nodes['type'] = 'real'
        self.osmid_to_nid_dict = {r.osmid: r.node_id for r in raw_nodes.itertuples()}
        self.nid_to_osmid_dict = {r.node_id: r.osmid for r in raw_nodes.itertuples()}

        raw_edges = edgesOX.copy().reset_index()
        raw_edges['link_id'] = np.arange(raw_edges.shape[0])
        raw_edges['start_node_id'] = raw_edges['u'].map(self.osmid_to_nid_dict)
        raw_edges['end_node_id'] = raw_edges['v'].map(self.osmid_to_nid_dict)
        raw_edges['start_osmid'] = raw_edges['u'].astype(object)
        raw_edges['end_osmid'] = raw_edges['v'].astype(object)
        raw_edges['type'] = raw_edges['highway']
        raw_edges['length'] = raw_edges['length'].astype(float)
        raw_edges['lanes'] = raw_edges['lanes'].fillna('1')
        raw_edges['lanes'] = raw_edges['lanes'].apply(
            lambda x: str(x[0]) if isinstance(x, list) else str(x)
        )
        raw_edges['maxmph'] = raw_edges['maxspeed'].fillna('25 mph')
        raw_edges['maxmph'] = [
            float(list(filter(lambda x: x.isdigit(), s.split()))[0])
            for s in raw_edges['maxmph'].apply(
                lambda x: str(x[0]) if isinstance(x, list) else str(x)
            ).tolist()
        ]
        raw_edges['geometry'] = raw_edges['geometry'].apply(wkt.dumps)
        raw_edges['capacity'] = raw_edges['lanes'].astype(int) * 1000

        self.nodes = raw_nodes.drop(['x', 'y', 'street_count', 'geometry', 'highway', 'osmid'], axis=1)
        self.edges = raw_edges[['link_id', 'start_node_id', 'end_node_id', 'type',
                                'length', 'maxmph', 'lanes', 'capacity',
                                'start_osmid', 'end_osmid', 'geometry']]

    def set_exit(self, random_flag, des_ex_id):
        exit_id = self.nodes.sample(n=1).node_id.values[0] if random_flag else des_ex_id
        self.exit_node_id = exit_id
        self.nodes.loc[self.nodes['node_id'] == exit_id, 'type'] = 'vn_sink'
        self.nodes.loc[self.nodes['node_id'] == exit_id, 'node_osmid'] = 'vn_sink'
        self.edges.loc[self.edges['end_node_id'] == exit_id, 'type'] = 'vl_out'
        self.edges.loc[self.edges['end_node_id'] == exit_id, 'end_osmid'] = 'vn_sink'
        self.edges.reset_index()

    def create_demand(self, cars_per_node):
        rows = []
        for _, row in self.nodes.iterrows():
            for _ in range(cars_per_node):
                rows.append({
                    'origin_node_id': row['node_id'],
                    'destin_node_id': self.exit_node_id,
                    'origin_osmid': row['node_osmid'],
                    'destin_osmid': 'vn_sink',
                })
        self.demand = pd.DataFrame(rows)

    def create_demand_with_orig_dest(self, num_of_vehs, orig, dest):
        orig_osmid = self.nodes.loc[self.nodes.node_id == orig, 'node_osmid'].values[0]
        rows = [{
            'origin_node_id': orig,
            'destin_node_id': dest,
            'origin_osmid': orig_osmid,
            'destin_osmid': 'vn_sink',
        }] * num_of_vehs
        self.demand = pd.DataFrame(rows)

    def save_data(self, save_dir=None):
        save_name = self.name.lower().replace(' ', '_').replace(',', '')
        cwd = save_dir if save_dir else os.getcwd()
        self.demand.to_csv(os.path.join(cwd, f'traffic_inputs_{save_name}_od.csv'), index=False)
        self.nodes.to_csv(os.path.join(cwd, f'traffic_inputs_{save_name}_nodes.csv'), index=False)
        self.edges.to_csv(os.path.join(cwd, f'traffic_inputs_{save_name}_edges.csv'), index=False)

    def plot_links_and_nodes(self, link_ids, node_ids=None, origin=None, destination=None):
        link_geos = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.edges.iloc[link_ids].geometry.tolist()))
        all_edges = gpd.GeoDataFrame(geometry=gpd.GeoSeries.from_wkt(self.edges['geometry'].tolist()))
        all_nodes = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
            x=self.nodes['lon'].tolist(), y=self.nodes['lat'].tolist()))

        fig, ax = plt.subplots(1, 1)
        fig.patch.set_facecolor('white')
        ax.set_axis_off()
        all_nodes.plot(ax=ax, color='black', alpha=0.7)
        all_edges.plot(ax=ax, color='black', alpha=0.7)
        link_geos.plot(ax=ax, color='g', zorder=2, linewidth=8.0, alpha=0.8)

        if node_ids is not None:
            node_geos = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
                x=self.nodes.loc[node_ids, 'lon'].tolist(),
                y=self.nodes.loc[node_ids, 'lat'].tolist()))
            node_geos.plot(ax=ax, zorder=2, color='g', markersize=400, marker='$◯$')

        if origin is not None and destination is not None:
            offset = 0.0003
            for nid, label, color in [(origin, 'O', 'b'), (destination, 'D', 'r')]:
                x, y = self.nodes.loc[nid, 'lon'], self.nodes.loc[nid, 'lat']
                ax.text(x + offset, y + offset, label, fontsize=14, fontweight='bold', color=color)
        return fig, ax
