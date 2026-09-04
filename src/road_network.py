import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import LineString
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from src.graph_cache import get_graph
from src.network_artifact import load_network_artifact, write_network_artifact
from src.network_pruning import (
    consolidate_intersections,
    filter_highways,
    largest_component,
    parse_lanes,
    parse_speed_kph,
    prepare_source_graph,
    project_graph,
    topology_simplify,
)


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
        self.stage_maps = {}    # {stage_name: {'nodes_xy': [...], 'edges_pairs': [...]}}
        self.cache_metadata = {}

    def load_artifact(self, artifact_dir):
        """Load a canonical network artifact without downloading or pruning."""
        nodes, edges, manifest = load_network_artifact(artifact_dir)
        graph = nx.MultiDiGraph(crs="EPSG:4326")
        node_osmids = {}
        for row in nodes.itertuples():
            node_id = int(row.node_id)
            osmid = getattr(row, "node_osmid", node_id)
            node_osmids[node_id] = osmid
            graph.add_node(osmid, osmid=osmid, x=float(row.lon), y=float(row.lat))
        for row in edges.itertuples():
            source_ids = getattr(row, "source_edge_ids", "[]")
            if isinstance(source_ids, str):
                try:
                    source_ids = tuple(json.loads(source_ids))
                except (TypeError, ValueError):
                    source_ids = (source_ids,)
            geometry = getattr(row, "geometry", None)
            if isinstance(geometry, str):
                geometry = wkt.loads(geometry)
            maxmph = float(getattr(row, "maxmph", 25.0))
            highway = str(getattr(row, "type", "primary")).split("|")
            graph.add_edge(
                node_osmids[int(row.start_node_id)],
                node_osmids[int(row.end_node_id)],
                key=getattr(row, "edge_key", 0),
                link_id=int(row.link_id),
                geometry=geometry,
                length=float(row.length),
                travel_time=float(getattr(
                    row, "travel_time",
                    float(row.length) / (maxmph * 1609.344 / 3600.0),
                )),
                speed_kph=maxmph * 1.609344,
                maxspeed=f"{maxmph} mph",
                highway=highway[0] if len(highway) == 1 else tuple(highway),
                lanes_numeric=float(getattr(row, "lanes", 1)),
                lanes=str(getattr(row, "lanes", 1)),
                source_edge_ids=tuple(source_ids),
            )
        self.graph = graph
        self.nodes = nodes.sort_values("node_id").reset_index(drop=True)
        self.edges = edges.sort_values("link_id").reset_index(drop=True)
        self.osmid_to_nid_dict = {
            osmid: node_id for node_id, osmid in node_osmids.items()
        }
        self.nid_to_osmid_dict = dict(node_osmids)
        self.stage_counts = dict(manifest.get("stage_counts") or {
            "loaded_artifact": {"nodes": len(nodes), "edges": len(edges)},
        })
        self.stage_maps = {"loaded_artifact": self._snapshot_map()}
        return manifest

    def _snapshot_map(self):
        """Capture node coords, edge pairs, and edge geometry polylines."""
        nids = sorted(self.graph.nodes, key=str)
        nodes_xy = [(self.graph.nodes[n].get('x', 0), self.graph.nodes[n].get('y', 0))
                    for n in nids]
        pos = {n: nodes_xy[i] for i, n in enumerate(nids)}
        edges_pairs = []
        edges_geom = []
        edge_keys = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            edges_pairs.append((u, v))
            edge_keys.append(k)
            geom = d.get('geometry')
            if geom is not None and hasattr(geom, 'coords'):
                coords = [(c[0], c[1]) for c in geom.coords]
            elif u in pos and v in pos:
                coords = [pos[u], pos[v]]
            else:
                coords = []
            edges_geom.append(coords)
        return {'_node_ids': nids, 'nodes_xy': nodes_xy,
                'edges_pairs': edges_pairs, 'edges_geom': edges_geom,
                'edge_keys': edge_keys,
                '_n': len(nids), '_e': len(edges_pairs)}

    def _keep_largest_scc(self):
        """Keep only the largest strongly connected component."""
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

    def get_map(self, no_lat, so_lat, east_long, west_long,
                highway_types=None, merge_chains=True, contract_threshold=30,
                prune_dead_ends=False, suppress_t_junctions=False,
                apply_cleaning=True, intersection_tolerance=0,
                cache_policy="reuse"):
        """Download OSM graph and apply topology-preserving cleaning.

        Stages:
          01_downloaded_unsimplified -- raw, unsimplified OSM graph
          02_highway_filtered        -- requested highway classes
          03_topology_simplified     -- true nondecision nodes removed
          04_intersections_consolidated -- optional metric consolidation
          05_largest_wcc             -- diagnostic weak component
          06_final_scc               -- model-ready directed component

        Args:
            highway_types: OSM highway filter, e.g. ['motorway','trunk',...]
            merge_chains: Enable topology-aware pass-through simplification.
            contract_threshold: Deprecated and ignored. It previously selected
                unsafe connected-component contraction.
            prune_dead_ends: Iteratively remove directed source/sink dead ends
                before the final SCC extraction.
            suppress_t_junctions: Reserved compatibility option. T-junction
                suppression is not applied implicitly; callers must use an
                explicit contraction threshold instead.
            apply_cleaning: If false, retain the filtered graph before the
                final component extraction.
            intersection_tolerance: OSMnx node-buffer radius in metres. Zero
                disables proximity consolidation.
        """
        if suppress_t_junctions:
            raise ValueError(
                'suppress_t_junctions is not implemented; use contract_threshold '
                'and merge_chains explicitly'
            )
        coords = (west_long, so_lat, east_long, no_lat)
        if cache_policy not in {"reuse", "refresh", "require"}:
            raise ValueError("cache_policy must be reuse, refresh, or require")
        downloaded, self.cache_metadata = get_graph(
            coords,
            highway_types=highway_types,
            simplify=False,
            retain_all=True,
            force_refresh=cache_policy == "refresh",
            require_cached=cache_policy == "require",
            return_metadata=True,
        )
        downloaded = prepare_source_graph(downloaded)
        self.graph = downloaded
        self.stage_counts['01_downloaded_unsimplified'] = {
            'nodes': len(self.graph), 'edges': self.graph.number_of_edges(),
        }
        self.stage_maps['01_downloaded_unsimplified'] = self._snapshot_map()

        if highway_types:
            self.graph = filter_highways(self.graph, highway_types)
        self.stage_counts['02_highway_filtered'] = {
            'nodes': len(self.graph), 'edges': self.graph.number_of_edges(),
        }
        self.stage_maps['02_highway_filtered'] = self._snapshot_map()

        if apply_cleaning and merge_chains:
            self.graph = topology_simplify(self.graph)
        self.stage_counts['03_topology_simplified'] = {
            'nodes': len(self.graph), 'edges': self.graph.number_of_edges(),
        }
        self.stage_maps['03_topology_simplified'] = self._snapshot_map()

        if apply_cleaning and float(intersection_tolerance) > 0:
            projected = project_graph(self.graph)
            projected, _, _ = consolidate_intersections(
                projected, float(intersection_tolerance)
            )
            self.graph = project_graph(projected, to_crs='EPSG:4326')
        self.stage_counts['04_intersections_consolidated'] = {
            'nodes': len(self.graph), 'edges': self.graph.number_of_edges(),
        }
        self.stage_maps['04_intersections_consolidated'] = self._snapshot_map()

        if prune_dead_ends:
            self._prune_dead_ends()
        wcc = largest_component(self.graph, strong=False)
        self.stage_counts['05_largest_wcc'] = {
            'nodes': len(wcc), 'edges': wcc.number_of_edges(),
        }
        original_graph = self.graph
        self.graph = wcc
        self.stage_maps['05_largest_wcc'] = self._snapshot_map()
        self.graph = largest_component(original_graph, strong=True)
        self.stage_counts['06_final_scc'] = {
            'nodes': len(self.graph), 'edges': self.graph.number_of_edges(),
        }
        self.stage_maps['06_final_scc'] = self._snapshot_map()
        self.rearrange_data()

    def _prune_dead_ends(self):
        """Remove directed source/sink dead ends before SCC extraction.

        This is intentionally conservative and deterministic.  The final SCC
        pass remains mandatory, so disabling this option never leaves a graph
        with disconnected optimization components.
        """
        removed = 0
        while self.graph.number_of_nodes():
            dead = sorted(
                [n for n in self.graph.nodes
                 if self.graph.in_degree(n) == 0 or self.graph.out_degree(n) == 0],
                key=str,
            )
            if not dead:
                break
            self.graph.remove_nodes_from(dead)
            removed += len(dead)
        if removed:
            print(f"  Pruned {removed} directed dead-end nodes.")

    def _filter_by_highway(self, highway_types):
        """Remove edges whose highway tag is not in the given list."""
        if highway_types:
            self.graph = filter_highways(self.graph, highway_types)

    def _merge_degree2_chains(self):
        """Merge all pass-through nodes (undirected-degree = 2).

        A node with exactly 2 unique neighbors is a pass-through -- traffic
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

        def _build_merged(in_e, out_e, mid_xy=None):
            ie = in_e[3]; oe = out_e[3]
            gi = ie.get('geometry'); go = oe.get('geometry')
            if gi is not None and go is not None:
                if mid_xy:
                    inc = list(gi.coords); inc[-1] = mid_xy
                    out = list(go.coords); out[0] = mid_xy
                    geom = LineString(inc + out[1:])
                else:
                    geom = LineString(list(gi.coords) + list(go.coords)[1:])
            elif gi is not None:
                if mid_xy:
                    inc = list(gi.coords); inc[-1] = mid_xy
                    geom = LineString(inc)
                else:
                    geom = gi
            elif go is not None:
                if mid_xy:
                    out = list(go.coords); out[0] = mid_xy
                    geom = LineString(out)
                else:
                    geom = go
            else:
                geom = None
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

                # Get pass-through node position for geometry fallback
                n_x = self.graph.nodes[n].get('x')
                n_y = self.graph.nodes[n].get('y')
                mid_xy = (float(n_x), float(n_y)) if n_x is not None and n_y is not None else None

                self.graph.remove_node(n)
                # Merge all parallel edge pairs to preserve distinct paths
                for ie in n1_to_n:
                    for oe in n_to_n2:
                        merged = _build_merged(ie, oe, mid_xy=mid_xy)
                        self.graph.add_edge(n1, n2, **merged)
                for ie in n2_to_n:
                    for oe in n_to_n1:
                        merged = _build_merged(ie, oe, mid_xy=mid_xy)
                        self.graph.add_edge(n2, n1, **merged)
                iter_merged += 1

            total_merged += iter_merged

        if total_merged > 0:
            print(f"  Merged {total_merged} pass-through nodes.")

    def _merge_junctions(self, threshold=30):
        """Merge close-node clusters at interchanges into single junction nodes.

        Finds connected components of edges shorter than *threshold*. Each
        component with 3+ nodes is a "junction cluster" -- ramp connections
        at an interchange that should be a single routing node. Merges each
        cluster to a centroid node, preserving external edge properties
        (FFT, capacity, original road geometry snapped to the centroid).

        Internal short edges connecting cluster members are subsumed --
        the centroid already represents the entire interchange.
        """
        cluster_graph = nx.Graph()
        cluster_graph.add_nodes_from(self.graph.nodes)
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            length = float(d.get('length', 0))
            if 0 < length < threshold:
                cluster_graph.add_edge(u, v)

        clusters = list(nx.connected_components(cluster_graph))
        junctions = [c for c in clusters if len(c) >= 2]
        if not junctions:
            return

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
            new_id = f'_C{jid}'
            self.graph.add_node(new_id, x=cx, y=cy, lon=cx, lat=cy)

            for n in nodes:
                if n not in self.graph.nodes:
                    continue
                for u, v, k, d in list(self.graph.in_edges(n, keys=True, data=True)):
                    if u not in cluster:
                        nd = dict(d)
                        self._snap_geom_to_node(nd, u, new_id, (cx, cy), snap_end=True)
                        self.graph.add_edge(u, new_id, **nd)
                for u, v, k, d in list(self.graph.out_edges(n, keys=True, data=True)):
                    if v not in cluster:
                        nd = dict(d)
                        self._snap_geom_to_node(nd, new_id, v, (cx, cy), snap_end=False)
                        self.graph.add_edge(new_id, v, **nd)
                self.graph.remove_node(n)
            total_merged += len(nodes)

        for n in list(self.graph.nodes):
            while self.graph.has_edge(n, n):
                self.graph.remove_edge(n, n)

        if total_merged > 0:
            print(f"  Merged {total_merged} close nodes into {len(junctions)} junctions "
                  f"(< {threshold}m).")

    def _snap_geom_to_node(self, edge_data, src, dst, mid_xy, snap_end=True):
        """Snap edge geometry start or end to the centroid position."""
        geom = edge_data.get('geometry')
        if geom is not None and hasattr(geom, 'coords'):
            coords = list(geom.coords)
            if len(coords) >= 2:
                idx = -1 if snap_end else 0
                coords[idx] = (mid_xy[0], mid_xy[1])
                edge_data['geometry'] = LineString(coords)

    def rearrange_data(self):
        nodesOX, edgesOX = ox.graph_to_gdfs(self.graph)

        raw_nodes = pd.DataFrame(nodesOX.copy())
        if 'osmid' not in raw_nodes.columns:
            raw_nodes['osmid'] = raw_nodes.index
        raw_nodes = raw_nodes.reset_index(drop=True)
        raw_nodes['_stable_node_key'] = raw_nodes['osmid'].map(str)
        raw_nodes = raw_nodes.sort_values('_stable_node_key', kind='mergesort').reset_index(drop=True)
        raw_nodes['node_id'] = np.arange(raw_nodes.shape[0], dtype=int)
        raw_nodes['lon'] = raw_nodes['x']
        raw_nodes['lat'] = raw_nodes['y']
        raw_nodes['node_osmid'] = raw_nodes['osmid'].astype(object)
        raw_nodes['type'] = 'real'
        self.osmid_to_nid_dict = {r.osmid: r.node_id for r in raw_nodes.itertuples()}
        self.nid_to_osmid_dict = {r.node_id: r.osmid for r in raw_nodes.itertuples()}

        raw_edges = pd.DataFrame(edgesOX.copy().reset_index())
        for col in ('u', 'v', 'key'):
            if col not in raw_edges.columns:
                raw_edges[col] = ''
        raw_edges['_stable_edge_key'] = raw_edges.apply(
            lambda row: (str(row['u']), str(row['v']), str(row['key'])), axis=1
        )
        raw_edges = raw_edges.sort_values('_stable_edge_key', kind='mergesort').reset_index(drop=True)
        raw_edges['link_id'] = np.arange(raw_edges.shape[0], dtype=int)
        raw_edges['start_node_id'] = raw_edges['u'].map(self.osmid_to_nid_dict)
        raw_edges['end_node_id'] = raw_edges['v'].map(self.osmid_to_nid_dict)
        raw_edges['start_osmid'] = raw_edges['u'].astype(object)
        raw_edges['end_osmid'] = raw_edges['v'].astype(object)
        raw_edges['edge_key'] = raw_edges['key'].astype(object)
        raw_edges['type'] = raw_edges['highway'].apply(
            lambda value: '|'.join(map(str, value))
            if isinstance(value, (list, tuple, set)) else str(value)
        )
        raw_edges['length'] = raw_edges['length'].astype(float)
        lane_source = 'lanes_numeric' if 'lanes_numeric' in raw_edges else 'lanes'
        raw_edges['lanes'] = raw_edges[lane_source].apply(parse_lanes).round().astype(int)
        if 'speed_kph' in raw_edges:
            raw_edges['maxmph'] = raw_edges['speed_kph'].astype(float) / 1.609344
        else:
            raw_edges['maxmph'] = raw_edges.apply(
                lambda row: parse_speed_kph(row.get('maxspeed'), row.get('highway')) / 1.609344,
                axis=1,
            )
        if 'travel_time' not in raw_edges:
            raw_edges['travel_time'] = raw_edges['length'] / (
                raw_edges['maxmph'] * 1609.344 / 3600.0
            )
        raw_edges['geometry'] = raw_edges['geometry'].apply(wkt.dumps)
        raw_edges['capacity'] = raw_edges['lanes'] * 1000
        if 'source_edge_ids' not in raw_edges:
            raw_edges['source_edge_ids'] = raw_edges.apply(
                lambda row: (f"{row['u']}|{row['v']}|{row['key']}",), axis=1
            )
        raw_edges['source_edge_ids'] = raw_edges['source_edge_ids'].apply(
            lambda value: json.dumps(list(value) if isinstance(value, (list, tuple, set)) else [value])
        )

        drop_node_columns = [c for c in ['x', 'y', 'street_count', 'geometry', 'highway', 'osmid', '_stable_node_key'] if c in raw_nodes]
        self.nodes = raw_nodes.drop(drop_node_columns, axis=1)
        self.edges = raw_edges[['link_id', 'start_node_id', 'end_node_id', 'type',
                                'length', 'maxmph', 'lanes', 'capacity',
                                'travel_time', 'source_edge_ids',
                                'start_osmid', 'end_osmid', 'edge_key', 'geometry']]
        self.nodes = self.nodes.sort_values('node_id').reset_index(drop=True)
        self.edges = self.edges.sort_values('link_id').reset_index(drop=True)

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

    def create_demand_for_classes(self, demand_classes):
        """Create an exact, jointly simulated demand table.

        ``demand_classes`` may contain :class:`DemandClass` instances or
        mappings with ``origin``, ``destination``, ``vehicle_type``/``type``,
        and ``demand`` fields.  Unlike ``add_charging_info`` this preserves
        F1/F2 counts for every OD pair and never samples vehicle types.
        """
        rows = []
        for item in demand_classes:
            if isinstance(item, dict):
                origin = int(item['origin'])
                destination = int(item['destination'])
                vehicle_type = str(item.get('vehicle_type', item.get('type'))).upper()
                demand = int(item['demand'])
            else:
                origin = int(item.origin)
                destination = int(item.destination)
                vehicle_type = str(item.vehicle_type).upper()
                demand = int(item.demand)
            if vehicle_type not in {'F1', 'F2'}:
                raise ValueError(f'Unsupported vehicle type: {vehicle_type}')
            for _ in range(demand):
                rows.append({
                    'origin_node_id': origin,
                    'destin_node_id': destination,
                    'origin_osmid': self.nid_to_osmid_dict.get(origin, origin),
                    'destin_osmid': self.nid_to_osmid_dict.get(destination, destination),
                    'is_EV': vehicle_type == 'F2',
                    'need_to_charge': vehicle_type == 'F2',
                    'current_charge': 0,
                    'target_charge': 100,
                    'go_to_station_id': np.nan,
                })
        self.demand = pd.DataFrame(rows)
        return self.demand

    def save_data(self, save_dir=None):
        save_name = self.name.lower().replace(' ', '_').replace(',', '')
        cwd = save_dir if save_dir else os.getcwd()
        self.demand.to_csv(os.path.join(cwd, f'traffic_inputs_{save_name}_od.csv'), index=False)
        self.nodes.to_csv(os.path.join(cwd, f'traffic_inputs_{save_name}_nodes.csv'), index=False)
        self.edges.to_csv(os.path.join(cwd, f'traffic_inputs_{save_name}_edges.csv'), index=False)

    def export_artifact(self, output_dir, source=None):
        """Serialize the exact cleaned graph consumed by all later stages."""
        if not hasattr(self.edges, 'columns') or 'link_id' not in self.edges.columns:
            raise ValueError('RoadNet must be rearranged before exporting an artifact')
        return write_network_artifact(
            self.nodes,
            self.edges,
            output_dir,
            stage_counts=self.stage_counts,
            stage_maps=self.stage_maps,
            source=source,
        )

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
