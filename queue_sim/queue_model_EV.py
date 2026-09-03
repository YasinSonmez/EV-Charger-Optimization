from shapely.wkt import loads
import random
from ctypes import c_double
from . import interface
import numpy as np
import pandas as pd

###################################################################### ADDITION STARTS

# Class for Electric Vehicle Charging Station
class EV_Charging_Station:
    def __init__(self, station_id, in_link_id, out_link_id, node_id, ent_ex_node_id, lon, lat, ent_capacity, charging_capacity, exit_capacity, cost, simulation=None): # Create EV_Charging_Station object
        # Attributes from inputs
        self.station_id = station_id # Id of the station
        self.in_link_id = in_link_id # Id of the virtual link going into the station
        self.out_link_id = out_link_id  # Id of the virtual link going out of the station
        self.node_id = node_id  # Id of the virtual node associated with the station
        self.ent_ex_node_id = ent_ex_node_id # Id of the virtual node that represents the station's entrance/exit
        self.lon = lon # Longitude of the station
        self.lat = lat # Lattitude of the station
        self.ent_capacity = ent_capacity # Capacity of the entrance queue (in num. of vehicles)
        self.charging_capacity = charging_capacity # Number of charger ports
        self.exit_capacity = exit_capacity # Capacity of the exit queue (in num. of vehicles)
        self.cost = cost # Cost of charging
        self.simulation = simulation

        # Derived attributes
        self.exit_queue = []
        self.charging_vehicles = []
        self.ent_queue = []

    def run_station_model(self,t_now):
        node_id_dict = self.simulation.all_nodes
        link_id_dict = self.simulation.all_links
        agent_id_dict = self.simulation.all_agents
        node2link_dict = self.simulation.node2link_dict

        # Perform actions for all vehicles that are currently charging
        for (agent_id,t_in_charging_port) in self.charging_vehicles:
            # If the exit queue has space and the agent has reached its target charge, then move it to the exit queue
            if agent_id_dict[agent_id].current_charge == agent_id_dict[agent_id].target_charge:
                if len(self.exit_queue) < self.exit_capacity:
                    self.charging_vehicles = [(i,j) for (i,j) in self.charging_vehicles if i != agent_id]
                    self.exit_queue.append((agent_id,0))
                    agent_id_dict[agent_id].need_to_charge = False
                    # Move agent without incrementing the route pointer (because the agent is still in the station)
                    agent_id_dict[agent_id].move_agent_without_pointer(t_now,link_id_dict[self.out_link_id].start_nid,link_id_dict[self.out_link_id].end_nid,'station_exit_queue')
                else:
                    charging_vehicles_ind = [tuples[0] for tuples in self.charging_vehicles].index(agent_id)
                    new_time_val = list(self.charging_vehicles[charging_vehicles_ind])[1] + 1  # Can not change values in a tuple, so we convert it to a list first
                    self.charging_vehicles[charging_vehicles_ind] = (agent_id, new_time_val)
            # If the agent hasn't reached its target charge, then increment current charge and time counter
            else:
                agent_id_dict[agent_id].current_charge += 1
                charging_vehicles_ind = [tuples[0] for tuples in self.charging_vehicles].index(agent_id)
                new_time_val = list(self.charging_vehicles[charging_vehicles_ind])[1]+1 # Can not change values in a tuple, so we convert it to a list first
                self.charging_vehicles[charging_vehicles_ind] = (agent_id,new_time_val)

        # Increment the time counter of all vehicles in the entrance queue, except the one that has been there the longest
        if len(self.ent_queue) > 0:
            for (agent_id, t_in_ent_queue) in self.ent_queue[1:]:
                ent_queue_vehicles_ind = [tuples[0] for tuples in self.ent_queue].index(agent_id)
                new_time_val = list(self.ent_queue[ent_queue_vehicles_ind])[1] + 1  # Can not change values in a tuple, so we convert it to a list first
                self.ent_queue[ent_queue_vehicles_ind] = (agent_id, new_time_val)
            # For the agent that has been in the entrance queue for the longest time: If there is an available port, then move it to a charging port, o.w. increment its time counter
            if self.charging_capacity > len(self.charging_vehicles):
                agent_id = self.ent_queue.pop(0)[0]
                self.charging_vehicles.append((agent_id,0))
                agent_id_dict[agent_id].move_agent_without_pointer(t_now, link_id_dict[self.out_link_id].start_nid,link_id_dict[self.out_link_id].start_nid, 'charging')
            else:
                agent_id = self.ent_queue[0][0]
                ent_queue_vehicles_ind = [tuples[0] for tuples in self.ent_queue].index(agent_id)
                new_time_val = list(self.ent_queue[ent_queue_vehicles_ind])[1] + 1  # Can not change values in a tuple, so we convert it to a list first
                self.ent_queue[ent_queue_vehicles_ind] = (agent_id, new_time_val)

        # Increment the time counter of all vehicles in the exit queue, except the one that has been there the longest
        for (agent_id, t_in_ex_queue) in self.exit_queue:
            ex_queue_vehicles_ind = [tuples[0] for tuples in self.exit_queue].index(agent_id)
            new_time_val = list(self.exit_queue[ex_queue_vehicles_ind])[1] + 1  # Can not change values in a tuple, so we convert it to a list first
            self.exit_queue[ex_queue_vehicles_ind] = (agent_id, new_time_val)

###################################################################### ADDITION ENDS

class Node:
    def __init__(self, node_id, lon, lat, ntype, osmid=None, simulation=None):
        self.nid = node_id
        self.lon = lon
        self.lat = lat
        self.ntype = ntype
        self.osmid = osmid
        self.simulation = simulation
        # derived
        self.in_links = {}  # {in_link_id: straight_ahead_out_link_id, ...}
        self.out_links = []
        self.go_vehs = []  # veh that moves in this time step
        self.status = None

    def create_virtual_node(self):
        return Node('vn_source_{}'.format(self.nid), self.lon+0.001, self.lat+0.001, 'vn_source', simulation=self.simulation)

    def create_virtual_link(self):
        return Link('vl_in_{}'.format(self.nid), 100, 0,
                    100000, ###################################################################### ADDITION INLINE
                    0, 100000, 'vl_in', 'vn_source_{}'.format(self.nid), self.nid, 'LINESTRING({} {}, {} {})'.format(self.lon+0.001, self.lat+0.001, self.lon, self.lat), simulation=self.simulation)

    def calculate_straight_ahead_links(self, node_id_dict=None, link_id_dict=None):
        for il in self.in_links.keys():
            x_start = node_id_dict[link_id_dict[il].start_nid].lon
            y_start = node_id_dict[link_id_dict[il].start_nid].lat
            in_vec = (self.lon-x_start, self.lat-y_start)
            sa_ol = None  # straight ahead out link
            ol_dir = 180
            for ol in self.out_links:
                x_end = node_id_dict[link_id_dict[ol].end_nid].lon
                y_end = node_id_dict[link_id_dict[ol].end_nid].lat
                out_vec = (x_end-self.lon, y_end-self.lat)
                dot = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1])
                det = (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
                new_ol_dir = np.arctan2(det, dot)*180/np.pi
                if abs(new_ol_dir) < ol_dir:
                    sa_ol = ol
                    ol_dir = new_ol_dir
            if (abs(ol_dir) <= 45) and link_id_dict[il].ltype[0:2] != 'vl':
                self.in_links[il] = sa_ol

    def find_go_vehs(self, go_link, agent_id_dict=None, node_id_dict=None, link_id_dict=None, node2link_dict=None):
        go_vehs_list = []
        incoming_lanes = int(np.floor(go_link.lanes))
        incoming_vehs = len(go_link.queue_veh)
        for ln in range(min(incoming_lanes, incoming_vehs)):
            agent_id = go_link.queue_veh[ln]
            try:
                agent_next_node, ol, agent_dir = agent_id_dict[agent_id].prepare_agent(
                    self.nid, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
            except AssertionError:
                print(agent_id, agent_id_dict[agent_id].status, agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle, self.nid, self.in_links.keys(
                ), go_link.lid, go_link.queue_veh, link_id_dict[node2link_dict[(agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle)]].queue_veh)
            go_vehs_list.append(
                [agent_id, agent_next_node, go_link.lid, ol, agent_dir])
        return go_vehs_list

    def non_conflict_vehs(self, t_now, link_id_dict=None, agent_id_dict=None, node2link_dict=None, node_id_dict=None):

        go_vehs = []
        # a primary direction
        in_links = [l for l in self.in_links.keys() if len(
            link_id_dict[l].queue_veh) > 0]

        if len(in_links) == 0:
            return go_vehs
        go_link = link_id_dict[random.choice(in_links)]
        go_vehs_list = self.find_go_vehs(go_link, agent_id_dict=agent_id_dict,
                                         link_id_dict=link_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
        go_vehs += go_vehs_list

        # a non-conflicting direction
        if (np.min([veh[-1] for veh in go_vehs_list]) < -45) or (go_link.ltype == 'vl_in'):
            return go_vehs  # no opposite veh allows to move if there is left turn veh in the primary direction; or if the primary incoming link is a virtual link
        if self.in_links[go_link.lid] is None:
            return go_vehs  # no straight ahead opposite links
        op_go_link = link_id_dict[self.in_links[go_link.lid]]
        link_id = node2link_dict.get(
            (op_go_link.end_nid, op_go_link.start_nid), None)
        if link_id not in link_id_dict:
            # straight ahead link is one way
            return go_vehs
        op_go_link = link_id_dict[link_id]
        op_go_vehs_list = self.find_go_vehs(op_go_link, agent_id_dict=agent_id_dict,
                                            link_id_dict=link_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
        # self.go_vehs += [veh for veh in op_go_vehs_list if veh[-1]>-45] ### only straight ahead or right turns allowed for vehicles from the opposite side
        go_vehs += [veh for veh in op_go_vehs_list if veh[-1] > -45]
        return go_vehs

    def run_node_model(self, t_now):

        node_id_dict = self.simulation.all_nodes
        link_id_dict = self.simulation.all_links
        agent_id_dict = self.simulation.all_agents
        node2link_dict = self.simulation.node2link_dict
        go_vehs = self.non_conflict_vehs(t_now=t_now, link_id_dict=link_id_dict,
                                         agent_id_dict=agent_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
        # Agent reaching destination
        for [agent_id, next_node, il, ol, _] in go_vehs:
            veh_len = agent_id_dict[agent_id].veh_len

            # arrival
            if (next_node is None) and (self.nid == agent_id_dict[agent_id].destin_nid):
                link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                agent_id_dict[agent_id].move_agent(
                    t_now, self.nid, None, 'arr')

###################################################################### ADDITION STARTS

                agent_id_dict[agent_id].arrival_time = t_now
                agent_id_dict[agent_id].travel_time = t_now - agent_id_dict[agent_id].dept_time

###################################################################### ADDITION ENDS

            # no storage capacity downstream
            elif link_id_dict[ol].st_c < veh_len:
                pass  # no blocking, as # veh = # lanes
            # inlink-sending, outlink-receiving both permits
            elif link_id_dict[il].ou_c >= 1 and link_id_dict[ol].in_c >= 1:
                # before move agent as it uses the old agent.cl_enter_time
                link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                agent_id_dict[agent_id].move_agent(
                    t_now, self.nid, next_node, 'flow')
                link_id_dict[ol].receive_veh(agent_id)
            # either inlink-sending or outlink-receiving or both exhaust
            else:
                # control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                toss_coin = random.choices(
                    [0, 1], weights=[1-control_cap, control_cap], k=1)
                if toss_coin[0]:  # vehicle can move
                    # before move agent as it uses the old agent.cl_enter_time
                    link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                    agent_id_dict[agent_id].move_agent(
                        t_now, self.nid, next_node, 'chance')
                    link_id_dict[ol].receive_veh(agent_id)
                else:  # even though vehicle cannot move, the remaining capacity needs to be adjusted
                    if link_id_dict[il].ou_c < link_id_dict[ol].in_c:
                        link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c-1)
                    elif link_id_dict[ol].in_c < link_id_dict[il].ou_c:
                        link_id_dict[ol].in_c = max(0, link_id_dict[ol].in_c-1)
                    else:
                        link_id_dict[il].ou_c -= 1  ###################################################################### WHY NOT max(0, link_id_dict[il].ou_c-1) ??????????????
                        link_id_dict[ol].in_c -= 1  ###################################################################### WHY NOT max(0, link_id_dict[ol].IN_c-1) ??????????????

###################################################################### ADDITION STARTS

    # find_go_vehs for station entrance/exit (find the vehicles to move in this timestep at a station entrance/exit node)
    def station_ent_ex_find_go_vehs(self, go_link, agent_id_dict=None, node_id_dict=None, link_id_dict=None,
                                    node2link_dict=None, station_id_dict=None):
        go_vehs_list = []

        # go_link is the link that exits the station
        if go_link.ltype == 'Out_Station':
            # Get the station associated with the link
            station_id = [s for s in station_id_dict.keys() if station_id_dict[s].out_link_id == go_link.lid][0]
            station = station_id_dict[station_id]
            if len(station.exit_queue) > 0:
                # At this point, we know that there are vehicles at the exit queue of the station and the exit road has a single lane
                # Pick the agent that has been at the queue for the longest time
                agent_id = station.exit_queue[0][0]
            else:
                # The exit queue is empty, so we return an empty list
                return go_vehs_list

            # Find the agent's next node
            try:
                agent_next_node, ol, agent_dir = agent_id_dict[agent_id].prepare_agent(
                    self.nid, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
            except AssertionError:
                print(agent_id, agent_id_dict[agent_id].status, agent_id_dict[agent_id].cls,
                      agent_id_dict[agent_id].cle, self.nid, self.in_links.keys(
                    ), go_link.lid, go_link.queue_veh, link_id_dict[
                          node2link_dict[(agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle)]].queue_veh)
            go_vehs_list.append([agent_id, agent_next_node, go_link.lid, ol, agent_dir])
            return go_vehs_list
        # In this case, the primary direction is not a station outlink, so we proceed as in the standard case
        else:
            incoming_lanes = int(np.floor(go_link.lanes))
            incoming_vehs = len(go_link.queue_veh)
            for ln in range(min(incoming_lanes, incoming_vehs)):
                agent_id = go_link.queue_veh[ln]
                try:
                    agent_next_node, ol, agent_dir = agent_id_dict[agent_id].prepare_agent(
                        self.nid, node2link_dict=node2link_dict, node_id_dict=node_id_dict)
                except AssertionError:
                    print(agent_id, agent_id_dict[agent_id].status, agent_id_dict[agent_id].cls,
                          agent_id_dict[agent_id].cle, self.nid, self.in_links.keys(
                        ), go_link.lid, go_link.queue_veh, link_id_dict[
                              node2link_dict[(agent_id_dict[agent_id].cls, agent_id_dict[agent_id].cle)]].queue_veh)
                go_vehs_list.append(
                    [agent_id, agent_next_node, go_link.lid, ol, agent_dir])
            return go_vehs_list

    # Find the vehicles that have non-conflicting directions at a station entrance/exit node
    def station_ent_ex_non_conflict_vehs(self, t_now, link_id_dict=None, agent_id_dict=None, node2link_dict=None,
                                         node_id_dict=None, station_id_dict=None):

        go_vehs = []
        inlink_station = []
        inlink_not_station = []

        # At a station entrance/exit, one of the in-links is the link that goes from the station to that node
        # This in-link has type 'Out_Station'
        for l in self.in_links.keys():
            if link_id_dict[l].ltype == 'Out_Station':
                # Check if the inlink from the station has queued vehicles (i.e. waiting to exit)
                station = [s for s in station_id_dict.keys() if
                           (station_id_dict[s].out_link_id == l) and (len(station_id_dict[s].exit_queue) > 0)]
                if station:
                    inlink_station.append(l)
            else:
                # Find the in-links that do not come from the station and has queued vehicles
                if len(link_id_dict[l].queue_veh) > 0:
                    inlink_not_station.append(l)

        # Combine in-links from the station and non-station links
        in_links = inlink_not_station + inlink_station
        # If the in_links is empty, then return an empty list
        if len(in_links) == 0:
            return go_vehs

        # Randomly, choose a primary direction from the list of inlinks
        go_link = link_id_dict[random.choice(in_links)]
        # Select the vehicles from the go_link
        go_vehs_list = self.station_ent_ex_find_go_vehs(go_link, agent_id_dict=agent_id_dict,
                                                        link_id_dict=link_id_dict, node2link_dict=node2link_dict,
                                                        node_id_dict=node_id_dict, station_id_dict=station_id_dict)
        go_vehs += go_vehs_list

        # Get vehicles from the non-conflicting directions
        if (np.min([veh[-1] for veh in go_vehs_list]) < -45) or (go_link.ltype == 'vl_in'):
            return go_vehs  # No opposite vehicle allows to move if the vehicle in the primary direction is making a left turn or the the primary incoming link is a virtual link
        if self.in_links[go_link.lid] is None:
            return go_vehs  # No straight ahead opposite links
        op_go_link = link_id_dict[self.in_links[go_link.lid]]
        link_id = node2link_dict.get((op_go_link.end_nid, op_go_link.start_nid), None)
        if link_id not in link_id_dict:
            return go_vehs  # Straight ahead link is one way
        op_go_link = link_id_dict[link_id]
        op_go_vehs_list = self.station_ent_ex_find_go_vehs(op_go_link, agent_id_dict=agent_id_dict,
                                                           link_id_dict=link_id_dict, node2link_dict=node2link_dict,
                                                           node_id_dict=node_id_dict, station_id_dict=station_id_dict)
        go_vehs += [veh for veh in op_go_vehs_list if
                    veh[-1] > -45]  # Only straight ahead or right turns allowed for vehicles from the opposite side
        return go_vehs

    # Run node model for station entrance/exit node
    def run_station_ent_ex_node_model(self, t_now):

        node_id_dict = self.simulation.all_nodes
        link_id_dict = self.simulation.all_links
        agent_id_dict = self.simulation.all_agents
        node2link_dict = self.simulation.node2link_dict
        station_id_dict = self.simulation.all_charging_stations

        # Get the vehicles that can move at this timestep
        go_vehs = self.station_ent_ex_non_conflict_vehs(t_now=t_now, link_id_dict=link_id_dict,
                                         agent_id_dict=agent_id_dict, node2link_dict=node2link_dict, node_id_dict=node_id_dict, station_id_dict=station_id_dict)

        for [agent_id, next_node, il, ol, _] in go_vehs:
            veh_len = agent_id_dict[agent_id].veh_len
            station_id = [s for s in station_id_dict.keys() if (station_id_dict[s].ent_ex_node_id == self.nid)][0]

            # A candidate charger may be located at an OD destination.  In
            # that case ``prepare_agent`` correctly returns no next link, but
            # the old code indexed ``link_id_dict[None]`` and crashed with
            # ``KeyError(None)``.  Handle this exactly like ordinary arrival,
            # also removing a vehicle that is leaving the station.
            if next_node is None:
                if self.nid != agent_id_dict[agent_id].destin_nid:
                    raise RuntimeError(
                        f'Agent {agent_id} has no next link at non-destination '
                        f'node {self.nid}'
                    )
                if link_id_dict[il].ltype == 'Out_Station':
                    if (station_id_dict[station_id].exit_queue and
                            station_id_dict[station_id].exit_queue[0][0] == agent_id):
                        station_id_dict[station_id].exit_queue.pop(0)
                else:
                    link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                agent_id_dict[agent_id].move_agent(
                    t_now, self.nid, None, 'arr'
                )
                agent_id_dict[agent_id].arrival_time = t_now
                agent_id_dict[agent_id].travel_time = (
                    t_now - agent_id_dict[agent_id].dept_time
                )
                continue

            # In terms of the agent's current/next link type, there are 3 cases:
            #   1. Agent goes from road to station
            #   2. Agent goes from station to road
            #   3. Agent goes from road to road

            # In terms of the agent's current/next link type, there are 3 cases:
            #   1. Agent goes from road to station
            #   2. Agent goes from station to road
            #   3. Agent goes from road to road

            # First, we consider the case where the agent's current link is the station's exit link
            if link_id_dict[il].ltype == 'Out_Station':
                # No storage capacity downstream
                if link_id_dict[ol].st_c < veh_len:
                    # # Increment the counter of the amount of time that the agent has spent in the exit queue
                    # ex_queue_vehicles_ind = [tuples[0] for tuples in station_id_dict[station_id].exit_queue].index(agent_id)
                    # new_time_val = list(station_id_dict[station_id].exit_queue[ex_queue_vehicles_ind])[1] + 1  # Can not change values in a tuple, so we convert it to a list first
                    # station_id_dict[station_id].exit_queue[ex_queue_vehicles_ind] = (agent_id, new_time_val)
                    pass
                # The agent can move if the outlink's receiving capacity allows
                elif link_id_dict[ol].in_c >= 1:
                    # Remove agent from the station's exit_queue
                    station_id_dict[station_id].exit_queue.pop(0)
                    # Move agent to next link
                    agent_id_dict[agent_id].move_agent(t_now, self.nid, next_node, 'flow')
                    link_id_dict[ol].receive_veh(agent_id)
                # The outlink receiving capacity exhausts
                else:
                    control_cap = link_id_dict[ol].in_c
                    toss_coin = random.choices([0, 1], weights=[1 - control_cap, control_cap], k=1)
                    if toss_coin[0]:  # vehicle can move
                        # Remove agent from the station's exit_queue
                        station_id_dict[station_id].exit_queue.pop(0)
                        # Move agent to next link
                        agent_id_dict[agent_id].move_agent(t_now, self.nid, next_node, 'chance')
                        link_id_dict[ol].receive_veh(agent_id)
                    else:
                        # # Increment the counter of the amount of time that the agent has spent in the exit queue
                        # ex_queue_vehicles_ind = 0#[tuples[0] for tuples in station_id_dict[station_id].exit_queue].index(agent_id)
                        # new_time_val = list(station_id_dict[station_id].exit_queue[ex_queue_vehicles_ind])[ 1] + 1  # Can not change values in a tuple, so we convert it to a list first
                        # station_id_dict[station_id].exit_queue[ex_queue_vehicles_ind] = (agent_id, new_time_val)

                        # Adjust the outlink receiving capacity
                        link_id_dict[ol].in_c = max(0, link_id_dict[ol].in_c - 1)
            # Now, we consider the case where the agent's current link is a standard road
            else:
                # Address the case where the agent is an EV that needs to charge at this station (case 1 from the list above)
                if agent_id_dict[agent_id].need_to_charge and agent_id_dict[agent_id].go_to_station_id == station_id: # CAN ALSO CHECK IF THE AGENTS NEXT NODE IS THE STATION
                    # There is space in the entrance queue of the station and the inlink has sufficient output capacity
                    if station_id_dict[station_id].ent_capacity > len(station_id_dict[station_id].ent_queue) and link_id_dict[il].ou_c >= 1:
                        # Remove vehicle from its current link
                        link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                        # Move vehicle to the entrance queue of the station
                        station_id_dict[station_id].ent_queue.append((agent_id, 0))
                        agent_id_dict[agent_id].move_agent_without_pointer(t_now, self.nid, next_node, "station_entrance_queue")
                        if next_node == self.nid+1:
                            # print("Agent #" + str(agent_id) + " moved to the entrance queue of the charging station.")
                            pass
                    # There is space in the entrance queue of the station, but the inlink does not have sufficient output capacity
                    elif station_id_dict[station_id].ent_capacity > len(station_id_dict[station_id].ent_queue) and link_id_dict[il].ou_c < 1:
                        control_cap = link_id_dict[il].ou_c
                        toss_coin = random.choices([0, 1], weights=[1 - control_cap, control_cap], k=1)
                        if toss_coin[0]:  # Vehicle can move
                            # Remove agent from current link
                            link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                            # Move agent to the entrance queue of the station
                            station_id_dict[station_id].ent_queue.append((agent_id, t_now))
                            agent_id_dict[agent_id].move_agent_without_pointer(t_now, self.nid, next_node, "station_entrance_queue_chance")
                        else:
                            # Adjust the inlink output capacity
                            link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c - 1)
                    # The inlink has sufficient output capacity, but the entrance queue is full
                    elif station_id_dict[station_id].ent_capacity <= len(station_id_dict[station_id].ent_queue) and link_id_dict[il].ou_c >= 1:
                        pass
                    # Both the inlink output capacity and the station entrance queue exhaust
                    else:
                        link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c - 1)
                # Address the case where the agent proceeds to the straight ahead link
                else:
                    # no storage capacity downstream
                    if link_id_dict[ol].st_c < veh_len:
                        pass  # no blocking, as # veh = # lanes
                    # inlink-sending, outlink-receiving both permits
                    elif link_id_dict[il].ou_c >= 1 and link_id_dict[ol].in_c >= 1:
                        # before move agent as it uses the old agent.cl_enter_time
                        link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                        agent_id_dict[agent_id].move_agent(
                            t_now, self.nid, next_node, 'flow')
                        link_id_dict[ol].receive_veh(agent_id)
                    # either inlink-sending or outlink-receiving or both exhaust
                    else:
                        # control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                        control_cap = min(link_id_dict[il].ou_c, link_id_dict[ol].in_c)
                        toss_coin = random.choices(
                            [0, 1], weights=[1 - control_cap, control_cap], k=1)
                        if toss_coin[0]:  # vehicle can move
                            # before move agent as it uses the old agent.cl_enter_time
                            link_id_dict[il].send_veh(t_now, agent_id, agent_id_dict)
                            agent_id_dict[agent_id].move_agent(
                                t_now, self.nid, next_node, 'chance')
                            link_id_dict[ol].receive_veh(agent_id)
                        else:  # even though vehicle cannot move, the remaining capacity needs to be adjusted
                            if link_id_dict[il].ou_c < link_id_dict[ol].in_c:
                                link_id_dict[il].ou_c = max(0, link_id_dict[il].ou_c - 1)
                            elif link_id_dict[ol].in_c < link_id_dict[il].ou_c:
                                link_id_dict[ol].in_c = max(0, link_id_dict[ol].in_c - 1)
                            else:
                                link_id_dict[il].ou_c -= 1
                                link_id_dict[ol].in_c -= 1

###################################################################### ADDITION ENDS

class Link:
    def __init__(self, link_id, lanes, length, maxmph, fft, capacity, ltype, start_nid, end_nid, geometry, simulation=None): ###################################################################### ADDITION INLINE
        # input
        self.lid = link_id
        self.lanes = lanes
        self.length = length
        self.fft = fft
        self.capacity = capacity
        self.ltype = ltype
        self.start_nid = start_nid
        self.end_nid = end_nid
        self.geometry = loads(geometry)
        self.simulation = simulation
        # derived
        # at least allow any vehicle to pass. i.e., the road won't block any vehicle because of the road length
        self.store_cap = max(18, length*lanes)
        self.in_c = self.capacity/3600.0  # capacity in veh/s
        self.ou_c = self.capacity/3600.0
        self.st_c = self.store_cap  # remaining storage capacity
        self.midpoint = list(self.geometry.interpolate(
            0.5, normalized=True).coords)[0]
        # empty
        self.queue_veh = []  # [(agent, t_enter), (agent, t_enter), ...]
        self.run_veh = []
        # [(t_enter, dur), ...] travel time of each agent left the link in a given period; reset at times
        self.travel_time_list = []
        self.travel_time = fft
        self.start_node = None
        self.end_node = None

###################################################################### ADDITION STARTS

        self.tot_entering_vehs = 0
        self.running_travel_time_list = []
        self.completed_travel_time_list = []
        self.completed_travel_time_list_test = []
        self.ave_travel_time = 0
        self.ave_flow = 0
        self.occup_time = 0
        self.maxmph = maxmph
        self.density = 0

###################################################################### ADDITION ENDS

    def send_veh(self, t_now, agent_id, agent_id_dict=None):
        # remove the agent from queue
        self.queue_veh = [v for v in self.queue_veh if v != agent_id]
        self.ou_c = max(0, self.ou_c-1)
        if self.ltype[0:2] != 'vl':
            self.travel_time_list.append(
                (t_now, t_now - agent_id_dict[agent_id].cl_enter_time))

###################################################################### ADDITION STARTS

        self.completed_travel_time_list.append([agent_id, t_now - agent_id_dict[agent_id].cl_enter_time])
        self.completed_travel_time_list_test.append(self.running_travel_time_list[[item[0] for item in self.running_travel_time_list].index(agent_id)])
        self.running_travel_time_list = [tuple for tuple in self.running_travel_time_list if tuple[0] != agent_id]

        self.density = self.density - 1

###################################################################### ADDITION ENDS

    def receive_veh(self, agent_id):
        self.run_veh.append(agent_id)
        self.in_c = max(0, self.in_c-1)

###################################################################### ADDITION STARTS

        self.tot_entering_vehs += 1
        self.running_travel_time_list.append([agent_id, 0])

        self.density += 1

###################################################################### ADDITION ENDS

    def run_link_model(self, t_now):
        if t_now % 60 == 0:
            self.update_travel_time(
                t_now, link_time_lookback_freq=60, g=self.simulation.g)
        for agent_id in self.run_veh:
            if self.simulation.all_agents[agent_id].cl_enter_time < t_now - self.fft:
                self.queue_veh.append(agent_id)
        self.run_veh = [v for v in self.run_veh if v not in self.queue_veh]
        # remaining spaces on link for the node model to move vehicles to this link
        self.st_c = self.store_cap - \
            np.sum(
                [self.simulation.all_agents[agent_id].veh_len for agent_id in self.run_veh+self.queue_veh])
        self.in_c, self.ou_c = self.capacity/3600, self.capacity/3600

    def update_travel_time(self, t_now, link_time_lookback_freq=None, g=None):
        self.travel_time_list = [(t_rec, dur) for (
            t_rec, dur) in self.travel_time_list if (t_now-t_rec < link_time_lookback_freq)]
        if len(self.travel_time_list) > 0:
            self.travel_time = np.mean(
                [dur for (_, dur) in self.travel_time_list])
            g.update_edge(self.start_nid, self.end_nid,
                          c_double(self.travel_time))


class Agent:
    def __init__(self, id, origin_nid, destin_nid, dept_time, veh_len, gps_reroute, is_EV, need_to_charge, current_charge, target_charge, go_to_station_id, simulation=None): ###################################################################### ADDITION INLINE
        # input
        self.aid = id
        self.origin_nid = origin_nid
        self.destin_nid = destin_nid
        self.dept_time = dept_time
        self.veh_len = veh_len
        self.gps_reroute = gps_reroute
        self.simulation = simulation
        # derived
        self.cls = 'vn_source_{}'.format(
            self.origin_nid)  # current link start node
        self.cle = self.origin_nid  # current link end node
        # Empty
        self.route_igraph = []
        self.route_link_ids = []
        self.find_route = None
        self.status = None
        self.cl_enter_time = None

###################################################################### ADDITION STARTS

        self.is_EV = is_EV
        self.need_to_charge = need_to_charge
        self.current_charge = current_charge
        self.target_charge = target_charge
        self.go_to_station_id = go_to_station_id

        self.route_pointer = 1

        self.arrival_time = np.nan

###################################################################### ADDITION ENDS
        
    def load_trips(self, t_now):
        if (self.dept_time == t_now):
            initial_edge = self.simulation.node2link_dict[self.route_igraph[0]]
            self.simulation.all_links[initial_edge].run_veh.append(self.aid)
            self.status = 'loaded'
            self.cl_enter_time = t_now

###################################################################### ADDITION STARTS
            self.simulation.all_links[initial_edge].running_travel_time_list.append([self.aid, 0])
###################################################################### ADDITION ENDS

    def prepare_agent(self, node_id, node2link_dict=None, node_id_dict=None):
        assert self.cle == node_id, "agent next node {} is not the transferring node {}, route {}".format(
            self.cle, node_id, self.route_igraph)
        if self.destin_nid == node_id:  # current node is agent destination
            return None, None, 0  # id, next_node, dir

###################################################################### ADDITION STARTS

        # agent_next_node = [
        #     end for (start, end) in self.route_igraph if start == node_id][0]

        agent_next_node = self.route_igraph[self.route_pointer][1]
        if self.route_igraph[self.route_pointer][0] != node_id:
            print("Error with prepare agent")
            pass

###################################################################### ADDITION STARTS

        route_edge_index = self.route_pointer - 1
        if self.route_link_ids and route_edge_index < len(self.route_link_ids):
            ol = int(self.route_link_ids[route_edge_index])
        else:
            ol = node2link_dict[(node_id, agent_next_node)]
        x_start, y_start = node_id_dict[self.cls].lon, node_id_dict[self.cls].lat
        x_mid, y_mid = node_id_dict[node_id].lon, node_id_dict[node_id].lat
        x_end, y_end = node_id_dict[agent_next_node].lon, node_id_dict[agent_next_node].lat
        in_vec, out_vec = (x_mid-x_start, y_mid -
                           y_start), (x_end-x_mid, y_end-y_mid)
        dot, det = (in_vec[0]*out_vec[0] + in_vec[1]*out_vec[1]
                    ), (in_vec[0]*out_vec[1] - in_vec[1]*out_vec[0])
        agent_dir = np.arctan2(det, dot)*180/np.pi
        return agent_next_node, ol, agent_dir

    def move_agent(self, t_now, new_cls, new_cle, new_status):
        self.cls = new_cls
        self.cle = new_cle
        self.status = new_status
        self.cl_enter_time = t_now

###################################################################### ADDITION STARTS

        self.route_pointer += 1

    def move_agent_without_pointer(self, t_now, new_cls, new_cle, new_status):
        self.cls = new_cls
        self.cle = new_cle
        self.status = new_status
        self.cl_enter_time = t_now

###################################################################### ADDITION ENDS

    def get_path(self, g=None):

###################################################################### ADDITION STARTS

        if self.need_to_charge: # The agent is an EV that needs to charge; Find the shortest path that passes through a charging station
            shortest_dist = 10e8
            sp_route = []
            charging_stations_dict = self.simulation.all_charging_stations
            for station_id in charging_stations_dict.keys():
                sp1 = g.dijkstra(self.origin_nid, charging_stations_dict[station_id].ent_ex_node_id)
                sp2 = g.dijkstra(charging_stations_dict[station_id].ent_ex_node_id, self.destin_nid)
                sp1_route = sp1.route(charging_stations_dict[station_id].ent_ex_node_id)
                sp2_route = sp2.route(self.destin_nid)
                sp_dist = sp1.distance(charging_stations_dict[station_id].ent_ex_node_id) + sp2.distance(self.destin_nid)
                if sp_dist < shortest_dist:
                    shortest_dist = sp_dist
                    sp_route = ([(self.cls, self.origin_nid)]
                                + [(start_nid, end_nid) for (start_nid, end_nid) in sp1_route]
                                + [(start_nid, end_nid) for (start_nid, end_nid) in sp2_route])
                    opt_stat_id = station_id
            if shortest_dist > 10e7:
                sp1.clear()
                sp2.clear()
                self.route_igraph = []
                return 'no_path_found'
            else:
                self.route_igraph = sp_route
                self.go_to_station_id = opt_stat_id
                sp1.clear()
                sp2.clear()
                # print("Route of agent " + str(self.aid) + ": " + str(sp_route))
            return 'path_found'

        else: # The agent does not need to charge, so we find the shortest path to its destination
            sp = g.dijkstra(self.cle, self.destin_nid)
            sp_dist = sp.distance(self.destin_nid)
            if sp_dist > 10e7:
                sp.clear()
                self.route_igraph = []
                return 'no_path_found'
            else:
                sp_route = sp.route(self.destin_nid)
                self.route_igraph = [(self.cls, self.cle)] + [(start_nid, end_nid)
                                                              for (start_nid, end_nid) in sp_route]
                sp.clear()
                return 'path_found'

###################################################################### ADDITION ENDS

class Simulation:
    def __init__(self, NodeClass=Node, LinkClass=Link, ChargingStationClass=EV_Charging_Station): ###################################################################### ADDITION INLINE
        self.g = None
        self.all_nodes = dict()
        self.all_links = dict()
        self.all_agents = dict()
        assert issubclass(
            NodeClass, Node), 'arg: NodeClass, must submit Node class that is a Node'
        assert issubclass(
            LinkClass, Link), 'arg: LinkClass, must submit Link class that is a Link'
        self.NodeClass = NodeClass
        self.LinkClass = LinkClass
        
###################################################################### ADDITION STARTS
        
        self.all_charging_stations = dict()
        assert issubclass(
            ChargingStationClass, EV_Charging_Station), 'arg: ChargingStationClass, must submit EV_Charging_Station class that is an EV_Charging_Station'
        self.ChargingStationClass = ChargingStationClass
        
###################################################################### ADDITION ENDS

    def create_network(self, nodes_df, links_df, charging_stations_df): ###################################################################### ADDITION INLINE

        # create graph
        nodes_df = nodes_df.copy()
        links_df = links_df.copy()
        charging_stations_df = charging_stations_df.copy()
        links_df['lanes'] = pd.to_numeric(links_df['lanes'], errors='coerce').fillna(1.0)
        links_df['capacity'] = links_df['lanes'] * 1900 ###################################################################### ADDITION INLINE
###################################################################### ADDITION STARTS
        links_df.loc[links_df['type'] == 'In_Station','capacity'] = np.nan
        links_df.loc[links_df['type'] == 'Out_Station', 'capacity'] = np.nan
###################################################################### ADDITION ENDS
        links_df['fft'] = np.where(
            links_df['lanes'] <= 0, 1e8, links_df['length']/links_df['maxmph']*2.2369)
        self.g = interface.from_dataframe(
            links_df, 'start_node_id', 'end_node_id', 'fft')

        # Create link and node objects
        nodes = []
        links = []
        for row in nodes_df.itertuples():
            real_node = self.NodeClass(getattr(row, 'node_id'), getattr(row, 'lon'), getattr(
                row, 'lat'), getattr(row, 'type'), getattr(row, 'node_osmid'), simulation=self)
            virtual_node = real_node.create_virtual_node()
            virtual_link = real_node.create_virtual_link()
            nodes.append(real_node)
            nodes.append(virtual_node)
            links.append(virtual_link)
        for row in links_df.itertuples():
            real_link = self.LinkClass(getattr(row, 'link_id'), getattr(row, 'lanes'), getattr(row, 'length'),
                                       getattr(row, 'maxmph'), ###################################################################### ADDITION INLINE
                                       getattr(row, 'fft'), getattr( row, 'capacity'), getattr(row, 'type'), getattr(row, 'start_node_id'), getattr(row, 'end_node_id'), getattr(row, 'geometry'), simulation=self)
            links.append(real_link)

        self.all_links = {link.lid: link for link in links}
        self.pair_to_link_ids = {}
        for link in links:
            self.pair_to_link_ids.setdefault((link.start_nid, link.end_nid), []).append(link.lid)
        for pair in self.pair_to_link_ids:
            self.pair_to_link_ids[pair] = sorted(self.pair_to_link_ids[pair])
        # Kept for legacy simulator internals only when the pair is unique.
        self.node2link_dict = {
            pair: link_ids[0]
            for pair, link_ids in self.pair_to_link_ids.items()
            if len(link_ids) == 1
        }
        self.all_nodes = {node.nid: node for node in nodes}
        for link_id, link in self.all_links.items():
            self.all_nodes[link.start_nid].out_links.append(link_id)
            self.all_nodes[link.end_nid].in_links[link_id] = None
        for node in self.all_nodes.values():
            node.calculate_straight_ahead_links(
                node_id_dict=self.all_nodes, link_id_dict=self.all_links)
                
###################################################################### ADDITION STARTS

        # Create charging station objects from the input dataframe
        charging_stations = []
        for row in charging_stations_df.itertuples():
            real_station = self.ChargingStationClass(getattr(row, 'station_id'), getattr(row, 'in_link_id'), getattr(row, 'out_link_id'), getattr(row, 'node_id'),
                                                     getattr(row, 'ent_ex_node_id'), getattr(row, 'lon'), getattr(row, 'lat'),
                                                     getattr(row, 'ent_capacity'), getattr(row, 'charging_capacity'), getattr(row, 'exit_capacity'), getattr(row, 'cost'), simulation=self)
            charging_stations.append(real_station)

        # Create station dictionary for quick look-up
        self.all_charging_stations = {charging_station.station_id: charging_station for charging_station in charging_stations}
         
###################################################################### ADDITION ENDS

    def resolve_link_id(self, start_nid, end_nid, link_id=None):
        """Resolve a route edge without silently collapsing parallel links."""
        if link_id is not None:
            if int(link_id) not in self.all_links:
                raise KeyError(f"Unknown link_id {link_id}")
            link = self.all_links[int(link_id)]
            if link.start_nid != start_nid or link.end_nid != end_nid:
                raise ValueError(f"link_id {link_id} does not connect {start_nid}->{end_nid}")
            return int(link_id)
        candidates = self.pair_to_link_ids.get((start_nid, end_nid), [])
        if len(candidates) != 1:
            raise ValueError(
                f"Edge pair {start_nid}->{end_nid} is ambiguous; provide link_id. "
                f"Candidates: {candidates}"
            )
        return candidates[0]

    def create_demand(self, od_df):

        if 'agent_id' not in od_df.columns:
            od_df['agent_id'] = np.arange(od_df.shape[0])
        # Preserve an explicitly supplied departure schedule.  The BPR
        # calibration uses this to inject a controlled rate over a finite
        # window; overwriting it here collapses every sweep level into a
        # single time-zero batch and makes the measured x/y pairs invalid.
        if 'dept_time' not in od_df.columns:
            od_df['dept_time'] = 0
        od_df['veh_len'] = 8
        od_df['gps_reroute'] = 0
        od_df = od_df.sample(frac=1, random_state=getattr(self, 'random_seed', None)).reset_index(
            drop=True)  # randomly shuffle rows
        # print('# trips {}'.format(od_df.shape[0]))

        for row in od_df.itertuples():
            self.all_agents[row.agent_id] = Agent(getattr(row, 'agent_id'), getattr(row, 'origin_node_id'), getattr(
                row, 'destin_node_id'), getattr(row, 'dept_time'), getattr(row, 'veh_len'), getattr(row, 'gps_reroute'), getattr(row, 'is_EV'), getattr(row, 'need_to_charge'), getattr(row, 'current_charge'), getattr(row, 'target_charge'), getattr(row, 'go_to_station_id'), simulation=self) ###################################################################### ADDITION INLINE
