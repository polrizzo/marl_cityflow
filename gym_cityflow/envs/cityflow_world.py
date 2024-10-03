import json
import os
import cityflow
import numpy as np
import math
from gym_cityflow.envs.cityflow_intersection import Intersection


class World(object):
    """
    World Class is mainly used for creating a CityFlow engine and maintain information about CityFlow world.
    """

    def __init__(self, cityflow_config, thread_num=1, **kwargs):
        print("building world...")
        self.eng = cityflow.Engine(cityflow_config, thread_num=thread_num)
        with open(cityflow_config) as f:
            cityflow_config = json.load(f)
        self.roadnet = self._get_roadnet(cityflow_config)
        self.RIGHT = True  # vehicles moves on the right side, currently always set to true due to CityFlow's mechanism
        self.interval = cityflow_config["interval"]

        # get all non virtual intersections
        self.intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersection_ids = [i["id"] for i in self.intersections]

        # create non-virtual Intersections
        print("creating intersections...")
        non_virtual_intersections = [i for i in self.roadnet["intersections"] if not i["virtual"]]
        self.intersections = [Intersection(i, self) for i in non_virtual_intersections]
        self.intersection_ids = [i["id"] for i in non_virtual_intersections]
        self.id2intersection = {i.id: i for i in self.intersections}
        self.id2idx = {i: idx for idx, i in enumerate(self.id2intersection)}
        print("intersections created.")

        # id of all roads and lanes
        print("parsing roads...")
        self.all_roads = []
        self.all_lanes = []
        self.all_lanes_speed = {}
        self.lane_length = {}

        for road in self.roadnet["roads"]:
            self.all_roads.append(road["id"])
            i = 0
            road_l = self.get_road_length(road)
            for lane in road["lanes"]:
                self.all_lanes.append(road["id"] + "_" + str(i))
                self.all_lanes_speed[road["id"] + "_" + str(i)] = lane['maxSpeed']
                self.lane_length[road["id"] + "_" + str(i)] = road_l
                i += 1

            iid = road["startIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, True)
            iid = road["endIntersection"]
            if iid in self.intersection_ids:
                self.id2intersection[iid].insert_road(road, False)

        for i in self.intersections:
            i.sort_roads()

        print("roads parsed.")

        # initializing info functions
        self.info_functions = {
            "vehicles": (lambda: self.eng.get_vehicles(include_waiting=True)),
            "lane_count": self.eng.get_lane_vehicle_count,
            "lane_waiting_count": self.eng.get_lane_waiting_vehicle_count,
            "lane_vehicles": self.eng.get_lane_vehicles,
            "time": self.eng.get_current_time,
            "vehicle_distance": self.eng.get_vehicle_distance,
            "pressure": self.get_pressure,
            "lane_waiting_time_count": self.get_lane_waiting_time_count,
            "lane_delay": self.get_lane_delay,
            "real_delay": self.get_real_delay,
            "vehicle_trajectory": self.get_vehicle_trajectory,
            "history_vehicles": self.get_history_vehicles,
            "phase": self.get_cur_phase,
            "throughput": self.get_cur_throughput,
            "average_travel_time": self.get_average_travel_time
            # "action_executed": self.get_executed_action
        }
        self.fns = []
        self.info = {}
        self.vehicle_waiting_time = {}  # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_trajectory = {}  # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.history_vehicles = set()
        self.real_delay = {}

        # # get in_lines and out_lanes
        # self.list_entering_lanes, self.list_exiting_lanes = self.get_in_out_lanes()

        # record lanes' vehicles to calculate arrive_leave_time
        self.dic_lane_vehicle_previous_step = {key: None for key in self.all_lanes}
        self.dic_lane_vehicle_current_step = {key: None for key in self.all_lanes}
        self.dic_vehicle_arrive_leave_time = dict()  # cumulative

        print("world built.")

    def reset_vehicle_info(self):
        """
        Reset vehicle infos, including waiting_time, trajectory, etc.

        :param: None
        :return: None
        """
        self.vehicle_waiting_time = {}  # key: vehicle_id, value: the waiting time of this vehicle since last halt.
        self.vehicle_trajectory = {}  # key: vehicle_id, value: [[lane_id_1, enter_time, time_spent_on_lane_1], ... , [lane_id_n, enter_time, time_spent_on_lane_n]]
        self.history_vehicles = set()
        self.real_delay = {}
        self.dic_lane_vehicle_previous_step = {key: None for key in self.all_lanes}
        self.dic_lane_vehicle_current_step = {key: None for key in self.all_lanes}
        self.dic_vehicle_arrive_leave_time = dict()

    def _update_arrive_time(self, list_vehicle_arrive):
        """
        Update enter time of vehicles.

        :param list_vehicle_arrive: vehicles' id that have entered in roadnet
        :return: None
        """
        ts = self.eng.get_current_time()
        # init vehicle enter leave time
        for vehicle in list_vehicle_arrive:
            if vehicle not in self.dic_vehicle_arrive_leave_time:
                self.dic_vehicle_arrive_leave_time[vehicle] = {"enter_time": ts, "leave_time": np.nan,
                                                               "cost_time": np.nan}
            else:
                # print("vehicle: %s already exists in entering lane!"%vehicle)
                pass

    def _update_left_time(self, list_vehicle_left):
        """
        Update left time of vehicles.

        :param list_vehicle_left: vehicles' id that have left from roadnet
        :return: None
        """
        ts = self.eng.get_current_time()
        # update the time for vehicle to leave entering lane
        for vehicle in list_vehicle_left:
            try:
                self.dic_vehicle_arrive_leave_time[vehicle]["leave_time"] = ts
                self.dic_vehicle_arrive_leave_time[vehicle]["cost_time"] = ts - \
                                                                           self.dic_vehicle_arrive_leave_time[vehicle][
                                                                               "enter_time"]
            except KeyError:
                print("vehicle not recorded when entering!")

    def update_current_measurements(self):
        """
        Update information, including enter time of vehicle, left time of vehicle, lane id that vehicles are running, etc.

        :param: None
        :return: None
        """

        def _change_lane_vehicle_dic_to_list(dic_lane_vehicle):
            list_lane_vehicle = []
            for value in dic_lane_vehicle.values():
                if value:
                    list_lane_vehicle.extend(value)
            return list_lane_vehicle

        # contain outflow lanes
        self.dic_lane_vehicle_current_step = self.eng.get_lane_vehicles()

        # get vehicle list
        self.list_lane_vehicle_current_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_current_step)
        self.list_lane_vehicle_previous_step = _change_lane_vehicle_dic_to_list(self.dic_lane_vehicle_previous_step)
        list_vehicle_new_arrive = list(
            set(self.list_lane_vehicle_current_step) - set(self.list_lane_vehicle_previous_step))
        list_vehicle_new_left = list(
            set(self.list_lane_vehicle_previous_step) - set(self.list_lane_vehicle_current_step))
        self._update_arrive_time(list_vehicle_new_arrive)
        self._update_left_time(list_vehicle_new_left)

    def get_cur_throughput(self):
        """
        Get vehicles' count in the whole roadnet at current step.

        :param: None
        :return throughput: throughput in the whole roadnet at current step
        """
        throughput = 0
        for dic in self.dic_vehicle_arrive_leave_time:
            vehicle = self.dic_vehicle_arrive_leave_time[dic]
            if (not np.isnan(vehicle["cost_time"])) and vehicle["leave_time"] <= self.eng.get_current_time():
                throughput += 1
        return throughput

    def get_executed_action(self):
        """
        Get executed action of each intersection at current step.

        :param: None
        :return actions: executed action of each intersection at current step
        """
        actions = []
        for i in self.intersections:
            actions.append(i.action_executed)
        return actions

    def get_cur_phase(self):
        """
        Get current phase of each intersection.

        :param: None
        :return phases: current phase of each intersection
        """
        phases = []
        for i in self.intersections:
            phases.append(i.current_phase)
        return phases

    def get_pressure(self):
        """
        Get pressure of each intersection.
        Pressure of an intersection equals to number of vehicles that in in_lanes minus number of vehicles that in out_lanes.

        :param: None
        :return pressures: pressure of each intersection
        """
        vehicles = self.eng.get_lane_vehicle_count()
        pressures = {}
        for i in self.intersections:
            pressure = 0
            in_lanes = []
            for road in i.in_roads:
                from_zero = (road["startIntersection"] == i.id) if self.RIGHT else (
                        road["endIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    in_lanes.append(road["id"] + "_" + str(n))
            out_lanes = []
            for road in i.out_roads:
                from_zero = (road["endIntersection"] == i.id) if self.RIGHT else (
                        road["startIntersection"] == i.id)
                for n in range(len(road["lanes"]))[::(1 if from_zero else -1)]:
                    out_lanes.append(road["id"] + "_" + str(n))
            for lane in vehicles.keys():
                if lane in in_lanes:
                    pressure += vehicles[lane]
                if lane in out_lanes:
                    pressure -= vehicles[lane]
            pressures[i.id] = pressure
        return pressures

    def get_vehicle_lane(self):
        """
        Get current lane id of each vehicle that is running.

        :param: None
        :return vehicle_lane: current lane id of each vehicle that is running
        """
        # get the current lane of each vehicle. {vehicle_id: lane_id}
        vehicle_lane = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        for lane in self.all_lanes:
            for vehicle in lane_vehicles[lane]:
                vehicle_lane[vehicle] = lane
        return vehicle_lane

    def get_vehicle_waiting_time(self):
        """
        Get waiting time of vehicles according to vehicle's speed.
        If a vehicle's speed less than 0.1m/s, then its waiting time would be added 1s.

        :param: None
        :return vehicle_waiting_time: waiting time of vehicles
        """
        # the waiting time of vehicle since last halt.
        vehicles = self.eng.get_vehicles(include_waiting=False)
        vehicle_speed = self.eng.get_vehicle_speed()
        for vehicle in vehicles:
            if vehicle not in self.vehicle_waiting_time.keys():
                self.vehicle_waiting_time[vehicle] = 0
            if vehicle_speed[vehicle] < 0.1:
                self.vehicle_waiting_time[vehicle] += 1
            else:
                self.vehicle_waiting_time[vehicle] = 0
        return self.vehicle_waiting_time

    def get_lane_waiting_time_count(self):
        """
        Get waiting time of vehicles in each lane.

        :param: None
        :return lane_waiting_time: waiting time of vehicles in each lane
        """
        # the sum of waiting times of vehicles on the lane since their last halt.
        lane_waiting_time = {}
        lane_vehicles = self.eng.get_lane_vehicles()
        vehicle_waiting_time = self.get_vehicle_waiting_time()
        for lane in self.all_lanes:
            lane_waiting_time[lane] = 0
            for vehicle in lane_vehicles[lane]:
                lane_waiting_time[lane] += vehicle_waiting_time[vehicle]
        return lane_waiting_time

    def get_lane_delay(self):
        """
        Get approximate delay of each lane.
        Approximate delay of each lane equals to (1 - lane_avg_speed)/lane_speed_limit.

        :param: None
        :return lane_delay: approximate delay of each lane
        """
        # the delay of each lane: 1 - lane_avg_speed/speed_limit
        lane_vehicles = self.eng.get_lane_vehicles()
        lane_delay = {}
        lanes = self.all_lanes
        vehicle_speed = self.eng.get_vehicle_speed()

        for lane in lanes:
            vehicles = lane_vehicles[lane]
            lane_vehicle_count = len(vehicles)
            lane_avg_speed = 0.0
            for vehicle in vehicles:
                speed = vehicle_speed[vehicle]
                lane_avg_speed += speed
            if lane_vehicle_count == 0:
                lane_avg_speed = self.all_lanes_speed[lane]
            else:
                lane_avg_speed /= lane_vehicle_count
            lane_delay[lane] = 1 - lane_avg_speed / self.all_lanes_speed[lane]
        return lane_delay

    def get_vehicle_trajectory(self):
        """
        Get trajectory of vehicles that have entered in roadnet, including vehicle_id, enter time, leave time or current time.

        :param: None
        :return vehicle_trajectory: trajectory of vehicles that have entered in roadnet
        """
        # lane_id and time spent on the corresponding lane that each vehicle went through
        vehicle_lane = self.get_vehicle_lane()  # get vehicles on tne roads except turning
        vehicles = self.eng.get_vehicles(include_waiting=False)
        for vehicle in vehicles:
            if vehicle not in self.vehicle_trajectory:
                self.vehicle_trajectory[vehicle] = [[vehicle_lane[vehicle], int(self.eng.get_current_time()), 0]]
            else:
                if vehicle not in vehicle_lane.keys():  # vehicle is turning
                    continue
                if vehicle_lane[vehicle] == self.vehicle_trajectory[vehicle][-1][
                    0]:  # vehicle is running on the same lane
                    self.vehicle_trajectory[vehicle][-1][2] += 1
                else:  # vehicle has changed the lane
                    self.vehicle_trajectory[vehicle].append(
                        [vehicle_lane[vehicle], int(self.eng.get_current_time()), 0])
        return self.vehicle_trajectory

    def get_history_vehicles(self):
        """
        Get vehicles that have entered in roadnet.

        :param: None
        :return history_vehicles: information of vehicles that have entered in roadnet.
        """
        self.history_vehicles.update(self.eng.get_vehicles())
        return self.history_vehicles

    def _get_roadnet(self, cityflow_config):
        """
        Read information from roadnet file in the config file.

        :param cityflow_config: config file of a roadnet
        :return roadnet: information of a roadnet
        """
        roadnet_file = os.path.join(cityflow_config["dir"], cityflow_config["roadnetFile"])
        with open(roadnet_file) as f:
            roadnet = json.load(f)
        return roadnet

    def subscribe(self, fns):
        """
        Subscribe information you want to get when training the model.

        :param fns: information name you want to get
        :return: None
        """
        if isinstance(fns, str):
            fns = [fns]
        for fn in fns:
            if fn in self.info_functions:
                if not fn in self.fns:
                    self.fns.append(fn)
            else:
                raise Exception("info function %s not exists" % fn)

    def step(self, actions=None):
        """
        Take relative actions and update information,
        including global information, measurements and trajectory, etc.

        :param actions: actions list to be executed at all intersections at the next step
        :return: None
        """
        #  update previous measurement
        self.dic_lane_vehicle_previous_step = self.dic_lane_vehicle_current_step

        if actions is not None:
            for i, action in enumerate(actions):
                self.intersections[i].step(action, self.interval)
        self.eng.next_step()
        self._update_infos()
        # update current measurement
        self.update_current_measurements()
        self.vehicle_trajectory = self.get_vehicle_trajectory()

    def reset(self):
        """
        Reset information, including waiting_time, trajectory, etc.

        :param: None
        :return: None
        """
        self.eng.reset()
        for I in self.intersections:
            I.reset()
        self._update_infos()
        # reset vehicles info
        self.reset_vehicle_info()

    def _update_infos(self):
        """
        Update global information after reset or each step.

        :param: None
        :return: None
        """
        self.info = {}
        for fn in self.fns:
            self.info[fn] = self.info_functions[fn]()

    def get_info(self, info):
        """
        Get specific information.

        :param info: the name of the specific information
        :return _info: specific information
        """
        _info = self.info[info]
        return _info

    def get_average_travel_time(self):
        """
        Get average travel time of all vehicles.

        :param: None
        :return tvg_time: average travel time of all vehicles
        """
        tvg_time = self.eng.get_average_travel_time()
        return tvg_time

    def get_lane_queue_length(self):
        """
        Get queue length of all lanes in the traffic network.

        :param: None
        :return lane_q_length: queue length of all lanes
        """
        lane_q_length = self.eng.get_lane_waiting_vehicle_count()
        return lane_q_length

    def get_road_length(self, road):
        """
        Calculate the length of a road.

        :param road: information about a road
        :return road_length: length of a specific road
        """
        point_x = road['points'][0]['x'] - road['points'][1]['x']
        point_y = road['points'][0]['y'] - road['points'][1]['y']
        road_length = math.sqrt((point_x ** 2) + (point_y ** 2))
        return road_length

    def get_real_delay(self):
        """
        Calculate average real delay.
        Real delay of a vehicle is defined as the time a vehicle has traveled within the environment minus the expected travel time.

        :param: None
        :return avg_delay: average real delay of all vehicles
        """
        self.vehicle_trajectory = self.get_vehicle_trajectory()
        for v in self.vehicle_trajectory:
            # get road level routes of vehicle
            routes = self.vehicle_trajectory[v]  # lane_level
            for idx, lane in enumerate(routes):
                # speed = min(self.all_lanes_speed[lane[0]], float(info['speed']))
                speed = min(self.all_lanes_speed[lane[0]], 11.11)
                lane_length = self.lane_length[lane[0]]
                if idx == len(routes) - 1:  # the last lane
                    # judge whether the vehicle run over the whole lane.
                    dis = self.eng.get_vehicle_distance()
                    lane_length = dis[v] if v in dis.keys() else lane_length
                planned_tt = float(lane_length) / speed
                real_delay = lane[-1] - planned_tt if lane[-1] > planned_tt else 0.
                if v not in self.real_delay.keys():
                    self.real_delay[v] = real_delay
                else:
                    self.real_delay[v] += real_delay

        avg_delay = 0.
        count = 0
        for dic in self.real_delay.items():
            avg_delay += dic[1]
            count += 1
        avg_delay = avg_delay / count
        return avg_delay