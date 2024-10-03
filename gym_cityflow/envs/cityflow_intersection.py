from math import atan2, pi


class Intersection(object):
    """
    Intersection Class is mainly used for describing crossing information and defining acting methods.
    """

    def __init__(self, intersection, world):
        self.id = intersection["id"]
        self.world = world
        self.eng = self.world.eng

        # incoming and outgoing roads of each intersection, clock-wise order from North
        self.roads = []
        self.outs = []
        self.directions = []
        self.out_roads = None
        self.in_roads = None

        # links and phase information of each intersection
        self.current_phase = 0
        self.roadlinks = []
        self.lanelinks_of_roadlink = []
        self.startlanes = []
        self.lanelinks = []
        self.phase_available_roadlinks = []
        self.phase_available_lanelinks = []
        self.phase_available_startlanes = []

        # create yellow phases; in cityflow, yellow phases' id is 0
        phases = intersection["trafficLight"]["lightphases"]
        self.all_phases = [i for i in range(len(phases))]
        self.yellow_phase_id = [0]
        self.yellow_phase_time = 5
        self.phases = [i for i in range(len(phases)) if not i in self.yellow_phase_id]  # mapping from model output to cityflow phase id
        # parsing links and phases
        for roadlink in intersection["roadLinks"]:
            self.roadlinks.append((roadlink["startRoad"], roadlink["endRoad"]))
            lanelinks = []
            for lanelink in roadlink["laneLinks"]:
                startlane = roadlink["startRoad"] + "_" + str(lanelink["startLaneIndex"])
                self.startlanes.append(startlane)
                endlane = roadlink["endRoad"] + "_" + str(lanelink["endLaneIndex"])
                lanelinks.append((startlane, endlane))
            self.lanelinks.extend(lanelinks)
            self.lanelinks_of_roadlink.append(lanelinks)

        self.startlanes = list(set(self.startlanes))

        for i in self.phases:
            phase = phases[i]
            self.phase_available_roadlinks.append(phase["availableRoadLinks"])
            phase_available_lanelinks = []
            phase_available_startlanes = []
            for roadlink_id in phase["availableRoadLinks"]:
                lanelinks_of_roadlink = self.lanelinks_of_roadlink[roadlink_id]
                phase_available_lanelinks.extend(lanelinks_of_roadlink)
                for lanelinks in lanelinks_of_roadlink:
                    phase_available_startlanes.append(lanelinks[0])
            self.phase_available_lanelinks.append(phase_available_lanelinks)
            phase_available_startlanes = list(set(phase_available_startlanes))
            self.phase_available_startlanes.append(phase_available_startlanes)

        # init action, phase and time
        self.action_before_yellow = None
        self.action_executed = None
        self._current_phase = None
        self.current_phase_time = 0

        self.reset()

    def insert_road(self, road, out):
        """
        insert_road
        It's used to append a road into self.road and add the corresponding direction with the added road.

        :param road: newly added road
        :param out: newly added out
        :return: None
        """
        self.roads.append(road)
        self.outs.append(out)
        self.directions.append(self._get_direction(road, out))

    def sort_roads(self):
        """
        sort_roads
        Sort roads information by arranging an order.

        :return: None
        """
        # self.world.RIGHT: decide whether to sort from right side,
        # currently always set to true due to CityFlow's mechanism.
        order = sorted(range(len(self.roads)),
                       key=lambda i: (self.directions[i], self.outs[i] if self.world.RIGHT else not self.outs[i]))
        self.roads = [self.roads[i] for i in order]
        self.directions = [self.directions[i] for i in order]
        self.outs = [self.outs[i] for i in order]
        self.out_roads = [self.roads[i] for i, x in enumerate(self.outs) if x]
        self.in_roads = [self.roads[i] for i, x in enumerate(self.outs) if not x]

    def _change_phase(self, phase, interval, typ='init'):
        """
        Change current phase and calculate time duration of current phase.

        :param phase: true phase id (including yellows)
        :param interval: the non-acting time slice
        :param typ: calculation type of current phase time,
        'init' means calculate from scratch,
        'add' means current phase time add interval time.
        :return: None
        """
        assert typ in ['init', 'add'], "Type must be 'init' or 'add'"
        self.eng.set_tl_phase(self.id, phase)
        self._current_phase = phase
        if typ == 'add':
            self.current_phase_time += interval
        else:
            self.current_phase_time = interval

    def step(self, action, interval):
        """
        Take relative actions according to interval.

        :param action: the changes to take
        :param interval: the non-acting time slice
        :return: None
        """
        # if current phase is yellow, then continue to finish the yellow phase
        # recall self._current_phase means true phase id (including yellows)
        # self.current_phase means phase id in self.phases (excluding yellow)
        if self._current_phase in self.yellow_phase_id:
            if self.current_phase_time == self.yellow_phase_time:
                self._change_phase(self.phases[self.action_before_yellow], interval, 'add')
                self.current_phase = self.action_before_yellow
                self.action_executed = self.action_before_yellow
            else:
                self.current_phase_time += interval
        else:
            if action == self.current_phase:
                self.current_phase_time += interval
            else:
                if self.yellow_phase_time > 0:
                    # yellow(red) phase is arranged behind each green light
                    self._change_phase(self.yellow_phase_id[0], interval)
                    self.action_before_yellow = action
                else:
                    self._change_phase(self.phases[action], interval)
                    self.current_phase = action
                    self.action_executed = action

    def reset(self):
        """
        Reset information, including current_phase, action_before_yellow and action_executed, etc.

        :param: None
        :return: None
        """
        # record phase info
        self.current_phase = 0  # phase id in self.phases (excluding yellow)
        if len(self.phases) == 0:
            self._current_phase = 0
        else:
            self._current_phase = self.phases[0]  # true phase id (including yellow)
        self.eng.set_tl_phase(self.id, self._current_phase)
        self.current_phase_time = 0
        self.action_before_yellow = None
        self.action_executed = None


    def _get_direction(self, road, out=True):
        if out:
            x = road["points"][1]["x"] - road["points"][0]["x"]
            y = road["points"][1]["y"] - road["points"][0]["y"]
        else:
            x = road["points"][-2]["x"] - road["points"][-1]["x"]
            y = road["points"][-2]["y"] - road["points"][-1]["y"]
        tmp = atan2(x, y)
        return tmp if tmp >= 0 else (tmp + 2 * pi)