from typing import Tuple, List
import random


class Grid:
    def __init__(self,
                 min_x: float,
                 min_y: float,
                 step_x: float,
                 step_y: float,
                 index: Tuple[int, int]):
        """
        Attributes:
            min_x, min_y, max_x, max_y: boundary of current grid
            index = (i, j): grid index in the matrix
        """
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = min_x + step_x
        self.max_y = min_y + step_y
        self.index = index

    def in_cell(self, p: Tuple[float, float]):
        if self.min_x <= p[0] <= self.max_x and self.min_y <= p[1] <= self.max_y:
            return True
        else:
            return False

    def sample_point(self):
        x = self.min_x + random.random() * (self.max_x - self.min_x)
        y = self.min_y + random.random() * (self.max_y - self.min_y)

        return x, y

    def equal(self, other):
        return self.index == other.index

    def __eq__(self, other):
        if not type(other) == Grid:
            return False
        return self.index == other.index

    def __hash__(self):
        return hash(self.index)


class GridMap:
    def __init__(self,
                 n: int,
                 min_x: float,
                 min_y: float,
                 max_x: float,
                 max_y: float):
        """
        Geographical map after griding
        Parameters:
             n: cell count
             min_x, min_y, max_x, max_y: boundary of the map
        """
        min_x -= 1e-6
        min_y -= 1e-6
        max_x += 1e-6
        max_y += 1e-6
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        step_x = (max_x - min_x) / n
        step_y = (max_y - min_y) / n
        self.step_x = step_x
        self.step_y = step_y

        # Spatial map, n x n matrix of grids
        self.map: List[List[Grid]] = list()
        for i in range(n):
            self.map.append(list())
            for j in range(n):
                self.map[i].append(Grid(min_x + step_x * i, min_y + step_y * j, step_x, step_y, (i, j)))

    def get_adjacent(self, g: Grid) -> List[Tuple[int, int]]:
        """
        Get 8 adjacent grids of g
        """
        i, j = g.index
        adjacent_index = [(i - 1, j - 1), (i - 1, j), (i - 1, j + 1), (i, j + 1),
                          (i, j - 1), (i + 1, j + 1), (i + 1, j), (i + 1, j - 1)]
        adjacent_index_new = []
        # Remove grids out of bound
        for index in adjacent_index:
            if len(self.map) > index[0] >= 0 and len(self.map[0]) > index[1] >= 0:
                adjacent_index_new.append(index)
        return adjacent_index_new

    def is_adjacent_grids(self, g1: Grid, g2: Grid):
        return True if g2.index in self.get_adjacent(g1) else False

    def get_list_map(self):
        list_map = []
        for li in self.map:
            list_map.extend(li)
        return list_map

    def get_all_transition(self):
        transitions = []
        for g in self.get_list_map():
            # start transition
            transitions.append((Transition(g, g, 1)))
            adjacent_grids = self.get_adjacent(g)
            transitions.append(Transition(g, g, 0))
            for (i, j) in adjacent_grids:
                transitions.append(Transition(g, self.map[i][j]))
            # end transition
            transitions.append(Transition(g, g, 2))
        return transitions

    def get_normal_transition(self):
        transitions = []
        for g in self.get_list_map():
            adjacent_grids = self.get_adjacent(g)
            transitions.append(Transition(g, g, 0))
            for (i, j) in adjacent_grids:
                transitions.append(Transition(g, self.map[i][j]))
        return transitions

    @property
    def size(self):
        return len(self.map) * len(self.map[0])


class Transition:
    def __init__(self, g1: Grid, g2: Grid, flag=0):
        self.g1 = g1
        self.g2 = g2
        self.flag = flag

    def __eq__(self, other):
        if not type(other) == Transition:
            return False
        return self.g1 == other.g1 and self.g2 == other.g2 and self.flag == other.flag

    def __hash__(self):
        return hash(self.g1.index + self.g2.index + (self.flag,))
