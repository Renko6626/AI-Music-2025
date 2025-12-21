from typing import Tuple
import numpy as np
from .ga_framework import CrossoverStrategy, Individual, MusicIndividual

class OnePointCrossover(CrossoverStrategy):
    def cross(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        p1_data, p2_data = parent1.data, parent2.data
        point = np.random.randint(1, len(p1_data)-1)
        
        c1_data = np.concatenate([p1_data[:point], p2_data[point:]])
        c2_data = np.concatenate([p2_data[:point], p1_data[point:]])
        
        return MusicIndividual(c1_data), MusicIndividual(c2_data)
