from typing import Tuple
import numpy as np
from .ga_framework import MutationStrategy, Individual, MusicIndividual

class TranspositionMutation(MutationStrategy):
    """移调变异"""
    def mutate(self, individual: Individual) -> Individual:
        gene = individual.data.copy()
        shift = np.random.randint(-5, 6)
        if shift == 0: return MusicIndividual(gene)
        
        mask = gene > 1 # 假设 >1 才是音高
        gene[mask] += shift
        # 这里为了演示简单，没有做越界检查，实际需要 clip
        gene[mask] = np.clip(gene[mask], 0, 127) 
        return MusicIndividual(gene)

class PointMutation(MutationStrategy):
    """随机点变异"""
    def __init__(self, prob=0.1):
        self.prob = prob

    def mutate(self, individual: Individual) -> Individual:
        gene = individual.data.copy()
        for i in range(len(gene)):
            if np.random.random() < self.prob:
                gene[i] = np.random.randint(0, 128) # 简单随机
        return MusicIndividual(gene)

class InversionMutation(MutationStrategy):
    """倒影变异"""
    def mutate(self, individual: Individual) -> Individual:
        gene = individual.data.copy()
        # 简单倒影逻辑：以第一个音为轴
        valid_notes = gene[gene > 1]
        if len(valid_notes) > 0:
            pivot = valid_notes[0]
            mask = gene > 1
            gene[mask] = 2 * pivot - gene[mask]
            gene[mask] = np.clip(gene[mask], 0, 127)
        return MusicIndividual(gene)