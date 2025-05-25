"""
BotFinderTool - Herramienta de detección de bots en redes sociales usando teoría de grafos
"""

__version__ = "1.0.0"
__author__ = "Grupo 6"
__description__ = "Modelo basado en teoría de grafos para detección de bots en redes sociales"

from .data_loader import CSVDataLoader
from .graph_builder import SocialNetworkGraph
from .bot_detector import BotDetector
from .visualizer import GraphVisualizer

__all__ = [
    'CSVDataLoader',
    'SocialNetworkGraph', 
    'BotDetector',
    'GraphVisualizer'
] 