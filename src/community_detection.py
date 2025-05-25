"""
Módulo para detección de comunidades en redes sociales
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import defaultdict, Counter

# Imports opcionales con manejo de errores mejorado
COMMUNITY_LOUVAIN_AVAILABLE = False
CDLIB_AVAILABLE = False
IGRAPH_AVAILABLE = False
INFOMAP_AVAILABLE = False

try:
    import community as community_louvain
    COMMUNITY_LOUVAIN_AVAILABLE = True
except ImportError:
    logging.warning("python-louvain no está disponible")

try:
    from cdlib import algorithms, evaluation, NodeClustering
    CDLIB_AVAILABLE = True
except ImportError:
    logging.warning("cdlib no está disponible")

try:
    import igraph as ig
    IGRAPH_AVAILABLE = True
except ImportError:
    logging.warning("python-igraph no está disponible")

try:
    import infomap
    INFOMAP_AVAILABLE = True
except ImportError:
    logging.warning("infomap no está disponible")

class CommunityDetector:
    """
    Clase para detectar comunidades en grafos de redes sociales
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el detector de comunidades
        
        Args:
            graph (nx.DiGraph): Grafo de red social
        """
        self.graph = graph
        self.undirected_graph = graph.to_undirected()
        self.communities = {}
        self.logger = logging.getLogger(__name__)
    
    def get_available_methods(self) -> Dict[str, bool]:
        """
        Retorna los métodos de detección de comunidades disponibles
        
        Returns:
            Dict[str, bool]: Mapeo de método a disponibilidad
        """
        methods = {
            'louvain': COMMUNITY_LOUVAIN_AVAILABLE,
            'leiden': IGRAPH_AVAILABLE,
            'infomap': IGRAPH_AVAILABLE,
            'label_propagation': True,  # Siempre disponible con NetworkX
            'greedy_modularity': True,  # Siempre disponible con NetworkX
        }
        
        available_methods = {method: available for method, available in methods.items() if available}
        
        self.logger.info(f"Métodos de detección de comunidades disponibles: {list(available_methods.keys())}")
        
        return methods
    
    def detect_communities_louvain(self, resolution: float = 1.0, 
                                 random_state: int = 42) -> Dict[str, int]:
        """
        Detecta comunidades usando algoritmo de Louvain
        
        Args:
            resolution (float): Parámetro de resolución
            random_state (int): Semilla aleatoria
            
        Returns:
            Dict[str, int]: Mapeo de usuario a ID de comunidad
        """
        if not COMMUNITY_LOUVAIN_AVAILABLE:
            raise ImportError("python-louvain no está disponible. Instale con: pip install python-louvain")
            
        try:
            # Crear grafo con pesos para Louvain
            weighted_graph = self.undirected_graph.copy()
            
            # Asegurar que todas las aristas tengan peso
            for u, v, data in weighted_graph.edges(data=True):
                if 'weight' not in data:
                    weighted_graph[u][v]['weight'] = 1.0
            
            # Aplicar algoritmo de Louvain
            partition = community_louvain.best_partition(
                weighted_graph, 
                resolution=resolution,
                random_state=random_state,
                weight='weight'
            )
            
            self.communities['louvain'] = partition
            
            num_communities = len(set(partition.values()))
            modularity = community_louvain.modularity(partition, weighted_graph, weight='weight')
            
            self.logger.info(f"Louvain: {num_communities} comunidades detectadas, "
                           f"modularidad: {modularity:.4f}")
            
            return partition
            
        except Exception as e:
            self.logger.error(f"Error en detección de Louvain: {str(e)}")
            raise
    
    def detect_communities_leiden(self, resolution: float = 1.0) -> Dict[str, int]:
        """
        Detecta comunidades usando algoritmo de Leiden
        
        Args:
            resolution (float): Parámetro de resolución
            
        Returns:
            Dict[str, int]: Mapeo de usuario a ID de comunidad
        """
        if not IGRAPH_AVAILABLE:
            raise ImportError("python-igraph no está disponible. Instale con: pip install python-igraph leidenalg")
            
        try:
            # Convertir a igraph
            igraph_net = self._networkx_to_igraph(self.undirected_graph)
            
            # Aplicar Leiden
            leiden_communities = igraph_net.community_leiden(
                objective_function='modularity',
                resolution_parameter=resolution,
                weights='weight'
            )
            
            # Convertir resultado
            partition = {}
            for i, community in enumerate(leiden_communities):
                for node_idx in community:
                    node_name = igraph_net.vs[node_idx]['name']
                    partition[node_name] = i
            
            self.communities['leiden'] = partition
            
            num_communities = len(leiden_communities)
            modularity = leiden_communities.modularity
            
            self.logger.info(f"Leiden: {num_communities} comunidades detectadas, "
                           f"modularidad: {modularity:.4f}")
            
            return partition
            
        except Exception as e:
            self.logger.error(f"Error en detección de Leiden: {str(e)}")
            raise
    
    def detect_communities_infomap(self) -> Dict[str, int]:
        """
        Detecta comunidades usando algoritmo Infomap
        
        Returns:
            Dict[str, int]: Mapeo de usuario a ID de comunidad
        """
        if not IGRAPH_AVAILABLE:
            raise ImportError("python-igraph no está disponible. Instale con: pip install python-igraph")
            
        try:
            # Convertir a igraph
            igraph_net = self._networkx_to_igraph(self.graph)  # Usar grafo dirigido
            
            # Aplicar Infomap
            infomap_communities = igraph_net.community_infomap(
                edge_weights='weight',
                vertex_weights=None
            )
            
            # Convertir resultado
            partition = {}
            for i, community in enumerate(infomap_communities):
                for node_idx in community:
                    node_name = igraph_net.vs[node_idx]['name']
                    partition[node_name] = i
            
            self.communities['infomap'] = partition
            
            num_communities = len(infomap_communities)
            modularity = infomap_communities.modularity
            
            self.logger.info(f"Infomap: {num_communities} comunidades detectadas, "
                           f"modularidad: {modularity:.4f}")
            
            return partition
            
        except Exception as e:
            self.logger.error(f"Error en detección de Infomap: {str(e)}")
            raise
    
    def detect_communities_label_propagation(self) -> Dict[str, int]:
        """
        Detecta comunidades usando propagación de etiquetas
        
        Returns:
            Dict[str, int]: Mapeo de usuario a ID de comunidad
        """
        try:
            communities_generator = nx.community.asyn_lpa_communities(
                self.undirected_graph, 
                weight='weight'
            )
            
            # Convertir generador a lista
            communities = list(communities_generator)
            
            # Convertir a diccionario
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            
            self.communities['label_propagation'] = partition
            
            num_communities = len(communities)
            self.logger.info(f"Propagación de etiquetas: {num_communities} comunidades detectadas")
            
            return partition
            
        except Exception as e:
            self.logger.error(f"Error en propagación de etiquetas: {str(e)}")
            raise
    
    def detect_communities_greedy_modularity(self) -> Dict[str, int]:
        """
        Detecta comunidades usando optimización greedy de modularidad
        
        Returns:
            Dict[str, int]: Mapeo de usuario a ID de comunidad
        """
        try:
            communities = nx.community.greedy_modularity_communities(
                self.undirected_graph,
                weight='weight'
            )
            
            # Convertir a diccionario
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            
            self.communities['greedy_modularity'] = partition
            
            num_communities = len(communities)
            modularity = nx.community.modularity(self.undirected_graph, communities, weight='weight')
            
            self.logger.info(f"Greedy modularity: {num_communities} comunidades detectadas, "
                           f"modularidad: {modularity:.4f}")
            
            return partition
            
        except Exception as e:
            self.logger.error(f"Error en greedy modularity: {str(e)}")
            raise
    
    def _networkx_to_igraph(self, nx_graph: nx.Graph) -> 'ig.Graph':
        """
        Convierte grafo NetworkX a igraph
        
        Args:
            nx_graph (nx.Graph): Grafo NetworkX
            
        Returns:
            ig.Graph: Grafo igraph
        """
        try:
            import igraph as ig
            
            # Crear mapeo de nodos a índices
            nodes = list(nx_graph.nodes())
            node_to_idx = {node: i for i, node in enumerate(nodes)}
            
            # Crear lista de aristas
            edges = []
            weights = []
            
            for u, v, data in nx_graph.edges(data=True):
                edges.append((node_to_idx[u], node_to_idx[v]))
                weights.append(data.get('weight', 1.0))
            
            # Crear grafo igraph
            igraph_net = ig.Graph(
                n=len(nodes),
                edges=edges,
                directed=nx_graph.is_directed()
            )
            
            # Agregar nombres de nodos y pesos
            igraph_net.vs['name'] = nodes
            igraph_net.es['weight'] = weights
            
            return igraph_net
            
        except Exception as e:
            self.logger.error(f"Error convirtiendo a igraph: {str(e)}")
            raise
    
    def analyze_community_structure(self, partition: Dict[str, int]) -> Dict:
        """
        Analiza la estructura de comunidades detectadas
        
        Args:
            partition (Dict[str, int]): Mapeo de nodos a comunidades
            
        Returns:
            Dict: Análisis de estructura comunitaria
        """
        # Estadísticas básicas
        communities_list = list(set(partition.values()))
        num_communities = len(communities_list)
        
        # Tamaños de comunidades
        community_sizes = Counter(partition.values())
        
        # Calcular modularidad
        communities_sets = []
        for comm_id in communities_list:
            community_nodes = {node for node, comm in partition.items() if comm == comm_id}
            communities_sets.append(community_nodes)
        
        modularity = nx.community.modularity(self.undirected_graph, communities_sets, weight='weight')
        
        # Análisis de densidad intra e inter-comunitaria
        intra_edges = 0
        inter_edges = 0
        
        for u, v in self.undirected_graph.edges():
            if partition[u] == partition[v]:
                intra_edges += 1
            else:
                inter_edges += 1
        
        total_edges = self.undirected_graph.number_of_edges()
        
        analysis = {
            'num_communities': num_communities,
            'modularity': modularity,
            'community_sizes': dict(community_sizes),
            'avg_community_size': np.mean(list(community_sizes.values())),
            'std_community_size': np.std(list(community_sizes.values())),
            'largest_community_size': max(community_sizes.values()),
            'smallest_community_size': min(community_sizes.values()),
            'intra_community_edges': intra_edges,
            'inter_community_edges': inter_edges,
            'intra_community_ratio': intra_edges / total_edges if total_edges > 0 else 0,
            'inter_community_ratio': inter_edges / total_edges if total_edges > 0 else 0
        }
        
        return analysis
    
    def get_community_dataframe(self, partition: Dict[str, int]) -> pd.DataFrame:
        """
        Convierte partition a DataFrame para análisis
        
        Args:
            partition (Dict[str, int]): Mapeo de nodos a comunidades
            
        Returns:
            pd.DataFrame: DataFrame con información de comunidades
        """
        df = pd.DataFrame([
            {'user_id': user, 'community': comm}
            for user, comm in partition.items()
        ])
        
        return df.set_index('user_id')
    
    def find_bridge_nodes(self, partition: Dict[str, int]) -> List[str]:
        """
        Encuentra nodos puente entre comunidades
        
        Args:
            partition (Dict[str, int]): Mapeo de nodos a comunidades
            
        Returns:
            List[str]: Lista de nodos puente
        """
        bridge_nodes = []
        
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                continue
                
            node_community = partition[node]
            neighbor_communities = {partition[neighbor] for neighbor in neighbors 
                                  if neighbor in partition}
            
            # Es nodo puente si tiene vecinos en múltiples comunidades
            if len(neighbor_communities) > 1:
                bridge_nodes.append(node)
        
        return bridge_nodes
    
    def compare_community_methods(self, methods: List[str] = None) -> pd.DataFrame:
        """
        Compara diferentes métodos de detección de comunidades
        
        Args:
            methods (List[str]): Lista de métodos a comparar
            
        Returns:
            pd.DataFrame: Comparación de métodos
        """
        if methods is None:
            methods = ['louvain', 'leiden', 'label_propagation', 'greedy_modularity']
        
        results = []
        
        for method in methods:
            try:
                if method == 'louvain':
                    partition = self.detect_communities_louvain()
                elif method == 'leiden':
                    partition = self.detect_communities_leiden()
                elif method == 'label_propagation':
                    partition = self.detect_communities_label_propagation()
                elif method == 'greedy_modularity':
                    partition = self.detect_communities_greedy_modularity()
                elif method == 'infomap':
                    partition = self.detect_communities_infomap()
                else:
                    continue
                
                analysis = self.analyze_community_structure(partition)
                
                result = {
                    'method': method,
                    'num_communities': analysis['num_communities'],
                    'modularity': analysis['modularity'],
                    'avg_community_size': analysis['avg_community_size'],
                    'largest_community_size': analysis['largest_community_size']
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error ejecutando método {method}: {str(e)}")
                continue
        
        return pd.DataFrame(results) 