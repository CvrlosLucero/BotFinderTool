"""
Módulo para construir y manejar grafos dirigidos ponderados de redes sociales
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import logging
from collections import defaultdict

class SocialNetworkGraph:
    """
    Clase para construir y manejar grafos dirigidos ponderados de redes sociales
    """
    
    def __init__(self):
        """
        Inicializa el constructor de grafos de redes sociales
        """
        self.graph = nx.DiGraph()
        self.user_features = {}
        self.interaction_weights = defaultdict(float)
        self.logger = logging.getLogger(__name__)
        
    def build_graph_from_interactions(self, interactions_df: pd.DataFrame, 
                                    user_features_df: Optional[pd.DataFrame] = None) -> nx.DiGraph:
        """
        Construye grafo dirigido ponderado desde DataFrame de interacciones
        
        Args:
            interactions_df (pd.DataFrame): DataFrame con interacciones
            user_features_df (pd.DataFrame, optional): DataFrame con características de usuarios
            
        Returns:
            nx.DiGraph: Grafo dirigido ponderado
        """
        self.graph.clear()
        
        # Agregar nodos (usuarios únicos)
        all_users = set(interactions_df['source_user'].unique()) | set(interactions_df['target_user'].unique())
        self.graph.add_nodes_from(all_users)
        
        # Agregar características de usuarios si están disponibles
        if user_features_df is not None:
            self.user_features = user_features_df.to_dict('index')
            for user_id in all_users:
                if user_id in self.user_features:
                    self.graph.nodes[user_id].update(self.user_features[user_id])
                    
        # Agregar aristas ponderadas
        for _, row in interactions_df.iterrows():
            source = row['source_user']
            target = row['target_user']
            weight = row['weight']
            interaction_type = row['interaction_type']
            
            if self.graph.has_edge(source, target):
                # Sumar pesos si la arista ya existe
                self.graph[source][target]['weight'] += weight
                # Agregar tipo de interacción
                if 'interaction_types' not in self.graph[source][target]:
                    self.graph[source][target]['interaction_types'] = set()
                self.graph[source][target]['interaction_types'].add(interaction_type)
            else:
                # Crear nueva arista
                self.graph.add_edge(source, target, 
                                  weight=weight, 
                                  interaction_types={interaction_type})
                                  
        self.logger.info(f"Grafo construido: {self.graph.number_of_nodes()} nodos, "
                        f"{self.graph.number_of_edges()} aristas")
        
        return self.graph
    
    def create_subgraph_by_interaction_type(self, interaction_type: str) -> nx.DiGraph:
        """
        Crea subgrafo filtrado por tipo de interacción
        
        Args:
            interaction_type (str): Tipo de interacción a filtrar
            
        Returns:
            nx.DiGraph: Subgrafo filtrado
        """
        edges_to_keep = []
        
        for source, target, data in self.graph.edges(data=True):
            if 'interaction_types' in data and interaction_type in data['interaction_types']:
                edges_to_keep.append((source, target, data))
                
        subgraph = nx.DiGraph()
        subgraph.add_nodes_from(self.graph.nodes(data=True))
        subgraph.add_edges_from(edges_to_keep)
        
        return subgraph
    
    def get_degree_centrality(self) -> Dict[str, float]:
        """
        Calcula centralidad de grado para todos los nodos
        
        Returns:
            Dict[str, float]: Centralidad de grado por usuario
        """
        return nx.degree_centrality(self.graph)
    
    def get_in_degree_centrality(self) -> Dict[str, float]:
        """
        Calcula centralidad de grado de entrada
        
        Returns:
            Dict[str, float]: Centralidad de grado de entrada por usuario
        """
        return nx.in_degree_centrality(self.graph)
    
    def get_out_degree_centrality(self) -> Dict[str, float]:
        """
        Calcula centralidad de grado de salida
        
        Returns:
            Dict[str, float]: Centralidad de grado de salida por usuario
        """
        return nx.out_degree_centrality(self.graph)
    
    def get_betweenness_centrality(self, k: Optional[int] = None) -> Dict[str, float]:
        """
        Calcula centralidad de intermediación
        
        Args:
            k (int, optional): Número de nodos para muestreo aproximado
            
        Returns:
            Dict[str, float]: Centralidad de intermediación por usuario
        """
        return nx.betweenness_centrality(self.graph, k=k)
    
    def get_closeness_centrality(self) -> Dict[str, float]:
        """
        Calcula centralidad de cercanía
        
        Returns:
            Dict[str, float]: Centralidad de cercanía por usuario
        """
        return nx.closeness_centrality(self.graph)
    
    def get_eigenvector_centrality(self, max_iter: int = 1000) -> Dict[str, float]:
        """
        Calcula centralidad de vector propio
        
        Args:
            max_iter (int): Máximo número de iteraciones
            
        Returns:
            Dict[str, float]: Centralidad de vector propio por usuario
        """
        try:
            return nx.eigenvector_centrality(self.graph, max_iter=max_iter)
        except nx.PowerIterationFailedConvergence:
            self.logger.warning("Centralidad de vector propio no convergió, usando valores aproximados")
            return nx.eigenvector_centrality_numpy(self.graph)
    
    def calculate_pagerank(self, alpha: float = 0.85, max_iter: int = 1000) -> Dict[str, float]:
        """
        Calcula PageRank para todos los nodos
        
        Args:
            alpha (float): Factor de amortiguación
            max_iter (int): Máximo número de iteraciones
            
        Returns:
            Dict[str, float]: Valores de PageRank por usuario
        """
        return nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter, weight='weight')
    
    def calculate_clustering_coefficient(self) -> Dict[str, float]:
        """
        Calcula coeficiente de agrupamiento para todos los nodos
        
        Returns:
            Dict[str, float]: Coeficiente de agrupamiento por usuario
        """
        # Convertir a no dirigido para clustering
        undirected_graph = self.graph.to_undirected()
        return nx.clustering(undirected_graph, weight='weight')
    
    def get_graph_statistics(self) -> Dict:
        """
        Obtiene estadísticas generales del grafo
        
        Returns:
            Dict: Estadísticas del grafo
        """
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph)
        }
        
        # Estadísticas de grados
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        
        stats['in_degree_stats'] = {
            'mean': np.mean(list(in_degrees.values())),
            'std': np.std(list(in_degrees.values())),
            'max': max(in_degrees.values()) if in_degrees else 0,
            'min': min(in_degrees.values()) if in_degrees else 0
        }
        
        stats['out_degree_stats'] = {
            'mean': np.mean(list(out_degrees.values())),
            'std': np.std(list(out_degrees.values())),
            'max': max(out_degrees.values()) if out_degrees else 0,
            'min': min(out_degrees.values()) if out_degrees else 0
        }
        
        # Estadísticas de pesos
        weights = [data['weight'] for _, _, data in self.graph.edges(data=True)]
        if weights:
            stats['weight_stats'] = {
                'mean': np.mean(weights),
                'std': np.std(weights),
                'max': max(weights),
                'min': min(weights)
            }
            
        return stats
    
    def get_strongly_connected_components(self) -> List[Set]:
        """
        Obtiene componentes fuertemente conectados
        
        Returns:
            List[Set]: Lista de componentes fuertemente conectados
        """
        return list(nx.strongly_connected_components(self.graph))
    
    def get_weakly_connected_components(self) -> List[Set]:
        """
        Obtiene componentes débilmente conectados
        
        Returns:
            List[Set]: Lista de componentes débilmente conectados
        """
        return list(nx.weakly_connected_components(self.graph))
    
    def find_shortest_paths(self, source: str, target: str = None) -> Dict:
        """
        Encuentra caminos más cortos desde un nodo fuente
        
        Args:
            source (str): Nodo fuente
            target (str, optional): Nodo objetivo específico
            
        Returns:
            Dict: Caminos más cortos y distancias
        """
        if target:
            try:
                path = nx.shortest_path(self.graph, source, target, weight='weight')
                length = nx.shortest_path_length(self.graph, source, target, weight='weight')
                return {'path': path, 'length': length}
            except nx.NetworkXNoPath:
                return {'path': None, 'length': float('inf')}
        else:
            paths = nx.single_source_shortest_path(self.graph, source)
            lengths = nx.single_source_shortest_path_length(self.graph, source)
            return {'paths': paths, 'lengths': lengths}
    
    def export_to_gexf(self, filepath: str):
        """
        Exporta el grafo a formato GEXF para visualización en Gephi
        
        Args:
            filepath (str): Ruta del archivo de salida
        """
        nx.write_gexf(self.graph, filepath)
        self.logger.info(f"Grafo exportado a {filepath}")
    
    def export_to_graphml(self, filepath: str):
        """
        Exporta el grafo a formato GraphML
        
        Args:
            filepath (str): Ruta del archivo de salida
        """
        nx.write_graphml(self.graph, filepath)
        self.logger.info(f"Grafo exportado a {filepath}")
    
    def get_node_attributes_dataframe(self) -> pd.DataFrame:
        """
        Convierte atributos de nodos a DataFrame
        
        Returns:
            pd.DataFrame: DataFrame con atributos de nodos
        """
        node_data = []
        for node, attributes in self.graph.nodes(data=True):
            row = {'user_id': node}
            row.update(attributes)
            node_data.append(row)
            
        return pd.DataFrame(node_data).set_index('user_id')
    
    def get_edge_list_dataframe(self) -> pd.DataFrame:
        """
        Convierte lista de aristas a DataFrame
        
        Returns:
            pd.DataFrame: DataFrame con aristas y sus atributos
        """
        edge_data = []
        for source, target, attributes in self.graph.edges(data=True):
            row = {
                'source': source,
                'target': target
            }
            row.update(attributes)
            edge_data.append(row)
            
        return pd.DataFrame(edge_data) 