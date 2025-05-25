"""
Módulo para visualización de grafos y resultados de detección de bots
"""

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import Counter

class GraphVisualizer:
    """
    Clase para visualizar grafos de redes sociales y resultados de detección de bots
    """
    
    def __init__(self, graph: nx.DiGraph):
        """
        Inicializa el visualizador
        
        Args:
            graph (nx.DiGraph): Grafo de red social
        """
        self.graph = graph
        self.logger = logging.getLogger(__name__)
        
        # Configurar estilo de matplotlib
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_degree_distribution(self, save_path: str = None, show_plot: bool = True):
        """
        Visualiza la distribución de grados del grafo
        
        Args:
            save_path (str): Ruta para guardar la imagen
            show_plot (bool): Mostrar el gráfico
        """
        # Calcular distribuciones
        in_degrees = [d for n, d in self.graph.in_degree()]
        out_degrees = [d for n, d in self.graph.out_degree()]
        total_degrees = [d for n, d in self.graph.degree()]
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Distribuciones de Grado de la Red Social', fontsize=16)
        
        # Grado de entrada
        axes[0,0].hist(in_degrees, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0,0].set_title('Distribución de Grado de Entrada')
        axes[0,0].set_xlabel('Grado de Entrada')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].set_yscale('log')
        
        # Grado de salida
        axes[0,1].hist(out_degrees, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0,1].set_title('Distribución de Grado de Salida')
        axes[0,1].set_xlabel('Grado de Salida')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].set_yscale('log')
        
        # Grado total
        axes[1,0].hist(total_degrees, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1,0].set_title('Distribución de Grado Total')
        axes[1,0].set_xlabel('Grado Total')
        axes[1,0].set_ylabel('Frecuencia')
        axes[1,0].set_yscale('log')
        
        # Grado entrada vs salida
        axes[1,1].scatter(in_degrees, out_degrees, alpha=0.6)
        axes[1,1].set_title('Grado de Entrada vs Grado de Salida')
        axes[1,1].set_xlabel('Grado de Entrada')
        axes[1,1].set_ylabel('Grado de Salida')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Distribución de grados guardada en {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_centrality_comparison(self, centralities: Dict[str, Dict[str, float]], 
                                 save_path: str = None, show_plot: bool = True):
        """
        Compara diferentes medidas de centralidad
        
        Args:
            centralities (Dict): Diccionario con diferentes centralidades
            save_path (str): Ruta para guardar la imagen
            show_plot (bool): Mostrar el gráfico
        """
        # Crear DataFrame con centralidades
        df_data = []
        for measure, values in centralities.items():
            for node, centrality in values.items():
                df_data.append({
                    'node': node,
                    'measure': measure,
                    'centrality': centrality
                })
        
        df = pd.DataFrame(df_data)
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Medidas de Centralidad', fontsize=16)
        
        # Distribuciones por medida
        measures = df['measure'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(measures)))
        
        for i, (measure, color) in enumerate(zip(measures, colors)):
            ax = axes[i//2, i%2]
            data = df[df['measure'] == measure]['centrality']
            
            ax.hist(data, bins=30, alpha=0.7, color=color, edgecolor='black')
            ax.set_title(f'Distribución de {measure.replace("_", " ").title()}')
            ax.set_xlabel('Valor de Centralidad')
            ax.set_ylabel('Frecuencia')
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparación de centralidades guardada en {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_network_overview(self, layout: str = 'spring', node_size_attr: str = 'pagerank',
                            node_color_attr: str = 'community', save_path: str = None, 
                            show_plot: bool = True, max_nodes: int = 1000):
        """
        Visualiza el grafo de la red social
        
        Args:
            layout (str): Tipo de layout ('spring', 'circular', 'random')
            node_size_attr (str): Atributo para el tamaño de nodos
            node_color_attr (str): Atributo para el color de nodos
            save_path (str): Ruta para guardar la imagen
            show_plot (bool): Mostrar el gráfico
            max_nodes (int): Máximo número de nodos a visualizar
        """
        # Crear subgrafo si es necesario
        if self.graph.number_of_nodes() > max_nodes:
            # Seleccionar nodos con mayor PageRank
            pagerank = nx.pagerank(self.graph)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes_to_keep = [node for node, _ in top_nodes]
            subgraph = self.graph.subgraph(nodes_to_keep).copy()
        else:
            subgraph = self.graph
        
        # Configurar layout
        if layout == 'spring':
            pos = nx.spring_layout(subgraph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(subgraph)
        else:
            pos = nx.random_layout(subgraph)
        
        # Calcular atributos para visualización
        if node_size_attr == 'pagerank':
            pagerank = nx.pagerank(subgraph)
            node_sizes = [pagerank[node] * 3000 for node in subgraph.nodes()]
        else:
            node_sizes = [subgraph.degree(node) * 10 for node in subgraph.nodes()]
        
        # Colores por comunidades o atributo
        if node_color_attr == 'community':
            try:
                import community as community_louvain
                undirected = subgraph.to_undirected()
                communities = community_louvain.best_partition(undirected)
                node_colors = [communities.get(node, 0) for node in subgraph.nodes()]
            except:
                node_colors = ['blue'] * subgraph.number_of_nodes()
        else:
            node_colors = ['blue'] * subgraph.number_of_nodes()
        
        # Crear visualización
        plt.figure(figsize=(16, 12))
        
        # Dibujar aristas
        nx.draw_networkx_edges(subgraph, pos, alpha=0.2, width=0.5, edge_color='gray')
        
        # Dibujar nodos
        nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.8, cmap=plt.cm.Set3)
        
        plt.title(f'Red Social - {subgraph.number_of_nodes()} nodos, {subgraph.number_of_edges()} aristas', 
                 fontsize=16)
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Visualización de red guardada en {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_bot_detection_results(self, bot_scores: Dict[str, float], 
                                  threshold: float = 0.5, save_path: str = None, 
                                  show_plot: bool = True):
        """
        Visualiza resultados de detección de bots
        
        Args:
            bot_scores (Dict): Scores de detección de bots
            threshold (float): Umbral para clasificar como bot
            save_path (str): Ruta para guardar la imagen
            show_plot (bool): Mostrar el gráfico
        """
        scores = list(bot_scores.values())
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resultados de Detección de Bots', fontsize=16)
        
        # Distribución de scores
        axes[0,0].hist(scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Umbral = {threshold}')
        axes[0,0].set_title('Distribución de Scores de Bot')
        axes[0,0].set_xlabel('Score de Bot')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].legend()
        
        # Clasificación binaria
        bots = [1 if score >= threshold else 0 for score in scores]
        bot_counts = Counter(bots)
        
        axes[0,1].pie([bot_counts[0], bot_counts[1]], 
                     labels=['Usuarios Normales', 'Bots'], 
                     autopct='%1.1f%%',
                     colors=['lightgreen', 'lightcoral'])
        axes[0,1].set_title('Clasificación de Usuarios')
        
        # Top bots
        top_bots = sorted(bot_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        bot_names = [f"Usuario_{i+1}" for i in range(len(top_bots))]
        bot_values = [score for _, score in top_bots]
        
        axes[1,0].barh(bot_names, bot_values, color='coral')
        axes[1,0].set_title('Top 20 Candidatos a Bots')
        axes[1,0].set_xlabel('Score de Bot')
        
        # Evolución de scores (ordenados)
        sorted_scores = sorted(scores, reverse=True)
        axes[1,1].plot(range(len(sorted_scores)), sorted_scores, 'b-', alpha=0.7)
        axes[1,1].axhline(threshold, color='red', linestyle='--', linewidth=2)
        axes[1,1].set_title('Scores Ordenados de Mayor a Menor')
        axes[1,1].set_xlabel('Ranking de Usuario')
        axes[1,1].set_ylabel('Score de Bot')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Resultados de detección guardados en {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def plot_community_analysis(self, communities: Dict[str, int], 
                               bot_scores: Dict[str, float] = None,
                               save_path: str = None, show_plot: bool = True):
        """
        Visualiza análisis de comunidades
        
        Args:
            communities (Dict): Mapeo de usuario a comunidad
            bot_scores (Dict): Scores de detección de bots
            save_path (str): Ruta para guardar la imagen
            show_plot (bool): Mostrar el gráfico
        """
        # Contar tamaños de comunidades
        community_sizes = Counter(communities.values())
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Comunidades', fontsize=16)
        
        # Distribución de tamaños de comunidades
        sizes = list(community_sizes.values())
        axes[0,0].hist(sizes, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0,0].set_title('Distribución de Tamaños de Comunidades')
        axes[0,0].set_xlabel('Tamaño de Comunidad')
        axes[0,0].set_ylabel('Frecuencia')
        axes[0,0].set_yscale('log')
        
        # Top comunidades por tamaño
        top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:15]
        comm_ids = [f"Comm_{comm_id}" for comm_id, _ in top_communities]
        comm_sizes = [size for _, size in top_communities]
        
        axes[0,1].bar(comm_ids, comm_sizes, color='lightgreen')
        axes[0,1].set_title('Top 15 Comunidades por Tamaño')
        axes[0,1].set_xlabel('ID de Comunidad')
        axes[0,1].set_ylabel('Número de Usuarios')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        if bot_scores:
            # Análisis de bots por comunidad
            community_bot_scores = {}
            for user, community in communities.items():
                if user in bot_scores:
                    if community not in community_bot_scores:
                        community_bot_scores[community] = []
                    community_bot_scores[community].append(bot_scores[user])
            
            # Promedio de scores por comunidad
            avg_scores = {}
            for comm, scores in community_bot_scores.items():
                avg_scores[comm] = np.mean(scores) if scores else 0
            
            # Visualizar comunidades sospechosas
            sorted_comms = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:15]
            comm_ids_bot = [f"Comm_{comm_id}" for comm_id, _ in sorted_comms]
            avg_bot_scores = [score for _, score in sorted_comms]
            
            axes[1,0].bar(comm_ids_bot, avg_bot_scores, color='coral')
            axes[1,0].set_title('Comunidades con Mayor Score Promedio de Bot')
            axes[1,0].set_xlabel('ID de Comunidad')
            axes[1,0].set_ylabel('Score Promedio de Bot')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Distribución de scores por comunidad (boxplot de las top 10)
            top_10_comms = [comm_id for comm_id, _ in sorted_comms[:10]]
            box_data = []
            box_labels = []
            
            for comm_id in top_10_comms:
                if comm_id in community_bot_scores:
                    box_data.append(community_bot_scores[comm_id])
                    box_labels.append(f"Comm_{comm_id}")
            
            if box_data:
                axes[1,1].boxplot(box_data, labels=box_labels)
                axes[1,1].set_title('Distribución de Scores de Bot por Comunidad')
                axes[1,1].set_xlabel('Comunidad')
                axes[1,1].set_ylabel('Score de Bot')
                axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Análisis de comunidades guardado en {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def create_interactive_network(self, bot_scores: Dict[str, float] = None,
                                 communities: Dict[str, int] = None,
                                 max_nodes: int = 500) -> go.Figure:
        """
        Crea visualización interactiva de la red con Plotly
        
        Args:
            bot_scores (Dict): Scores de detección de bots
            communities (Dict): Mapeo de usuario a comunidad
            max_nodes (int): Máximo número de nodos a visualizar
            
        Returns:
            go.Figure: Figura de Plotly interactiva
        """
        # Crear subgrafo si es necesario
        if self.graph.number_of_nodes() > max_nodes:
            pagerank = nx.pagerank(self.graph)
            top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes_to_keep = [node for node, _ in top_nodes]
            subgraph = self.graph.subgraph(nodes_to_keep).copy()
        else:
            subgraph = self.graph
        
        # Calcular layout
        pos = nx.spring_layout(subgraph, k=1, iterations=50)
        
        # Preparar datos de aristas
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Crear trace de aristas
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Preparar datos de nodos
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Información del nodo
            degree = subgraph.degree(node)
            text_info = f'Usuario: {node}<br>Grado: {degree}'
            
            if bot_scores and node in bot_scores:
                bot_score = bot_scores[node]
                text_info += f'<br>Score Bot: {bot_score:.3f}'
                node_colors.append(bot_score)
                node_sizes.append(max(10, bot_score * 30))
            else:
                node_colors.append(0)
                node_sizes.append(10)
            
            if communities and node in communities:
                text_info += f'<br>Comunidad: {communities[node]}'
            
            node_text.append(text_info)
        
        # Crear trace de nodos
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.02,
                    title="Score de Bot"
                ),
                line=dict(width=2)
            )
        )
        
        # Crear figura
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Red Social Interactiva - Detección de Bots',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#000000", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame,
                               top_n: int = 20, save_path: str = None, 
                               show_plot: bool = True):
        """
        Visualiza la importancia de características
        
        Args:
            feature_importance_df (pd.DataFrame): DataFrame con importancia de características
            top_n (int): Número de características top a mostrar
            save_path (str): Ruta para guardar la imagen
            show_plot (bool): Mostrar el gráfico
        """
        # Seleccionar top características
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(12, 8))
        
        # Crear gráfico de barras horizontales
        plt.barh(range(len(top_features)), top_features['importance'], 
                color='skyblue', edgecolor='black')
        
        # Configurar etiquetas
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Características Más Importantes para Detección de Bots')
        plt.gca().invert_yaxis()
        
        # Agregar valores en las barras
        for i, v in enumerate(top_features['importance']):
            plt.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Importancia de características guardada en {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close() 