"""
Módulo principal para detección de bots en redes sociales usando análisis de grafos
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
import logging
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings

try:
    from graph_builder import SocialNetworkGraph
    from community_detection import CommunityDetector
except ImportError:
    from .graph_builder import SocialNetworkGraph
    from .community_detection import CommunityDetector

class BotDetector:
    """
    Detector de bots usando análisis de grafos y machine learning
    """
    
    def __init__(self, social_graph: SocialNetworkGraph):
        """
        Inicializa el detector de bots
        
        Args:
            social_graph (SocialNetworkGraph): Grafo de red social construido
        """
        self.graph = social_graph.graph
        self.social_graph = social_graph
        self.community_detector = CommunityDetector(self.graph)
        self.features_df = None
        self.bot_scores = {}
        self.models = {}
        self.logger = logging.getLogger(__name__)
        
    def extract_graph_features(self) -> pd.DataFrame:
        """
        Extrae características basadas en grafos para todos los usuarios
        
        Returns:
            pd.DataFrame: DataFrame con características de grafos
        """
        features = {}
        
        # Centralidades
        self.logger.info("Calculando centralidades...")
        degree_centrality = self.social_graph.get_degree_centrality()
        in_degree_centrality = self.social_graph.get_in_degree_centrality()
        out_degree_centrality = self.social_graph.get_out_degree_centrality()
        betweenness_centrality = self.social_graph.get_betweenness_centrality()
        closeness_centrality = self.social_graph.get_closeness_centrality()
        
        try:
            eigenvector_centrality = self.social_graph.get_eigenvector_centrality()
        except:
            self.logger.warning("No se pudo calcular centralidad de vector propio")
            eigenvector_centrality = {node: 0.0 for node in self.graph.nodes()}
        
        # PageRank
        self.logger.info("Calculando PageRank...")
        pagerank = self.social_graph.calculate_pagerank()
        
        # Coeficiente de agrupamiento
        self.logger.info("Calculando coeficientes de agrupamiento...")
        clustering_coefficient = self.social_graph.calculate_clustering_coefficient()
        
        # Detectar comunidades
        self.logger.info("Detectando comunidades...")
        try:
            communities = self.community_detector.detect_communities_louvain()
        except:
            self.logger.warning("No se pudieron detectar comunidades")
            communities = {node: 0 for node in self.graph.nodes()}
        
        # Características de grado
        in_degrees = dict(self.graph.in_degree())
        out_degrees = dict(self.graph.out_degree())
        total_degrees = dict(self.graph.degree())
        
        # Características de pesos de aristas
        weighted_in_degrees = dict(self.graph.in_degree(weight='weight'))
        weighted_out_degrees = dict(self.graph.out_degree(weight='weight'))
        
        # Construir DataFrame de características
        nodes = list(self.graph.nodes())
        
        features_data = []
        for node in nodes:
            feature_row = {
                'user_id': node,
                'degree_centrality': degree_centrality.get(node, 0),
                'in_degree_centrality': in_degree_centrality.get(node, 0),
                'out_degree_centrality': out_degree_centrality.get(node, 0),
                'betweenness_centrality': betweenness_centrality.get(node, 0),
                'closeness_centrality': closeness_centrality.get(node, 0),
                'eigenvector_centrality': eigenvector_centrality.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'clustering_coefficient': clustering_coefficient.get(node, 0),
                'community': communities.get(node, 0),
                'in_degree': in_degrees.get(node, 0),
                'out_degree': out_degrees.get(node, 0),
                'total_degree': total_degrees.get(node, 0),
                'weighted_in_degree': weighted_in_degrees.get(node, 0),
                'weighted_out_degree': weighted_out_degrees.get(node, 0),
            }
            
            # Ratios y características derivadas
            if feature_row['in_degree'] > 0:
                feature_row['follower_following_ratio'] = feature_row['out_degree'] / feature_row['in_degree']
            else:
                feature_row['follower_following_ratio'] = 0
                
            if feature_row['total_degree'] > 0:
                feature_row['in_degree_ratio'] = feature_row['in_degree'] / feature_row['total_degree']
                feature_row['out_degree_ratio'] = feature_row['out_degree'] / feature_row['total_degree']
            else:
                feature_row['in_degree_ratio'] = 0
                feature_row['out_degree_ratio'] = 0
            
            features_data.append(feature_row)
        
        self.features_df = pd.DataFrame(features_data).set_index('user_id')
        
        # Agregar características adicionales si hay atributos de nodos
        node_attributes_df = self.social_graph.get_node_attributes_dataframe()
        if not node_attributes_df.empty:
            self.features_df = self.features_df.join(node_attributes_df, how='left')
        
        self.logger.info(f"Extraídas {len(self.features_df.columns)} características para {len(self.features_df)} usuarios")
        
        return self.features_df
    
    def detect_anomalies_isolation_forest(self, contamination: float = 0.1) -> Dict[str, float]:
        """
        Detecta usuarios anómalos usando Isolation Forest
        
        Args:
            contamination (float): Proporción esperada de outliers
            
        Returns:
            Dict[str, float]: Scores de anomalía por usuario (mayor = más anómalo)
        """
        if self.features_df is None:
            self.extract_graph_features()
        
        # Seleccionar características numéricas
        numeric_features = self.features_df.select_dtypes(include=[np.number])
        
        # Normalizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(numeric_features.fillna(0))
        
        # Entrenar Isolation Forest
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        anomaly_scores = isolation_forest.fit_predict(features_scaled)
        decision_scores = isolation_forest.decision_function(features_scaled)
        
        # Convertir a diccionario (invertir scores para que mayor = más anómalo)
        anomaly_dict = {}
        for i, user_id in enumerate(numeric_features.index):
            # Normalizar score a [0, 1] donde 1 es más anómalo
            anomaly_dict[user_id] = (1 - decision_scores[i]) / 2
        
        self.bot_scores['isolation_forest'] = anomaly_dict
        self.models['isolation_forest'] = {'model': isolation_forest, 'scaler': scaler}
        
        self.logger.info(f"Isolation Forest: {np.sum(anomaly_scores == -1)} usuarios anómalos detectados")
        
        return anomaly_dict
    
    def detect_bots_clustering(self, eps: float = 0.5, min_samples: int = 5) -> Dict[str, int]:
        """
        Detecta clusters de bots usando DBSCAN
        
        Args:
            eps (float): Distancia máxima entre puntos del mismo cluster
            min_samples (int): Número mínimo de puntos para formar cluster
            
        Returns:
            Dict[str, int]: Mapeo de usuario a cluster (-1 = outlier/posible bot)
        """
        if self.features_df is None:
            self.extract_graph_features()
        
        # Seleccionar características numéricas
        numeric_features = self.features_df.select_dtypes(include=[np.number])
        
        # Normalizar características
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(numeric_features.fillna(0))
        
        # Aplicar DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Convertir a diccionario
        cluster_dict = {}
        for i, user_id in enumerate(numeric_features.index):
            cluster_dict[user_id] = cluster_labels[i]
        
        # Contar outliers (posibles bots)
        num_outliers = np.sum(cluster_labels == -1)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        self.bot_scores['dbscan_clusters'] = cluster_dict
        self.models['dbscan'] = {'model': dbscan, 'scaler': scaler}
        
        self.logger.info(f"DBSCAN: {num_clusters} clusters, {num_outliers} outliers detectados")
        
        return cluster_dict
    
    def calculate_behavioral_features(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula características de comportamiento sospechoso de bots
        
        Returns:
            Dict[str, Dict[str, float]]: Características de comportamiento por usuario
        """
        behavioral_features = {}
        
        for node in self.graph.nodes():
            features = {}
            
            # Análisis de patrones de seguimiento
            neighbors = list(self.graph.neighbors(node))
            predecessors = list(self.graph.predecessors(node))
            
            # Ratio de reciprocidad
            mutual_connections = len(set(neighbors) & set(predecessors))
            total_connections = len(set(neighbors) | set(predecessors))
            features['reciprocity_ratio'] = mutual_connections / total_connections if total_connections > 0 else 0
            
            # Diversidad de tipos de interacción
            interaction_types = set()
            for neighbor in neighbors:
                edge_data = self.graph[node][neighbor]
                if 'interaction_types' in edge_data:
                    interaction_types.update(edge_data['interaction_types'])
            
            features['interaction_diversity'] = len(interaction_types)
            
            # Patrones temporales (si hay timestamps)
            edge_weights = []
            for neighbor in neighbors:
                edge_data = self.graph[node][neighbor]
                edge_weights.append(edge_data.get('weight', 1.0))
            
            if edge_weights:
                features['avg_interaction_weight'] = np.mean(edge_weights)
                features['std_interaction_weight'] = np.std(edge_weights)
                features['max_interaction_weight'] = np.max(edge_weights)
            else:
                features['avg_interaction_weight'] = 0
                features['std_interaction_weight'] = 0
                features['max_interaction_weight'] = 0
            
            # Coeficiente de actividad (grado de salida vs entrada)
            out_degree = self.graph.out_degree(node)
            in_degree = self.graph.in_degree(node)
            features['activity_coefficient'] = out_degree / (in_degree + 1)  # +1 para evitar división por 0
            
            behavioral_features[node] = features
        
        return behavioral_features
    
    def train_supervised_model(self, ground_truth_df: pd.DataFrame, 
                             test_size: float = 0.3) -> Dict[str, Any]:
        """
        Entrena modelo supervisado para detección de bots
        
        Args:
            ground_truth_df (pd.DataFrame): DataFrame con labels verdaderos
            test_size (float): Proporción de datos para test
            
        Returns:
            Dict[str, Any]: Resultados del entrenamiento y evaluación
        """
        if self.features_df is None:
            self.extract_graph_features()
        
        # Combinar características con labels
        combined_df = self.features_df.join(ground_truth_df, how='inner')
        
        if 'is_bot' not in combined_df.columns:
            raise ValueError("Columna 'is_bot' no encontrada en ground truth")
        
        # Preparar datos
        X = combined_df.select_dtypes(include=[np.number]).drop(columns=['is_bot'], errors='ignore')
        y = combined_df['is_bot'].astype(int)
        
        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.fillna(0))
        X_test_scaled = scaler.transform(X_test.fillna(0))
        
        # Entrenar Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Predicciones
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluación
        classification_rep = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except:
            auc_score = None
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Guardar modelo
        self.models['random_forest'] = {
            'model': rf_model,
            'scaler': scaler,
            'features': X.columns.tolist()
        }
        
        # Predicciones para todos los usuarios
        all_features_scaled = scaler.transform(X.fillna(0))
        all_predictions = rf_model.predict_proba(all_features_scaled)[:, 1]
        
        prediction_dict = {}
        for i, user_id in enumerate(X.index):
            prediction_dict[user_id] = all_predictions[i]
        
        self.bot_scores['random_forest'] = prediction_dict
        
        results = {
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix,
            'auc_score': auc_score,
            'feature_importance': feature_importance,
            'test_accuracy': classification_rep['accuracy'],
            'test_f1': classification_rep['macro avg']['f1-score']
        }
        
        self.logger.info(f"Modelo supervisado entrenado - Accuracy: {results['test_accuracy']:.3f}, "
                        f"F1: {results['test_f1']:.3f}")
        
        return results
    
    def combine_detection_methods(self, weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Combina múltiples métodos de detección en un score final
        
        Args:
            weights (Dict[str, float]): Pesos para cada método
            
        Returns:
            Dict[str, float]: Scores finales combinados
        """
        if weights is None:
            weights = {
                'isolation_forest': 0.3,
                'random_forest': 0.5,
                'dbscan_clusters': 0.2
            }
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        combined_scores = {}
        all_users = set()
        
        # Recopilar todos los usuarios
        for method_scores in self.bot_scores.values():
            all_users.update(method_scores.keys())
        
        # Combinar scores
        for user in all_users:
            final_score = 0.0
            total_weight_used = 0.0
            
            for method, weight in weights.items():
                if method in self.bot_scores and user in self.bot_scores[method]:
                    score = self.bot_scores[method][user]
                    
                    # Normalizar score de DBSCAN (outliers = 1, otros = 0)
                    if method == 'dbscan_clusters':
                        score = 1.0 if score == -1 else 0.0
                    
                    final_score += score * weight
                    total_weight_used += weight
            
            # Normalizar por peso total usado
            if total_weight_used > 0:
                combined_scores[user] = final_score / total_weight_used
            else:
                combined_scores[user] = 0.0
        
        self.bot_scores['combined'] = combined_scores
        
        return combined_scores
    
    def get_top_bot_candidates(self, method: str = 'combined', 
                              top_n: int = 100) -> pd.DataFrame:
        """
        Obtiene los principales candidatos a bots
        
        Args:
            method (str): Método de detección a usar
            top_n (int): Número de candidatos a retornar
            
        Returns:
            pd.DataFrame: DataFrame con candidatos ordenados por score
        """
        if method not in self.bot_scores:
            raise ValueError(f"Método '{method}' no disponible. Métodos disponibles: {list(self.bot_scores.keys())}")
        
        scores = self.bot_scores[method]
        
        # Crear DataFrame
        candidates_df = pd.DataFrame([
            {'user_id': user, 'bot_score': score}
            for user, score in scores.items()
        ]).sort_values('bot_score', ascending=False)
        
        # Agregar características si están disponibles
        if self.features_df is not None:
            candidates_df = candidates_df.set_index('user_id').join(
                self.features_df, how='left'
            ).reset_index()
        
        return candidates_df.head(top_n)
    
    def analyze_bot_communities(self, bot_threshold: float = 0.7) -> Dict:
        """
        Analiza la distribución de bots en comunidades
        
        Args:
            bot_threshold (float): Umbral para clasificar como bot
            
        Returns:
            Dict: Análisis de bots por comunidad
        """
        if 'combined' not in self.bot_scores:
            self.combine_detection_methods()
        
        # Detectar comunidades si no están en features
        if self.features_df is None or 'community' not in self.features_df.columns:
            self.extract_graph_features()
        
        # Identificar bots
        bot_users = {user for user, score in self.bot_scores['combined'].items() 
                    if score >= bot_threshold}
        
        # Analizar por comunidad
        community_analysis = {}
        
        for user_id, row in self.features_df.iterrows():
            community = row.get('community', 0)
            
            if community not in community_analysis:
                community_analysis[community] = {
                    'total_users': 0,
                    'bot_users': 0,
                    'bot_ratio': 0.0,
                    'avg_bot_score': 0.0,
                    'users': []
                }
            
            community_analysis[community]['total_users'] += 1
            community_analysis[community]['users'].append(user_id)
            
            if user_id in bot_users:
                community_analysis[community]['bot_users'] += 1
        
        # Calcular ratios y promedios
        for community, data in community_analysis.items():
            if data['total_users'] > 0:
                data['bot_ratio'] = data['bot_users'] / data['total_users']
                
                # Calcular score promedio de bots en la comunidad
                community_scores = [
                    self.bot_scores['combined'][user] 
                    for user in data['users'] 
                    if user in self.bot_scores['combined']
                ]
                
                data['avg_bot_score'] = np.mean(community_scores) if community_scores else 0.0
        
        return community_analysis
    
    def export_results(self, filepath: str, method: str = 'combined'):
        """
        Exporta resultados de detección a archivo CSV
        
        Args:
            filepath (str): Ruta del archivo de salida
            method (str): Método de detección a exportar
        """
        results_df = self.get_top_bot_candidates(method=method, top_n=len(self.bot_scores[method]))
        results_df.to_csv(filepath, index=False)
        self.logger.info(f"Resultados exportados a {filepath}")
    
    def apply_academic_heuristic_rules(self) -> Dict[str, float]:
        """
        Aplica reglas heurísticas académicas específicas para detección de bots
        basadas en el proyecto de investigación
        
        Returns:
            Dict[str, float]: Scores heurísticos por usuario
        """
        if self.features_df is None:
            self.extract_graph_features()
        
        heuristic_scores = {}
        
        for user_id, features in self.features_df.iterrows():
            score = 0.0
            
            # Regla 1: Bot tiene muchas conexiones pero nadie importante lo sigue
            if features['out_degree'] > features['in_degree'] * 2:
                score += 0.3
            
            # Regla 2: PageRank muy bajo (nadie importante lo sigue)
            if features['pagerank'] < 0.001:
                score += 0.2
                
            # Regla 3: Publica muy seguido pero no está bien conectado
            if features['out_degree'] > 30 and features['clustering_coefficient'] < 0.1:
                score += 0.25
                
            # Regla 4: No está bien conectado con otras cuentas humanas
            # (coeficiente de agrupamiento bajo)
            if features['clustering_coefficient'] < 0.05:
                score += 0.15
                
            # Regla 5: Alta actividad de salida, baja de entrada
            if features['follower_following_ratio'] > 3.0:
                score += 0.2
                
            # Regla 6: Centralidad de intermediación muy baja
            if features['betweenness_centrality'] < 0.001:
                score += 0.1
                
            # Regla 7: Grado de salida muy alto comparado con centralidad
            if features['out_degree'] > 40 and features['eigenvector_centrality'] < 0.05:
                score += 0.2
                
            # Regla 8: Patrón de bot (muchas menciones/retweets, pocas respuestas)
            behavioral_features = self.calculate_behavioral_features()
            if user_id in behavioral_features:
                if behavioral_features[user_id]['interaction_diversity'] < 2:
                    score += 0.15
                    
                if behavioral_features[user_id]['reciprocity_ratio'] < 0.1:
                    score += 0.15
                    
            # Normalizar score a [0,1]
            heuristic_scores[user_id] = min(1.0, score)
            
        self.bot_scores['academic_heuristics'] = heuristic_scores
        self.logger.info(f"Reglas heurísticas académicas aplicadas a {len(heuristic_scores)} usuarios")
        
        return heuristic_scores
    
    def get_academic_analysis_report(self, ground_truth_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Genera reporte de análisis académico siguiendo la metodología del proyecto
        
        Args:
            ground_truth_df (pd.DataFrame, optional): Ground truth para evaluación
            
        Returns:
            Dict: Reporte académico completo
        """
        if self.features_df is None:
            self.extract_graph_features()
            
        # Aplicar reglas heurísticas académicas
        heuristic_scores = self.apply_academic_heuristic_rules()
        
        # Estadísticas del grafo
        graph_stats = self.social_graph.get_graph_statistics()
        
        # Detección de comunidades
        try:
            communities = self.community_detector.detect_communities_louvain()
            community_analysis = self.community_detector.analyze_community_structure(communities)
        except:
            communities = {}
            community_analysis = {}
        
        # Análisis de centralidades
        centralities = {
            'pagerank': self.social_graph.calculate_pagerank(),
            'degree': self.social_graph.get_degree_centrality(),
            'betweenness': self.social_graph.get_betweenness_centrality(),
            'clustering': self.social_graph.calculate_clustering_coefficient()
        }
        
        # Crear reporte académico
        report = {
            'metodologia': {
                'tipo_grafo': 'Dirigido ponderado',
                'algoritmos_centralidad': ['PageRank', 'Grado', 'Intermediación', 'Agrupamiento'],
                'algoritmo_comunidades': 'Louvain',
                'reglas_heuristicas': 'Implementadas según proyecto académico'
            },
            'estadisticas_grafo': graph_stats,
            'analisis_comunidades': community_analysis,
            'distribuciones_centralidad': {
                'pagerank_stats': {
                    'media': np.mean(list(centralities['pagerank'].values())),
                    'std': np.std(list(centralities['pagerank'].values())),
                    'max': max(centralities['pagerank'].values()),
                    'min': min(centralities['pagerank'].values())
                },
                'clustering_stats': {
                    'media': np.mean(list(centralities['clustering'].values())),
                    'std': np.std(list(centralities['clustering'].values())),
                    'max': max(centralities['clustering'].values()),
                    'min': min(centralities['clustering'].values())
                }
            },
            'deteccion_bots': {
                'metodo_principal': 'Reglas heurísticas académicas',
                'total_usuarios': len(heuristic_scores),
                'candidatos_bots_umbral_07': len([s for s in heuristic_scores.values() if s >= 0.7]),
                'candidatos_bots_umbral_05': len([s for s in heuristic_scores.values() if s >= 0.5]),
                'score_promedio': np.mean(list(heuristic_scores.values()))
            }
        }
        
        # Evaluación si hay ground truth
        if ground_truth_df is not None:
            try:
                true_labels = []
                predicted_scores = []
                
                for user_id in ground_truth_df.index:
                    if user_id in heuristic_scores:
                        true_labels.append(ground_truth_df.loc[user_id, 'is_bot'])
                        predicted_scores.append(heuristic_scores[user_id])
                
                if true_labels:
                    from sklearn.metrics import roc_auc_score, average_precision_score
                    
                    auc = roc_auc_score(true_labels, predicted_scores)
                    ap = average_precision_score(true_labels, predicted_scores)
                    
                    # Umbral óptimo
                    thresholds = [0.3, 0.5, 0.7]
                    best_f1 = 0
                    best_threshold = 0.5
                    
                    for threshold in thresholds:
                        predictions = [1 if s >= threshold else 0 for s in predicted_scores]
                        from sklearn.metrics import f1_score
                        f1 = f1_score(true_labels, predictions)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
                    
                    report['evaluacion'] = {
                        'auc_roc': auc,
                        'average_precision': ap,
                        'mejor_f1_score': best_f1,
                        'umbral_optimo': best_threshold,
                        'usuarios_evaluados': len(true_labels),
                        'bots_reales': sum(true_labels)
                    }
                    
            except Exception as e:
                self.logger.error(f"Error en evaluación académica: {e}")
        
        return report 