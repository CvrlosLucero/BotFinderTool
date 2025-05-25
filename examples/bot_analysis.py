"""
BotFinderTool - Sistema Unificado de Detección de Bots en Redes Sociales
Análisis Integrado de Agentes Automatizados usando Teoría de Grafos

Sistema que combina automáticamente:
- Reglas heurísticas académicas
- Métodos de machine learning  
- Análisis de grafos y comunidades
- Visualizaciones completas

Autores: Carlos Lucero, Samantha Montero, Camilo Becerra
Universidad del Rosario - Teoría de Grafos
"""

import pandas as pd
import numpy as np
import logging
import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Importar módulos del proyecto
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import CSVDataLoader
from graph_builder import SocialNetworkGraph
from community_detection import CommunityDetector
from bot_detector import BotDetector
from visualizer import GraphVisualizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedBotAnalyzer:
    """
    Sistema unificado que combina todos los métodos de detección de bots
    """
    
    def __init__(self):
        self.data_loader = None
        self.graph_builder = None
        self.bot_detector = None
        self.visualizer = None
        self.results = {}
        
    def generate_comprehensive_dataset(self, n_users: int = 1000, n_bots: int = 80):
        """
        Genera un dataset comprehensivo que combina patrones académicos y generales
        """
        logger.info("Generando dataset comprehensivo...")
        
        # Crear directorio de datos si no existe
        os.makedirs('../data', exist_ok=True)
        
        # Parámetros del dataset
        np.random.seed(42)
        n_humans = n_users - n_bots
        
        # Generar usuarios con nomenclatura mixta
        bot_users = [f"bot_{i:03d}" for i in range(n_bots)]
        human_users = [f"user_{i:04d}" for i in range(n_humans)]
        all_users = bot_users + human_users
        
        interactions = []
        
        logger.info("Implementando patrones híbridos de comportamiento...")
        
        # PATRÓN 1: Bots forman clusters artificiales (académico)
        for i in range(300):
            source = np.random.choice(bot_users)
            target = np.random.choice(bot_users)
            if source != target:
                interactions.append({
                    'source_user': source,
                    'target_user': target,
                    'interaction_type': np.random.choice(['follow', 'retweet', 'mention']),
                    'weight': np.random.randint(1, 3),
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 15))
                })
        
        # PATRÓN 2: Bots spam masivo (académico + general)
        for i in range(1200):
            source = np.random.choice(bot_users)
            target = np.random.choice(human_users)
            interactions.append({
                'source_user': source,
                'target_user': target,
                'interaction_type': np.random.choice(['follow', 'mention'], p=[0.7, 0.3]),
                'weight': np.random.randint(1, 2),
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 7))
            })
        
        # PATRÓN 3: Interacciones humanas auténticas
        for i in range(2800):
            source = np.random.choice(human_users)
            target = np.random.choice(all_users)
            if source != target:
                interactions.append({
                    'source_user': source,
                    'target_user': target,
                    'interaction_type': np.random.choice(['follow', 'retweet', 'mention', 'reply', 'like'], 
                                                       p=[0.25, 0.25, 0.20, 0.15, 0.15]),
                    'weight': np.random.randint(1, 4),
                    'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 365))
                })
        
        # PATRÓN 4: Followers legítimos a bots
        for i in range(250):
            source = np.random.choice(human_users)
            target = np.random.choice(bot_users)
            interactions.append({
                'source_user': source,
                'target_user': target,
                'interaction_type': 'follow',
                'weight': 1,
                'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(30, 200))
            })
        
        # PATRÓN 5: Bots coordinados (general)
        bot_groups = [bot_users[i:i+10] for i in range(0, len(bot_users), 10)]
        for group in bot_groups:
            for _ in range(50):
                if len(group) > 1:
                    source, target = np.random.choice(group, 2, replace=False)
                    interactions.append({
                        'source_user': source,
                        'target_user': target,
                        'interaction_type': np.random.choice(['follow', 'retweet']),
                        'weight': np.random.randint(1, 3),
                        'timestamp': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(1, 10))
                    })
        
        # Guardar interacciones
        interactions_df = pd.DataFrame(interactions)
        interactions_df.to_csv('../data/interactions.csv', index=False)
        
        # Generar características de usuarios con patrones híbridos
        user_features = []
        for user in all_users:
            is_bot = user.startswith('bot_')
            
            if is_bot:
                # Características de bots combinando patrones académicos y generales
                features = {
                    'user_id': user,
                    'followers_count': np.random.randint(5, 120),  # Generalmente bajo
                    'following_count': np.random.randint(300, 1800),  # Generalmente alto
                    'tweets_count': np.random.randint(20, 400),  # Variable
                    'account_age_days': np.random.randint(1, 150),  # Generalmente nuevos
                    'verified': False,  # Nunca verificados
                    'profile_pic': np.random.choice([True, False], p=[0.15, 0.85]),  # Pocos con foto
                    'bio_length': np.random.randint(0, 25)  # Biografías cortas
                }
            else:
                # Características humanas diversas
                features = {
                    'user_id': user,
                    'followers_count': np.random.randint(20, 1200),
                    'following_count': np.random.randint(30, 500),
                    'tweets_count': np.random.randint(50, 4000),
                    'account_age_days': np.random.randint(180, 2500),
                    'verified': np.random.choice([True, False], p=[0.04, 0.96]),
                    'profile_pic': np.random.choice([True, False], p=[0.88, 0.12]),
                    'bio_length': np.random.randint(15, 160)
                }
            
            user_features.append(features)
        
        user_features_df = pd.DataFrame(user_features)
        user_features_df.to_csv('../data/user_features.csv', index=False)
        
        # Ground truth comprehensivo
        ground_truth = []
        for user in all_users:
            is_bot = user.startswith('bot_')
            if is_bot:
                # Asignar tipos específicos de bots
                bot_type = np.random.choice(['spam_bot', 'automated_agent', 'fake_follower'], 
                                          p=[0.6, 0.3, 0.1])
            else:
                bot_type = 'human'
                
            ground_truth.append({
                'user_id': user,
                'is_bot': is_bot,
                'bot_type': bot_type
            })
        
        ground_truth_df = pd.DataFrame(ground_truth)
        ground_truth_df.to_csv('../data/ground_truth.csv', index=False)
        
        logger.info(f"Dataset comprehensivo generado: {len(interactions)} interacciones, {len(all_users)} usuarios ({n_bots} bots)")
        return interactions_df, user_features_df, ground_truth_df
    
    def load_or_generate_data(self, use_existing: bool = False):
        """
        Carga datos existentes o genera nuevos datos comprehensivos
        """
        if use_existing:
            try:
                self.data_loader = CSVDataLoader()
                interactions_df = self.data_loader.load_interactions('../data/interactions.csv')
                user_features_df = self.data_loader.load_user_features('../data/user_features.csv')
                ground_truth_df = self.data_loader.load_ground_truth('../data/ground_truth.csv')
                logger.info("Datos cargados desde archivos existentes")
            except Exception as e:
                logger.info(f"No se pudieron cargar datos existentes ({e}), generando nuevos...")
                interactions_df, user_features_df, ground_truth_df = self.generate_comprehensive_dataset()
                self.data_loader = CSVDataLoader()
        else:
            interactions_df, user_features_df, ground_truth_df = self.generate_comprehensive_dataset()
            self.data_loader = CSVDataLoader()
        
        # Preprocesar datos
        interactions_df = self.data_loader.preprocess_interactions(interactions_df)
        summary = self.data_loader.get_interaction_summary(interactions_df)
        
        logger.info(f"Dataset procesado:")
        logger.info(f"  - Total interacciones: {summary['total_interactions']}")
        logger.info(f"  - Usuarios únicos: {summary['unique_users']}")
        logger.info(f"  - Tipos de interacción: {list(summary['interaction_types'].keys())}")
        
        return interactions_df, user_features_df, ground_truth_df
    
    def build_network_graph(self, interactions_df: pd.DataFrame, user_features_df: pd.DataFrame):
        """
        Construye el grafo de red social
        """
        logger.info("Construyendo grafo dirigido ponderado G = (V, E)")
        
        self.graph_builder = SocialNetworkGraph()
        graph = self.graph_builder.build_graph_from_interactions(interactions_df, user_features_df)
        
        stats = self.graph_builder.get_graph_statistics()
        logger.info(f"Grafo G = (V, E) construido:")
        logger.info(f"  - |V| (vértices/usuarios): {stats['nodes']}")
        logger.info(f"  - |E| (aristas/interacciones): {stats['edges']}")
        logger.info(f"  - Densidad: {stats['density']:.6f}")
        logger.info(f"  - Conectividad: {'Débilmente conectado' if stats['is_connected'] else 'Desconectado'}")
        
        return graph, stats
    
    def calculate_graph_metrics(self):
        """
        Calcula todas las métricas estructurales del grafo
        """
        logger.info("Calculando métricas estructurales completas...")
        
        # Centralidades
        pagerank = self.graph_builder.calculate_pagerank()
        degree_centrality = self.graph_builder.get_degree_centrality()
        betweenness = self.graph_builder.get_betweenness_centrality()
        clustering = self.graph_builder.calculate_clustering_coefficient()
        
        logger.info(f"  - PageRank: μ={np.mean(list(pagerank.values())):.6f}")
        logger.info(f"  - Centralidad de grado: μ={np.mean(list(degree_centrality.values())):.6f}")
        logger.info(f"  - Intermediación: μ={np.mean(list(betweenness.values())):.6f}")
        logger.info(f"  - Agrupamiento: μ={np.mean(list(clustering.values())):.6f}")
        
        return {
            'pagerank': pagerank,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness,
            'clustering_coefficient': clustering
        }
    
    def detect_communities(self):
        """
        Detecta comunidades usando algoritmo de Louvain
        """
        logger.info("Detectando comunidades (Algoritmo de Louvain)")
        
        community_detector = CommunityDetector(self.graph_builder.graph)
        communities = community_detector.detect_communities_louvain()
        community_analysis = community_detector.analyze_community_structure(communities)
        
        logger.info(f"Comunidades detectadas:")
        logger.info(f"  - Número de comunidades: {community_analysis['num_communities']}")
        logger.info(f"  - Modularidad: {community_analysis['modularity']:.4f}")
        logger.info(f"  - Tamaño promedio: {community_analysis['avg_community_size']:.1f}")
        logger.info(f"  - Comunidad más grande: {community_analysis['largest_community_size']}")
        
        return communities, community_analysis
    
    def apply_unified_bot_detection(self, ground_truth_df: pd.DataFrame):
        """
        Aplica todos los métodos de detección de bots de forma unificada
        """
        logger.info("Aplicando detección unificada de agentes automatizados...")
        
        self.bot_detector = BotDetector(self.graph_builder)
        
        # 1. Extraer características del grafo
        features_df = self.bot_detector.extract_graph_features()
        logger.info(f"Características extraídas: {features_df.shape[1]} métricas por usuario")
        
        detection_results = {}
        
        # 2. Reglas heurísticas académicas
        logger.info("Aplicando reglas heurísticas académicas...")
        academic_scores = self.bot_detector.apply_academic_heuristic_rules()
        detection_results['academic_heuristics'] = academic_scores
        
        # 3. Isolation Forest
        logger.info("Aplicando Isolation Forest...")
        isolation_scores = self.bot_detector.detect_anomalies_isolation_forest(contamination=0.08)
        detection_results['isolation_forest'] = isolation_scores
        
        # 4. DBSCAN Clustering
        logger.info("Aplicando DBSCAN clustering...")
        dbscan_clusters = self.bot_detector.detect_bots_clustering(eps=0.5, min_samples=5)
        detection_results['dbscan_clusters'] = dbscan_clusters
        
        # 5. Modelo supervisado (si hay ground truth)
        if not ground_truth_df.empty:
            logger.info("Entrenando modelo supervisado...")
            try:
                training_results = self.bot_detector.train_supervised_model(ground_truth_df, test_size=0.3)
                logger.info(f"Accuracy del modelo supervisado: {training_results['test_accuracy']:.3f}")
                logger.info("Top 5 características más importantes:")
                logger.info(training_results['feature_importance'].head(5).to_string())
                detection_results['random_forest'] = self.bot_detector.bot_scores.get('random_forest', {})
            except Exception as e:
                logger.error(f"Error en entrenamiento supervisado: {e}")
        
        # 6. Combinar todos los métodos con pesos optimizados
        logger.info("Combinando métodos con pesos optimizados...")
        
        # Pesos balanceados que combinan enfoque académico y práctico
        unified_weights = {
            'academic_heuristics': 0.4,  # Peso importante para reglas académicas
            'isolation_forest': 0.3,     # Método robusto para anomalías
            'random_forest': 0.3 if 'random_forest' in detection_results else 0.0  # Supervisado si disponible
        }
        
        # Ajustar pesos si no hay modelo supervisado
        if 'random_forest' not in detection_results:
            unified_weights['academic_heuristics'] = 0.6
            unified_weights['isolation_forest'] = 0.4
        
        combined_scores = self.bot_detector.combine_detection_methods(unified_weights)
        detection_results['combined'] = combined_scores
        
        return detection_results, features_df
    
    def evaluate_and_analyze_results(self, detection_results: Dict, ground_truth_df: pd.DataFrame):
        """
        Evalúa y analiza los resultados de detección
        """
        logger.info("Evaluando y analizando resultados...")
        
        # Obtener reporte académico completo
        academic_report = self.bot_detector.get_academic_analysis_report(ground_truth_df)
        
        # Top candidatos detectados
        top_bots = self.bot_detector.get_top_bot_candidates(method='combined', top_n=20)
        
        logger.info(f"Resultados de detección unificada:")
        if 'deteccion_bots' in academic_report:
            logger.info(f"  - Candidatos con umbral alto (0.7): {academic_report['deteccion_bots']['candidatos_bots_umbral_07']}")
            logger.info(f"  - Candidatos con umbral medio (0.5): {academic_report['deteccion_bots']['candidatos_bots_umbral_05']}")
            logger.info(f"  - Score promedio: {academic_report['deteccion_bots']['score_promedio']:.4f}")
        
        if 'evaluacion' in academic_report:
            eval_results = academic_report['evaluacion']
            logger.info("Evaluación con ground truth:")
            logger.info(f"  - AUC-ROC: {eval_results['auc_roc']:.4f}")
            logger.info(f"  - Average Precision: {eval_results['average_precision']:.4f}")
            logger.info(f"  - Mejor F1-Score: {eval_results['mejor_f1_score']:.4f}")
        
        logger.info(f"\nTop 10 agentes automatizados detectados:")
        for i, (_, row) in enumerate(top_bots.head(10).iterrows()):
            logger.info(f"  {i+1:2d}. {row['user_id']} (score: {row['bot_score']:.3f})")
        
        # Análisis por comunidades
        try:
            community_analysis_bots = self.bot_detector.analyze_bot_communities(bot_threshold=0.7)
            logger.info("Análisis de bots por comunidad (Top 5):")
            for comm_id, data in list(community_analysis_bots.items())[:5]:
                logger.info(f"  Comunidad {comm_id}: {data['bot_users']}/{data['total_users']} bots "
                           f"({data['bot_ratio']:.2%})")
        except Exception as e:
            logger.error(f"Error en análisis de comunidades: {e}")
        
        return academic_report, top_bots
    
    def generate_comprehensive_visualizations(self, centralities: Dict, detection_results: Dict, 
                                            communities: Dict, combined_scores: Dict):
        """
        Genera todas las visualizaciones del análisis
        """
        logger.info("Generando visualizaciones comprehensivas...")
        
        output_dir = '../results'
        os.makedirs(output_dir, exist_ok=True)
        
        self.visualizer = GraphVisualizer(self.graph_builder.graph)
        
        # 1. Análisis de centralidades
        self.visualizer.plot_centrality_comparison(
            centralities,
            save_path=f'{output_dir}/centralities_analysis.png',
            show_plot=False
        )
        
        # 2. Resultados de detección unificada
        self.visualizer.plot_bot_detection_results(
            combined_scores,
            threshold=0.5,
            save_path=f'{output_dir}/unified_bot_detection.png',
            show_plot=False
        )
        
        # 3. Análisis de comunidades
        self.visualizer.plot_community_analysis(
            communities,
            bot_scores=combined_scores,
            save_path=f'{output_dir}/community_analysis.png',
            show_plot=False
        )
        
        # 4. Vista general de la red
        self.visualizer.plot_network_overview(
            layout='spring',
            node_color_attr='community',
            max_nodes=300,
            save_path=f'{output_dir}/network_overview.png',
            show_plot=False
        )
        
        # 5. Visualización interactiva
        try:
            interactive_fig = self.visualizer.create_interactive_network(
                bot_scores=combined_scores,
                communities=communities,
                max_nodes=400
            )
            interactive_fig.update_layout(
                title="Red Social: Detección Unificada de Agentes Automatizados"
            )
            interactive_fig.write_html(f'{output_dir}/interactive_network.html')
            logger.info("Visualización interactiva guardada")
        except Exception as e:
            logger.error(f"Error en visualización interactiva: {e}")
    
    def export_comprehensive_results(self, detection_results: Dict, features_df: pd.DataFrame, 
                                   graph_stats: Dict, community_analysis: Dict, 
                                   academic_report: Dict):
        """
        Exporta todos los resultados del análisis
        """
        logger.info("Exportando resultados comprehensivos...")
        
        output_dir = '../results'
        
        # 1. Resultados principales de detección
        self.bot_detector.export_results(f'{output_dir}/unified_bot_detection.csv', method='combined')
        
        # 2. Características extraídas
        features_df.to_csv(f'{output_dir}/user_features.csv', index=False)
        
        # 3. Estadísticas del grafo
        with open(f'{output_dir}/graph_statistics.txt', 'w', encoding='utf-8') as f:
            f.write("ANÁLISIS UNIFICADO DE AGENTES AUTOMATIZADOS\n")
            f.write("Sistema BotFinderTool - Universidad del Rosario\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("METODOLOGÍA UNIFICADA:\n")
            f.write("- Grafo dirigido ponderado G = (V, E)\n")
            f.write("- Reglas heurísticas académicas integradas\n")
            f.write("- Machine Learning: Isolation Forest + Random Forest\n")
            f.write("- Detección de comunidades: Algoritmo de Louvain\n")
            f.write("- Métricas de centralidad: PageRank, Grado, Intermediación, Agrupamiento\n\n")
            
            f.write("ESTADÍSTICAS DEL GRAFO:\n")
            for key, value in graph_stats.items():
                f.write(f"- {key}: {value}\n")
            
            f.write(f"\nCOMUNIDADES DETECTADAS:\n")
            for key, value in community_analysis.items():
                if key != 'community_sizes':
                    f.write(f"- {key}: {value}\n")
        
        # 4. Reporte académico completo
        with open(f'{output_dir}/unified_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(academic_report, f, indent=2, ensure_ascii=False, default=str)
        
        # 5. Resumen de todos los métodos
        with open(f'{output_dir}/detection_methods_summary.json', 'w', encoding='utf-8') as f:
            summary = {
                'methods_applied': list(detection_results.keys()),
                'total_users_analyzed': len(features_df),
                'features_extracted': len(features_df.columns),
                'timestamp': datetime.now().isoformat()
            }
            json.dump(summary, f, indent=2, ensure_ascii=False)
    
    def run_complete_unified_analysis(self, use_existing_data: bool = False):
        """
        Ejecuta el análisis completo unificado
        """
        logger.info("=" * 70)
        logger.info("BOTFINDERTOOL - ANÁLISIS UNIFICADO")
        logger.info("Detección de Agentes Automatizados en Redes Sociales")
        logger.info("Sistema Integrado: Académico + Machine Learning + Teoría de Grafos")
        logger.info("=" * 70)
        
        # 1. Cargar o generar datos
        logger.info("\n1. PREPARACIÓN DE DATOS COMPREHENSIVOS")
        interactions_df, user_features_df, ground_truth_df = self.load_or_generate_data(use_existing_data)
        
        # 2. Construir grafo
        logger.info("\n2. CONSTRUCCIÓN DEL GRAFO DE RED SOCIAL")
        graph, graph_stats = self.build_network_graph(interactions_df, user_features_df)
        
        # 3. Métricas estructurales
        logger.info("\n3. CÁLCULO DE MÉTRICAS ESTRUCTURALES")
        centralities = self.calculate_graph_metrics()
        
        # 4. Detección de comunidades
        logger.info("\n4. DETECCIÓN DE COMUNIDADES")
        communities, community_analysis = self.detect_communities()
        
        # 5. Detección unificada de bots
        logger.info("\n5. DETECCIÓN UNIFICADA DE AGENTES AUTOMATIZADOS")
        detection_results, features_df = self.apply_unified_bot_detection(ground_truth_df)
        
        # 6. Evaluación y análisis
        logger.info("\n6. EVALUACIÓN Y ANÁLISIS DE RESULTADOS")
        academic_report, top_bots = self.evaluate_and_analyze_results(detection_results, ground_truth_df)
        
        # 7. Visualizaciones
        logger.info("\n7. GENERACIÓN DE VISUALIZACIONES")
        self.generate_comprehensive_visualizations(
            centralities, detection_results, communities, 
            detection_results['combined']
        )
        
        # 8. Exportar resultados
        logger.info("\n8. EXPORTACIÓN DE RESULTADOS")
        self.export_comprehensive_results(
            detection_results, features_df, graph_stats, 
            community_analysis, academic_report
        )
        
        # 9. Resumen final
        logger.info("\n" + "=" * 70)
        logger.info("ANÁLISIS UNIFICADO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 70)
        
        total_bots_detected = len([s for s in detection_results['combined'].values() if s >= 0.5])
        
        logger.info(f"✓ Grafo dirigido ponderado construido: {graph_stats['nodes']} nodos, {graph_stats['edges']} aristas")
        logger.info(f"✓ Comunidades detectadas: {community_analysis['num_communities']} (modularidad: {community_analysis['modularity']:.3f})")
        logger.info(f"✓ Agentes automatizados identificados: {total_bots_detected}")
        logger.info(f"✓ Métodos aplicados: {len(detection_results)} algoritmos unificados")
        if 'evaluacion' in academic_report:
            logger.info(f"✓ Precisión del sistema: {academic_report['evaluacion']['mejor_f1_score']:.3f}")
        
        logger.info(f"\nArchivos generados en ../results/:")
        logger.info("- unified_bot_detection.csv: Rankings unificados")
        logger.info("- user_features.csv: Características completas")
        logger.info("- centralities_analysis.png: Análisis de centralidades")
        logger.info("- unified_bot_detection.png: Resultados unificados")
        logger.info("- community_analysis.png: Análisis de comunidades")
        logger.info("- network_overview.png: Vista general de la red")
        logger.info("- interactive_network.html: Visualización interactiva")
        logger.info("- unified_analysis_report.json: Reporte completo")
        
        return self.results

def main():
    """
    Función principal del sistema unificado
    """
    parser = argparse.ArgumentParser(description='BotFinderTool - Sistema Unificado de Detección de Bots')
    parser.add_argument('--use-existing-data', action='store_true',
                       help='Usar datos existentes en lugar de generar nuevos')
    parser.add_argument('--generate-data-only', action='store_true',
                       help='Solo generar datos de ejemplo y salir')
    
    args = parser.parse_args()
    
    # Crear analizador unificado
    analyzer = UnifiedBotAnalyzer()
    
    if args.generate_data_only:
        logger.info("Generando solo datos de ejemplo...")
        analyzer.generate_comprehensive_dataset()
        logger.info("Datos comprehensivos generados exitosamente.")
        return
    
    # Ejecutar análisis completo unificado
    analyzer.run_complete_unified_analysis(use_existing_data=args.use_existing_data)

if __name__ == "__main__":
    main() 