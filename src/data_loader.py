"""
Módulo para cargar y procesar datos de redes sociales desde archivos CSV
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

class CSVDataLoader:
    """
    Clase para cargar y procesar datos de interacciones de redes sociales desde archivos CSV
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Inicializa el cargador de datos CSV
        
        Args:
            encoding (str): Codificación de archivos CSV
        """
        self.encoding = encoding
        self.logger = logging.getLogger(__name__)
        
    def load_interactions(self, filepath: str) -> pd.DataFrame:
        """
        Carga datos de interacciones desde archivo CSV
        
        Args:
            filepath (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con interacciones cargadas
            
        Expected CSV format:
            source_user, target_user, interaction_type, weight, timestamp
        """
        try:
            df = pd.read_csv(filepath, encoding=self.encoding)
            
            # Validar columnas requeridas
            required_cols = ['source_user', 'target_user', 'interaction_type']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Columnas faltantes: {missing_cols}")
                
            # Procesar tipos de datos
            df['source_user'] = df['source_user'].astype(str)
            df['target_user'] = df['target_user'].astype(str)
            
            # Agregar peso por defecto si no existe
            if 'weight' not in df.columns:
                df['weight'] = 1.0
            else:
                df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(1.0)
                
            # Procesar timestamp si existe
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                
            self.logger.info(f"Cargadas {len(df)} interacciones desde {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando {filepath}: {str(e)}")
            raise
    
    def load_user_features(self, filepath: str) -> pd.DataFrame:
        """
        Carga características de usuarios desde archivo CSV
        
        Args:
            filepath (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con características de usuarios
            
        Expected CSV format:
            user_id, followers_count, following_count, tweets_count, 
            account_age_days, verified, profile_pic, bio_length
        """
        try:
            df = pd.read_csv(filepath, encoding=self.encoding)
            
            # Validar columna principal
            if 'user_id' not in df.columns:
                raise ValueError("Columna 'user_id' requerida")
                
            df['user_id'] = df['user_id'].astype(str)
            df = df.set_index('user_id')
            
            # Procesar características numéricas
            numeric_cols = ['followers_count', 'following_count', 'tweets_count', 
                          'account_age_days', 'bio_length']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Procesar características booleanas
            bool_cols = ['verified', 'profile_pic']
            for col in bool_cols:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
                    
            self.logger.info(f"Cargadas características de {len(df)} usuarios desde {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando características de usuarios {filepath}: {str(e)}")
            raise
    
    def load_ground_truth(self, filepath: str) -> pd.DataFrame:
        """
        Carga datos de verdad fundamental (labels de bots conocidos)
        
        Args:
            filepath (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con labels de bots
            
        Expected CSV format:
            user_id, is_bot, bot_type
        """
        try:
            df = pd.read_csv(filepath, encoding=self.encoding)
            
            if 'user_id' not in df.columns or 'is_bot' not in df.columns:
                raise ValueError("Columnas 'user_id' e 'is_bot' requeridas")
                
            df['user_id'] = df['user_id'].astype(str)
            df['is_bot'] = df['is_bot'].astype(bool)
            df = df.set_index('user_id')
            
            self.logger.info(f"Cargados labels de {len(df)} usuarios desde {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error cargando ground truth {filepath}: {str(e)}")
            raise
    
    def preprocess_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesa las interacciones para análisis de grafos
        
        Args:
            df (pd.DataFrame): DataFrame de interacciones
            
        Returns:
            pd.DataFrame: DataFrame preprocesado
        """
        # Eliminar auto-loops
        df = df[df['source_user'] != df['target_user']].copy()
        
        # Agregar interacciones duplicadas sumando pesos
        df_grouped = df.groupby(['source_user', 'target_user', 'interaction_type']).agg({
            'weight': 'sum',
            'timestamp': 'min' if 'timestamp' in df.columns else lambda x: None
        }).reset_index()
        
        self.logger.info(f"Preprocesamiento completado. {len(df_grouped)} interacciones únicas")
        return df_grouped
    
    def filter_by_interaction_type(self, df: pd.DataFrame, 
                                 interaction_types: List[str]) -> pd.DataFrame:
        """
        Filtra interacciones por tipo
        
        Args:
            df (pd.DataFrame): DataFrame de interacciones
            interaction_types (List[str]): Tipos de interacción a mantener
            
        Returns:
            pd.DataFrame: DataFrame filtrado
        """
        filtered_df = df[df['interaction_type'].isin(interaction_types)].copy()
        self.logger.info(f"Filtradas {len(filtered_df)} interacciones de tipos: {interaction_types}")
        return filtered_df
    
    def get_interaction_summary(self, df: pd.DataFrame) -> Dict:
        """
        Genera resumen estadístico de las interacciones
        
        Args:
            df (pd.DataFrame): DataFrame de interacciones
            
        Returns:
            Dict: Resumen estadístico
        """
        summary = {
            'total_interactions': len(df),
            'unique_users': len(set(df['source_user'].unique()) | set(df['target_user'].unique())),
            'interaction_types': df['interaction_type'].value_counts().to_dict(),
            'weight_stats': df['weight'].describe().to_dict() if 'weight' in df.columns else None,
            'date_range': {
                'min': df['timestamp'].min(),
                'max': df['timestamp'].max()
            } if 'timestamp' in df.columns and df['timestamp'].notna().any() else None
        }
        
        return summary 