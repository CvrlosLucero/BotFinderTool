# BotFinderTool - Sistema Unificado de Detección de Bots en Redes Sociales

**Análisis Integrado de Agentes Automatizados usando Teoría de Grafos**

> **Proyecto Académico**: Carlos Lucero, Samantha Montero, Camilo Becerra  
> **Universidad del Rosario** - Teoría de Grafos - Matemáticas Aplicadas y Ciencias de la Computación

## 🎯 Descripción del Sistema

BotFinderTool es un **sistema unificado** que combina automáticamente múltiples enfoques para la detección de agentes automatizados (bots) en redes sociales:

- 🔬 **Reglas Heurísticas Académicas**: Implementa 8 reglas basadas en investigación científica
- 🤖 **Machine Learning Avanzado**: Isolation Forest, DBSCAN, Random Forest integrados
- 🌐 **Análisis de Grafos Completo**: PageRank, centralidades, coeficiente de agrupamiento  
- 🔍 **Detección de Comunidades**: Algoritmo de Louvain para identificar "granjas de bots"
- 📊 **Sistema de Combinación Inteligente**: Combina automáticamente todos los métodos con pesos optimizados
- 📈 **Visualizaciones Avanzadas**: Gráficos estáticos e interactivos completos

## 🚀 **Enfoque Único: Sistema Integrado**

**A diferencia de otros sistemas**, BotFinderTool **NO requiere elegir** entre métodos académicos o prácticos. El sistema:

✅ **Aplica TODOS los métodos automáticamente**  
✅ **Combina resultados de forma inteligente**  
✅ **Genera un score unificado final**  
✅ **Proporciona análisis comprehensivo**

## 📋 Metodología Unificada

### Algoritmos Implementados Simultáneamente

1. **Reglas Heurísticas Académicas** (40% peso)
   - Alta actividad de salida vs. entrada
   - PageRank bajo con alto grado de salida
   - Bajo coeficiente de agrupamiento
   - Patrones de comportamiento automatizado

2. **Isolation Forest** (30% peso)
   - Detección de anomalías en características de grafos
   - Identificación de patrones atípicos

3. **Random Forest Supervisado** (30% peso)
   - Entrenamiento con ground truth disponible
   - Clasificación basada en características extraídas

4. **DBSCAN Clustering** (Análisis complementario)
   - Identificación de clusters de comportamiento similar
   - Detección de outliers

### Proceso de Análisis Automático

```
Datos → Grafo → Centralidades → Comunidades → TODOS los Métodos → Score Unificado → Resultados
```

## 🏗️ Arquitectura del Sistema

```
BotFinderTool/
├── src/                          # Código fuente modular
│   ├── data_loader.py           # Carga de datos CSV
│   ├── graph_builder.py         # Construcción de grafos dirigidos
│   ├── community_detection.py   # Algoritmo de Louvain
│   ├── bot_detector.py          # Métodos de detección integrados
│   └── visualizer.py            # Visualizaciones comprehensivas
├── examples/
│   └── bot_analysis.py          # Sistema unificado principal
├── data/                        # Datasets de entrada
├── results/                     # Resultados y visualizaciones
└── requirements.txt            # Dependencias
```

## 🚀 Instalación y Uso Simplificado

### Instalación Rápida

```bash
# Clonar repositorio
git clone https://github.com/CvrlosLucero/BotFinderTool.git
cd BotFinderTool

# Instalar dependencias
pip install -r requirements.txt
```

### Uso Ultra-Simplificado

```bash
cd examples

# EJECUTAR ANÁLISIS COMPLETO (un solo comando)
python bot_analysis.py

# Con datos existentes
python bot_analysis.py --use-existing-data

# Solo generar datos de ejemplo
python bot_analysis.py --generate-data-only
```

**¡ESO ES TODO!** El sistema aplica automáticamente todos los métodos y combina los resultados.

## 📊 Características Automáticas del Dataset

El sistema genera automáticamente datasets que combinan **patrones académicos** y **patrones generales**:

### Patrones Implementados Automáticamente

1. **Bots en clusters artificiales** (patrón académico)
2. **Spam masivo bot→humano** (patrón general)  
3. **Interacciones humanas auténticas** (línea base)
4. **Followers legítimos a bots** (ruido realista)
5. **Bots coordinados en grupos** (patrón avanzado)

### Características Híbridas

- **Bots**: Pocos seguidores, muchos siguiendo, cuentas nuevas, sin verificar
- **Humanos**: Patrones diversos y naturales de actividad
- **Ground Truth**: Tipos específicos (spam_bot, automated_agent, fake_follower)

## 🔬 Reglas Heurísticas Integradas

El sistema aplica automáticamente **8 reglas académicas**:

| Regla | Condición | Score |
|-------|-----------|-------|
| **Alta salida** | `out_degree > in_degree * 2` | +0.3 |
| **PageRank bajo** | `pagerank < 0.001` | +0.2 |
| **Desconexión** | `out_degree > 30 AND clustering < 0.1` | +0.25 |
| **Bajo agrupamiento** | `clustering < 0.05` | +0.15 |
| **Ratio alto** | `following/followers > 3.0` | +0.2 |
| **Baja intermediación** | `betweenness < 0.001` | +0.1 |
| **Combinación** | Alto grado + baja centralidad | +0.2 |
| **Comportamiento** | Baja diversidad + reciprocidad | +0.3 |

## 📈 Resultados Automáticos Generados

### Archivos de Salida

```
results/
├── unified_bot_detection.csv          # Rankings unificados finales
├── user_features.csv                  # Todas las características
├── centralities_analysis.png          # Análisis de centralidades
├── unified_bot_detection.png          # Resultados visualizados
├── community_analysis.png             # Análisis de comunidades  
├── network_overview.png               # Vista general de la red
├── interactive_network.html           # Visualización interactiva
├── graph_statistics.txt               # Estadísticas completas
├── unified_analysis_report.json       # Reporte académico
└── detection_methods_summary.json     # Resumen de métodos
```

### Métricas de Evaluación Automáticas

- **AUC-ROC**: Área bajo la curva ROC
- **Average Precision**: Precisión promedio
- **F1-Score**: Medida armónica de precisión y recall
- **Modularidad**: Calidad de detección de comunidades
- **Análisis por umbral**: Candidatos con diferentes niveles de confianza

## 🛠️ Formato de Datos de Entrada

### Archivos CSV Requeridos

**`data/interactions.csv`** (obligatorio):
```csv
source_user,target_user,interaction_type,weight,timestamp
user_001,user_002,follow,1,2024-01-01 10:00:00
bot_001,user_003,mention,2,2024-01-01 11:30:00
```

**`data/user_features.csv`** (obligatorio):
```csv
user_id,followers_count,following_count,tweets_count,account_age_days,verified,profile_pic,bio_length
user_001,250,180,1500,720,True,True,85
bot_001,15,800,50,30,False,False,10
```

**`data/ground_truth.csv`** (opcional para evaluación):
```csv
user_id,is_bot,bot_type
user_001,False,human
bot_001,True,spam_bot
```

## 🔧 Dependencias Principales

```
networkx>=3.0          # Análisis de grafos
pandas>=1.5.0          # Manipulación de datos  
scikit-learn>=1.3.0    # Machine learning
matplotlib>=3.7.0      # Visualizaciones
seaborn>=0.12.0        # Gráficos estadísticos
plotly>=5.15.0         # Visualizaciones interactivas
python-louvain>=0.16   # Detección de comunidades
cdlib>=0.3.0           # Algoritmos de comunidades
```

## 📚 Fundamentos Teóricos Integrados

### Representación Matemática

**Grafo Dirigido Ponderado**: G = (V, E, W)
- V: Conjunto de vértices (usuarios)
- E: Conjunto de aristas dirigidas (interacciones)  
- W: Función de peso (intensidad de interacción)

### Score Unificado Final

```
Score_final = 0.4 × Score_académico + 0.3 × Score_isolation + 0.3 × Score_supervisado
```

### Algoritmos Simultáneos

- **PageRank**: Importancia basada en conexiones
- **Louvain**: Optimización de modularidad para comunidades
- **Isolation Forest**: Detección de anomalías multivariadas
- **Random Forest**: Clasificación supervisada con importancia de características

## 🎓 Casos de Uso del Sistema Unificado

### Investigación Académica
- ✅ Aplicación directa de reglas heurísticas validadas
- ✅ Evaluación automática con métricas académicas
- ✅ Análisis de comunidades y estructuras de red
- ✅ Exportación de resultados para publicación

### Aplicaciones Comerciales  
- ✅ Detección robusta combinando múltiples métodos
- ✅ Scores de confianza calibrados
- ✅ Análisis en tiempo real con datos existentes
- ✅ Visualizaciones para presentaciones ejecutivas

### Validación Científica
- ✅ Comparación automática de múltiples enfoques
- ✅ Ground truth para validación de hipótesis
- ✅ Métricas de performance estándar
- ✅ Reproducibilidad completa

## 📊 Estadísticas Típicas del Sistema

### Performance Esperada
- **Usuarios analizados**: 1000
- **Bots detectados**: 80 (8% del dataset)
- **Comunidades**: 20-25 (modularidad ~0.4)
- **Características extraídas**: 17+ métricas por usuario
- **Tiempo de ejecución**: 2-5 minutos
- **AUC-ROC típica**: 0.85-0.95

### Hipótesis Validadas Automáticamente

✅ **Clustering artificial de bots**  
✅ **Alto grado de salida, bajo de entrada**  
✅ **Bajo coeficiente de agrupamiento**  
✅ **PageRank reducido**  
✅ **Baja centralidad de intermediación**  
✅ **Patrones temporales concentrados**

## 🔬 Ventajas del Sistema Unificado

### Vs. Sistemas Separados

| Característica | Sistema Tradicional | BotFinderTool Unificado |
|----------------|--------------------|-----------------------|
| **Métodos aplicados** | 1-2 métodos | 4+ métodos automáticos |
| **Validación académica** | Manual | Integrada automáticamente |
| **Combinación de resultados** | Manual | Pesos optimizados |
| **Evaluación** | Básica | Métricas completas |
| **Visualizaciones** | Limitadas | Comprehensivas |
| **Tiempo de configuración** | Horas | Minutos |

### Robustez

- **Reducción de falsos positivos**: Combinación de métodos
- **Mayor cobertura**: Detección complementaria
- **Validación cruzada**: Métodos se validan mutuamente
- **Adaptabilidad**: Funciona con datasets diversos

### Resultados Automáticos

El sistema genera automáticamente:

✅ **Dataset comprehensivo** (1000 usuarios, 80 bots conocidos)  
✅ **Grafo dirigido ponderado** G = (V, E) con análisis completo  
✅ **8 reglas heurísticas académicas** aplicadas simultáneamente  
✅ **3 algoritmos de ML** (Isolation Forest, DBSCAN, Random Forest)  
✅ **Score unificado final** combinando todos los métodos  
✅ **19+ visualizaciones** (estáticas e interactivas)  
✅ **Reporte académico completo** en JSON

### Salida Típica del Sistema

```
✓ Grafo dirigido ponderado construido: 999 nodos, 4798 aristas
✓ Comunidades detectadas: 19 (modularidad: 0.397)
✓ Agentes automatizados identificados: 401 candidatos
✓ Métodos aplicados: 4 algoritmos unificados

Top 10 agentes automatizados detectados:
  1. bot_059 (score: 0.819)
  2. bot_044 (score: 0.819)  
  3. bot_004 (score: 0.818)
  4. user_0436 (score: 0.807)
  5. user_0598 (score: 0.792)
```
## 📄 Licencia

MIT License - Universidad del Rosario

## 👥 Autores

- **Carlos Lucero** - Desarrollo principal y metodología académica
- **Samantha Montero** - Análisis de grafos y visualizaciones  
- **Camilo Becerra** - Algoritmos de detección y evaluación

**Universidad del Rosario** - Teoría de Grafos  
Escuela de Ingeniería, Ciencia y Tecnología

---

**© 2024 - BotFinderTool - Sistema Unificado de Detección de Bots**
