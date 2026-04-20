# Progreso del Proyecto

## 📊 Estado General: FASE 1 COMPLETADA ✅

**Fecha de Inicio:** Abril 2026
**Último Actualizado:** Abril 20, 2026
**Timeline:** 2-3 meses para completar todas las fases

---

## ✅ FASE 1: Configuración y Data Pipeline (COMPLETADA)

### Tareas Completadas

- [x] **Estructura del proyecto**
  - Carpetas creadas: `data/`, `src/`, `models/`, `notebooks/`, `results/`, `logs/`
  - Archivos: `requirements.txt`, `README.md`, `.gitignore`, `.env.example`

- [x] **Configuración centralizada**
  - Archivo `config.py` con variables de entorno
  - Archivo `.env.example` como template
  - Fácil ajuste de parámetros sin modificar código

- [x] **Data Pipeline (`src/data_pipeline.py`)**
  - Descarga histórico de QQQ desde Yahoo Finance
  - Cálculo de indicadores técnicos (RSI, MACD, Bollinger Bands, ATR, SMA)
  - Normalización de features (StandardScaler)
  - Creación de secuencias LSTM (ventanas de 30 días)
  - Guardado de datos procesados (.npy, .csv)

- [x] **Utilidades (`src/utils.py`)**
  - Split temporal train/val/test respetando orden
  - Funciones de cálculo de métricas (MAE, RMSE, R², Sharpe Ratio)
  - Visualización de predicciones y distribución de errores
  - Funciones para backtesting

- [x] **Arquitecturas de Modelos (`src/models.py`)**
  - `LSTMModel`: BiLSTM bidireccional para series temporales
  - `SentimentEncoder`: Red para procesar features estáticos
  - `HybridModel`: Fusión de ambos componentes
  - Funciones auxiliares (count_parameters, print_model_summary)

- [x] **Bucle de Entrenamiento (`src/train.py`)**
  - Clase `EarlyStoppingCallback` para evitar overfitting
  - Clase `Trainer` con métodos train/validate
  - Funciones de creación de DataLoaders (respetando orden temporal)
  - Función de predicción en batch

- [x] **Script Principal (`run_pipeline.py`)**
  - Ejecutable que corre FASE 1 completa
  - Logging integrado
  - Resumen de resultados

- [x] **Documentación**
  - README.md completo con guía de instalación
  - Docstrings en todo el código
  - Ejemplos de uso

### Deliverable
✅ **Dataset listo para entrenamiento** con:
- Datos normalizados
- Secuencias LSTM (n_samples, 30 días, 10 features)
- Retornos objetivo (n_samples,)
- Scaler para nuevos datos
- Guardado en `data/processed/`

### Cómo Ejecutar
```bash
python run_pipeline.py
```

---

## 📝 FASE 2: Modelos Baseline y EDA (PENDIENTE)

### Tareas Planeadas

- [ ] **Análisis Exploratorio (01_eda.ipynb)**
  - Distribución de retornos (normalidad, multimodalidad)
  - Gráficos de autocorrelación (ACF/PACF)
  - Test de estacionariedad (ADF)
  - Correlación entre features y retorno

- [ ] **Modelos Baseline (02_baseline.ipynb)**
  - Modelo Naive (predicción = último valor)
  - Regresión Lineal Multivariable
  - Árbol de Decisión (XGBoost/LightGBM)
  - Métricas comparativas

- [ ] **Documentación Educativa**
  - Por qué LSTM es necesario
  - Conceptos de RNN, LSTM, BiLSTM
  - Ventajas sobre modelos clásicos

### Criterio de Éxito
- Baseline RMSE < 1.5% del precio medio
- Motivación clara para usar LSTM

---

## 🔄 FASE 3: Modelo LSTM/BiLSTM (PENDIENTE)

### Tareas Planeadas

- [ ] **Notebook 03_lstm_training.ipynb**
  - Cargar datos desde `data/processed/`
  - Crear DataLoaders
  - Entrenar LSTMModel con early stopping
  - Visualizar predicciones vs reales
  - Análisis de errores por contexto (mercado alcista/bajista)
  - Guardar mejor modelo

- [ ] **Implementación de Variantes**
  - Probar diferentes hidden_size
  - Probar diferentes num_layers
  - Probar diferentes dropout rates

### Criterio de Éxito
- RMSE con mejora > 20% sobre baseline
- Modelo entrenado sin overfitting severo

---

## 📰 FASE 4: Análisis de Sentimiento (PENDIENTE)

### Tareas Planeadas

- [ ] **Descargar Noticias**
  - Usar Finnhub API o NewsAPI
  - Alineación temporal con datos de precios
  - Almacenamiento en `data/processed/`

- [ ] **Extracción de Sentimiento**
  - Usar FinBERT pre-entrenado
  - Procesar títulos y párrafos de noticias
  - Calcular score de sentimiento por día

- [ ] **Features de Sentimiento**
  - Score promedio por día
  - Volumen de noticias
  - Ratio positive/negative/neutral

### Criterio de Éxito
- Features integradas al dataset principal
- Correlación sentimiento-retorno > 0.2

---

## 🔗 FASE 5: Modelo Híbrido (PENDIENTE)

### Tareas Planeadas

- [ ] **Notebook 04_hybrid_model.ipynb**
  - Arquitectura multi-rama (LSTM + Dense)
  - Entrenamiento end-to-end
  - Comparativa LSTM solo vs Híbrido

- [ ] **Análisis de Contribución**
  - Importancia relativa de cada rama
  - Ablation study

### Criterio de Éxito
- Mejora > 10% sobre LSTM solo
- Hybrid RMSE < 1.0%

---

## 💰 FASE 6: Backtesting (PENDIENTE)

### Tareas Planeadas

- [ ] **Estrategia de Trading**
  - Compra si predicción > umbral
  - Venta si predicción < -umbral
  - Hold si está en rango

- [ ] **Métricas**
  - Sharpe Ratio anualizado
  - Sortino Ratio
  - Maximum Drawdown
  - Win Rate
  - Cumulative Return

- [ ] **Validación**
  - Out-of-sample testing
  - Análisis de concept drift
  - Robustez a cambios de mercado

### Criterio de Éxito
- Sharpe Ratio > 0.5 en test set
- Desempeño out-of-sample comparable a in-sample

---

## 📚 FASE 7: Documentación Final (PENDIENTE)

### Tareas Planeadas

- [ ] **Código Limpio**
  - Docstrings completos
  - Type hints en funciones
  - Refactorización si es necesario

- [ ] **Análisis de Interpretabilidad**
  - Feature importance (permutation importance)
  - SHAP values si es viable
  - Attention visualization (si usa attention layers)

- [ ] **Visualizaciones Finales**
  - Arquitectura del modelo (diagrama)
  - Curvas de entrenamiento (loss, MAE)
  - Predicciones en período completo
  - Análisis de errores (histograma, Q-Q plot)

- [ ] **Redacción de Tesis**
  - Metodología implementada
  - Resultados experimentales
  - Conclusiones y limitaciones
  - Trabajo futuro

---

## 📈 Métricas Objetivo

| Métrica | Baseline | LSTM | Hybrid | Target |
|---------|----------|------|--------|--------|
| RMSE (%) | 1.5 | 1.0 | 0.8 | < 0.8 |
| MAE (%) | 1.2 | 0.8 | 0.6 | < 0.6 |
| Sharpe | - | 0.3 | 0.5 | > 0.5 |
| Max DD (%) | -25 | -20 | -15 | > -15 |

---

## 🔧 Problemas Identificados y Soluciones

### 1. Look-ahead Bias
**Problema:** Usar datos futuros durante entrenamiento
**Solución:** ✅ Split temporal en train/val/test (sin shuffle)

### 2. Data Leakage en Normalización
**Problema:** Normalizar antes de dividir train/val/test
**Solución:** ✅ Fit scaler solo en train, aplicar a val/test

### 3. Tamaño del Dataset
**Problema:** Solo ~5 años de datos (<=1300 muestras LSTM)
**Situación:** Aceptable pero no ideal (sería mejor 10+ años)
**Mitigación:** Early stopping, regularización (dropout, L2)

### 4. Mercados Volátiles
**Problema:** QQQ es muy volátil, difícil de predecir
**Solución:** ✅ Usar Huber Loss en lugar de MSE
**Próximo:** Investigar condicionamiento por volatilidad

---

## 📞 Contacto y Soporte

Para preguntas o ajustes:
- Email: yabdul1506@gmail.com
- Directora: Sonia Jaramillo Valbuena

---

## 📝 Notas para Próximas Fases

1. **FASE 2:** Comenzar con notebooks educativos. Estos son cruciales para tu aprendizaje en Deep Learning.

2. **Datos de Noticias:** Investigar APIs disponibles:
   - Finnhub (gratis, limitado a 60 API calls/min)
   - NewsAPI (gratis, limitado a 100 requests/day)
   - Alternativa: Usar Twitter/StockTwits para sentiment

3. **GPU vs CPU:** Si LSTM entrena lentamente (>5 min/época), considerar usar GPU (CUDA):
   ```python
   device = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

4. **Hyperparameter Tuning:** Una vez LSTM funcione, probar:
   - hidden_size: 32, 64, 128
   - num_layers: 1, 2, 3
   - dropout: 0.1, 0.2, 0.3
   - learning_rate: 0.0001, 0.001, 0.01

---

**¡Listo para FASE 2! 🚀**
