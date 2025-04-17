---
title: "Fundamentos de Sistemas de Recomendación en Aprendizaje Automático"
technologies: ["datascience", "machine-learning", "sistemas-de-recomendación"]
description: >-
  Aprende qué son los sistemas de recomendación, sus principales tipos y los algoritmos
  que los hacen funcionar. Una guía clara y pedagógica para estudiantes que comienzan
  en Machine Learning y Data Science.
---


# Sistemas de Recomendación y sus Algoritmos


Los sistemas de recomendación son una de las aplicaciones más prácticas del aprendizaje automático y la ciencia de datos. Se utilizan para sugerir productos, servicios o contenido a los usuarios en función de sus intereses, comportamiento o similitudes con otros usuarios. Estos sistemas están presentes en la mayoría de las plataformas modernas, desde sugerencias de películas en Netflix hasta productos en Amazon o publicaciones en redes sociales.


## ¿Qué es un sistema de recomendación?

Un **sistema de recomendación** es una herramienta de software que ayuda a los usuarios a descubrir elementos de interés. Algunos ejemplos de elementos que puede recomendar un sistema son:

- Películas o series (Netflix, Hulu)
- Productos (Amazon, Mercado Libre)
- Canciones o listas de reproducción (Spotify, Apple Music)
- Artículos de noticias (Google News, Pocket)
- Personas o perfiles (LinkedIn, Facebook)

El objetivo principal es **filtrar y priorizar información relevante** para el usuario, mejorando su experiencia y reduciendo la sobrecarga de información.


## Principales tipos de sistemas de recomendación

### 1. Filtrado Basado en Contenido (*Content-Based Filtering*)

Este tipo de sistema recomienda elementos similares a los que el usuario ha consumido o valorado positivamente en el pasado. Se basa en las **características del contenido** (por ejemplo, género, palabras clave, autor).

**Cómo funciona:**

- Cada elemento se representa como un vector de características.
- Se construye un perfil del usuario con base en los elementos que le han gustado.
- Se calculan similitudes (por ejemplo, con la similitud del coseno) entre el perfil del usuario y los nuevos elementos.
- Se recomiendan los elementos más parecidos.

**Ejemplo:** Si un usuario ha visto películas de acción con tramas rápidas y protagonistas fuertes, el sistema le recomendará otras películas con esas características, aunque nadie más las haya visto.

| Ventajas                                        | Desventajas                                         |
|------------------------------------------------|-----------------------------------------------------|
| Personalización individual                     | Riesgo de sobre-especialización                     |
| No requiere datos de otros usuarios            | No descubre nuevos intereses                        |
| Útil para recomendar elementos poco populares  | Limitado a lo que ya ha visto el usuario           |


### 2. Filtrado Colaborativo (*Collaborative Filtering*)

Este enfoque se basa en la interacción de múltiples usuarios. Parte de la idea de que “si a un grupo de usuarios similares les gusta un ítem, es probable que también te guste a ti”.

**Dos enfoques principales:**

- **Basado en usuarios:** Encuentra usuarios similares al usuario objetivo y recomienda los ítems que a ellos les gustaron.
- **Basado en ítems:** Encuentra ítems similares a los que el usuario ha valorado positivamente y los recomienda.

**Algoritmos comunes:**

- *k-Nearest Neighbors (k-NN)*
- Correlación de Pearson
- Factorización de matrices (*Matrix Factorization*), como SVD

**Ejemplo:** Si dos usuarios tienen gustos similares y uno de ellos valora positivamente una película nueva, esa película será recomendada al otro usuario.

| Ventajas                               | Desventajas                                                                 |
|----------------------------------------|------------------------------------------------------------------------------|
| Descubre contenido inesperado          | Problema del arranque en frío: usuarios o ítems nuevos sin datos suficientes |
| Aprovecha tendencias colectivas        | Matriz de datos dispersa: pocos ítems calificados por usuario               |


### 3. Sistemas Híbridos

Combinan varios enfoques para obtener mejores resultados. Por lo general, mezclan filtrado basado en contenido y colaborativo.

**Estrategias comunes de hibridación:**

- **Híbrido ponderado:** Se combinan los resultados de distintos modelos con diferentes pesos.
- **Híbrido conmutado:** Se utiliza un modelo u otro según la situación.
- **Híbrido en cascada:** Un modelo filtra candidatos y otro los ordena.
- **Híbrido de nivel meta:** Un modelo alimenta a otro (por ejemplo, usar perfiles de contenido dentro de un modelo colaborativo).

**Ejemplo:** Un servicio de música puede recomendar canciones basándose en el historial del usuario y en las preferencias de otros usuarios similares.

| Ventajas                                                             | Desventajas                                      |
|----------------------------------------------------------------------|--------------------------------------------------|
| Mayor precisión                                                      | Mayor complejidad en la implementación           |
| Soluciona problemas de arranque en frío y sobre-especialización      | Necesita coordinación entre modelos              |
| Más robusto frente a cambios en los datos                            |                                                  |



## Fundamentos Algorítmicos

Para construir estos sistemas, es necesario conocer algunas herramientas matemáticas y computacionales:

### Medidas de similitud

- **Similitud del coseno:** Mide el ángulo entre dos vectores.
- **Correlación de Pearson:** Mide la relación lineal entre las valoraciones.
- **Índice de Jaccard:** Mide la superposición entre conjuntos binarios.

### Reducción de dimensionalidad

- **SVD (Singular Value Decomposition)**
- **NMF (Non-negative Matrix Factorization)**
- Permite descubrir factores latentes que explican las preferencias.

### Representación de características

- **TF-IDF:** Representación de texto basada en frecuencia e importancia.
- **Codificación one-hot:** Para variables categóricas.
- **Embeddings:** Representaciones densas útiles en modelos avanzados y redes neuronales.


Los sistemas de recomendación son fundamentales en aplicaciones modernas. Existen distintos enfoques, cada uno con ventajas y desventajas. El diseño de un sistema adecuado dependerá del tipo de datos disponibles, el objetivo del sistema y los recursos computacionales.
