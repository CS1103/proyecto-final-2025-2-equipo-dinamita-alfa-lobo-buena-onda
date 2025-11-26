[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/o8XztwuW)
# Proyecto Final 2025-1: AI Neural Network
## **CS2013 Programación III** · Informe Final

### **Descripción**

> Ejemplo: Implementación de una red neuronal multicapa en C++ para clasificación de dígitos manuscritos.

### Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Diseño e implementación](#2-diseño-e-implementación)
5. [Ejecución](#3-ejecución)
6. [Análisis del rendimiento](#4-análisis-del-rendimiento)
7. [Trabajo en equipo](#5-trabajo-en-equipo)
8. [Conclusiones](#6-conclusiones)
9. [Bibliografía](#7-bibliografía)
10. [Licencia](#licencia)
---

### Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `equipo-dinamita-alfa-lobo-buena-onda`
* **Integrantes**:

  * Elias Alonso Usaqui Cabezas – 202420064 (Responsable de investigación teórica)
  * Alumno B – 209900002 (Desarrollo de la arquitectura)
  * Alumno C – 209900003 (Implementación del modelo)
  * Alumno D – 209900004 (Pruebas y benchmarking)
  * Alumno E – 209900005 (Documentación y demo)

> *Nota: Reemplazar nombres y roles reales.*

---

### Requisitos e instalación

1. **Compilador**: GCC 11 o superior
2. **Dependencias**:

   * CMake 3.18+
   * Eigen 3.4
   * \[Otra librería opcional]
3. **Instalación**:

   ```bash
   git clone https://github.com/EJEMPLO/proyecto-final.git
   cd proyecto-final
   mkdir build && cd build
   cmake ..
   make
   ```

> *Ejemplo de repositorio y comandos, ajustar según proyecto.*

---

### 1. Investigación teórica

* **Objetivo**: Explorar fundamentos y arquitecturas de redes neuronales.
* **Contenido de ejemplo**:


#1. Historia y evolución de las NNs.
     
- La historia de las redes neuronales artificiales comienza en 1943, cuando Warren McCulloch y Walter Pitts desarrollaron la primera neurona artificial, conocida como la neurona de McCulloch-Pitts. Su trabajo sentó las bases teóricas para entender cómo las neuronas podrían funcionar y ser modeladas con circuitos eléctricos. Posteriormente, en 1958, Frank Rosenblatt creó el perceptrón, el primer modelo de red neuronal entrenable, lo que marcó el inicio del interés práctico en este campo.
 
 - Durante los años 80, se introdujo el algoritmo de retropropagación (backpropagation), que permitió entrenar redes neuronales con múltiples capas, acelerando significativamente el desarrollo de la inteligencia artificial. En 1989, Yann LeCun desarrolló las redes neuronales convolucionales (CNN), inspiradas en el córtex visual y utilizadas especialmente en reconocimiento de imágenes.
 
 - El verdadero avance hacia el deep learning se dio en 2006 con la creación de las Deep Belief Networks (DBN), que lograron entrenar redes profundas con muchas capas. Más adelante, en 2014, surgieron las Generative Adversarial Networks (GAN), revolucionando la capacidad de las redes neuronales para generar contenido nuevo y realista. Estos hitos han llevado a las redes neuronales a convertirse en herramientas esenciales en múltiples aplicaciones actuales de inteligencia artificial.


  #2. Principales arquitecturas: MLP, CNN, RNN.
     
 - Las principales arquitecturas de redes neuronales han evolucionado significativamente desde sus inicios en los años 50. El Perceptrón, desarrollado por Frank Rosenblatt en 1958, fue la primera arquitectura y consistía en una sola capa de neuronas que realizaba clasificaciones binarias simples. Aunque fue un gran avance, su limitación radicaba en no poder resolver problemas no lineales.

 - En 1965 surgió el Multilayer Perceptron (MLP), que amplió el perceptrón a múltiples capas: una de entrada, capas ocultas y una de salida. Esta arquitectura permitió modelar problemas más complejos, aunque el entrenamiento era inicialmente muy difícil porque los pesos debían asignarse manualmente. Con la introducción en los años 80 del algoritmo de retropropagación (backpropagation), el aprendizaje en redes profundas se hizo viable, habilitando el desarrollo de modelos más sofisticados.

 - En 1989, Yann LeCun propuso las Redes Neuronales Convolucionales (CNN), inspiradas en el córtex visual de los animales. Las CNN son especialmente efectivas en tareas como el reconocimiento de imágenes, usando capas convolucionales para extraer características importantes y capas de pooling para reducir dimensión sin perder información relevante. Por otra parte, las Redes Neuronales Recurrentes (RNN) y su avance, las Long Short-Term Memory (LSTM) en 1997, están diseñadas para trabajar con datos secuenciales y temporales, siendo útiles en procesamiento de lenguaje y series temporales.

 - Finalmente, la llegada del deep learning en 2006 introdujo las Deep Belief Networks (DBN), que pudieron entrenar redes con muchas capas, y en 2014 las Generative Adversarial Networks (GAN) revolucionaron la generación de contenido. Estas arquitecturas nuevas han ampliado enormemente las capacidades de las redes neuronales en inteligencia artificial.

 - Estos hitos configuran las principales arquitecturas fundamentales que han impulsado el desarrollo y aplicaciones actuales de las redes neuronales artificiales.

 #3. Algoritmos de entrenamiento: backpropagation, optimizadores.

- El algoritmo de retropropagación (backpropagation) es fundamental para el entrenamiento de redes neuronales artificiales. Su función principal es ajustar los pesos sinápticos para minimizar la función de pérdida, lo que mejora la capacidad de la red para generalizar a nuevos datos. Este ajuste se realiza propagando el error desde la capa de salida hacia las capas anteriores, permitiendo que cada neurona contribuya a corregir el modelo. Durante el entrenamiento, el algoritmo calcula la tasa a la que cada neurona afecta la pérdida general usando la regla de la cadena del cálculo diferencial, lo cual permite optimizar eficientemente todos los parámetros mediante métodos como el descenso del gradiente.

- La retropropagación se basa en dos fases principales: la propagación hacia adelante, donde se calcula la salida de la red para una entrada dada, y la propagación hacia atrás, que ajusta los pesos y sesgos para minimizar el error medido por una función de pérdida. Esta función cuantifica la discrepancia entre la salida predicha y la salida deseada. El proceso es iterativo y continúa hasta que la red alcanza un nivel aceptable de precisión. Además, la tasa de aprendizaje, que determina el tamaño de los pasos para actualizar los pesos, es un hiperparámetro clave que influye en la eficacia y velocidad del entrenamiento.

- Los optimizadores, como el descenso de gradiente estocástico (SGD), Adam o RMSProp, son algoritmos que utilizan la información del gradiente calculado para actualizar los pesos con el objetivo de reducir la función de pérdida. Optimizar estos parámetros es crucial para evitar problemas comunes como convergencia lenta o caer en mínimos locales. En conjunto, la retropropagación y los optimizadores forman la base del aprendizaje supervisado en redes neuronales, permitiendo entrenar desde perceptrones multicapa básicos hasta complejas redes profundas que son hoy la base de la inteligencia artificial moderna.
---

### 2. Diseño e implementación

#### 2.1 Arquitectura de la solución

* **Patrones de diseño**: ejemplo: Factory para capas, Strategy para optimizadores.
* **Estructura de carpetas (ejemplo)**:

  ```
  proyecto-final/
  ├── src/
  │   ├── layers/
  │   ├── optimizers/
  │   └── main.cpp
  ├── tests/
  └── docs/
  ```

#### 2.2 Manual de uso y casos de prueba

* **Cómo ejecutar**: `./build/neural_net_demo input.csv output.csv`
* **Casos de prueba**:

  * Test unitario de capa densa.
  * Test de función de activación ReLU.
  * Test de convergencia en dataset de ejemplo.

> *Personalizar rutas, comandos y casos reales.*

---

### 3. Ejecución

> **Demo de ejemplo**: Video/demo alojado en `docs/demo.mp4`.
> Pasos:
>
> 1. Preparar datos de entrenamiento (formato CSV).
> 2. Ejecutar comando de entrenamiento.
> 3. Evaluar resultados con script de validación.

---

### 4. Análisis del rendimiento

* **Métricas de ejemplo**:

  * Iteraciones: 1000 épocas.
  * Tiempo total de entrenamiento: 2m30s.
  * Precisión final: 92.5%.
* **Ventajas/Desventajas**:

  * * Código ligero y dependencias mínimas.
  * – Sin paralelización, rendimiento limitado.
* **Mejoras futuras**:

  * Uso de BLAS para multiplicaciones (Justificación).
  * Paralelizar entrenamiento por lotes (Justificación).

---

### 5. Trabajo en equipo

| Tarea                     | Miembro  | Rol                       |
| ------------------------- | -------- | ------------------------- |
| Investigación teórica     | Alumno A | Documentar bases teóricas |
| Diseño de la arquitectura | Alumno B | UML y esquemas de clases  |
| Implementación del modelo | Alumno C | Código C++ de la NN       |
| Pruebas y benchmarking    | Alumno D | Generación de métricas    |
| Documentación y demo      | Alumno E | Tutorial y video demo     |

> *Actualizar con tareas y nombres reales.*

---

### 6. Conclusiones

* **Logros**: Implementar NN desde cero, validar en dataset de ejemplo.
* **Evaluación**: Calidad y rendimiento adecuados para propósito académico.
* **Aprendizajes**: Profundización en backpropagation y optimización.
* **Recomendaciones**: Escalar a datasets más grandes y optimizar memoria.

---

### 7. Bibliografía

> *Actualizar con bibliografia utilizada, al menos 4 referencias bibliograficas y usando formato IEEE de referencias bibliograficas.*

- Aprende Machine Learning, "Breve Historia de las Redes Neuronales Artificiales", https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/, [En línea]. Disponible en: https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/. [Accedido: 24-11-2025].

- "CONCEPTOS BÁSICOS SOBRE REDES NEURONALES," Grupo de Tecnología de Computadores, Universidad de Sevilla. [En línea]. Disponible en: https://grupo.us.es/gtocoma/pid/pid10/RedesNeuronales.htm. [Accedido: 24-11-2025].

- BM, "¿Qué es la retropropagación?", IBM Think, [En línea]. Disponible en: https://www.ibm.com/mx-es/think/topics/backpropagation. [Accedido: 24-11-2025].

- Sánchez Medina, J. J. (1998). Linealización del algoritmo de backpropagation para el entrenamiento de redes neuronales (Proyecto fin de carrera). Universidad de Las Palmas de Gran Canaria. https://accedacris.ulpgc.es/bitstream/10553/1983/1/1235.pdf
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
