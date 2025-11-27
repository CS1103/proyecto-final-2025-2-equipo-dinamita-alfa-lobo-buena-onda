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


* 1. Historia y evolución de las NNs.
     
    * La historia de las redes neuronales artificiales (NNs) comenzó en 1943, cuando Warren McCulloch y Walter Pitts desarrollaron la neurona de McCulloch-Pitts, el primer modelo teórico que sentó las bases para entender y modelar el funcionamiento neuronal mediante circuitos eléctricos. Quince años después, en 1958, Frank Rosenblatt creó el Perceptrón, marcando el inicio del interés práctico al ser el primer modelo de red neuronal entrenable, aunque limitado a la resolución de problemas linealmente separables. 

    * La primera gran pausa se superó en la década de 1980 con la reintroducción del algoritmo de Retropropagación (Backpropagation). Este avance fue crucial, ya que permitió entrenar eficientemente redes neuronales con múltiples capas, superando las limitaciones del Perceptrón. El desarrollo se aceleró significativamente. En 1989, Yann LeCun propuso las Redes Neuronales Convolucionales (CNN), inspiradas en el córtex visual y optimizadas para el reconocimiento de imágenes. 
 
    * El verdadero salto al Deep Learning ocurrió en 2006 con la creación de las Deep Belief Networks (DBN), que hicieron viable el entrenamiento de redes profundas con muchas capas (decenas o cientos). Posteriormente, en 2014, surgieron las Generative Adversarial Networks (GAN), que revolucionaron la capacidad de las redes para generar contenido nuevo y fotorrealista. Estos hitos han transformado las NNs en herramientas esenciales para múltiples aplicaciones de inteligencia artificial moderna.

* 2. Principales arquitecturas: MLP, CNN, RNN.
     
    * Las arquitecturas fundamentales de las redes neuronales están diseñadas para manejar tipos de datos y problemas específicos, cada una con una estructura única:

    * a) Perceptrón Multicapa (MLP)
 
        * El MLP es la arquitectura fundamental de las redes neuronales feedforward (de alimentación hacia adelante). Se compone de una capa de entrada, una o más capas ocultas y una capa de salida.
         
         * Función: Permite modelar relaciones complejas y resolver problemas no lineales al aplicar funciones de activación en las capas ocultas.
         
         * Uso: Clasificación y regresión en datos tabulares y en tareas donde las características de entrada son fijas.

    * b) Redes Neuronales Convolucionales (CNN)

        * Las CNN son la arquitectura estándar para el procesamiento de datos con una estructura de cuadrícula, como las imágenes.
 
        * Composición: Utilizan capas convolucionales que aplican filtros para extraer características importantes (bordes, texturas) de la entrada, y capas de pooling para reducir la dimensionalidad de los datos sin perder información crítica.
 
        * Uso: Reconocimiento de imágenes, visión por computadora, detección de objetos y análisis de vídeo.
 
    * c) Redes Neuronales Recurrentes (RNN)
 
        * Las RNN están diseñadas específicamente para manejar datos secuenciales y temporales, donde la salida en un momento $t$ depende de las entradas y los estados de momentos $t-1$.

        * Composición: Introducen un bucle de retroalimentación que permite que la información persista entre pasos de tiempo, dándoles una "memoria".
 
        * Limitación y Avance: Las RNN básicas sufren del problema del gradiente desvaneciente al manejar secuencias largas. Esto se resolvió con la creación de las Long Short-Term Memory (LSTM), que introducen celdas de memoria y compuertas (input, forget, output) para regular el flujo de información.
 
        * Uso: Procesamiento de Lenguaje Natural (PLN), series temporales, traducción automática y reconocimiento de voz.

* 3. Algoritmos de entrenamiento: backpropagation, optimizadores.

    * a) Retropropagación (Backpropagation)
 
        * La retropropagación es el algoritmo central que permite el aprendizaje en redes neuronales. Su objetivo es ajustar los pesos sinápticos de la red para minimizar la función de pérdida (o costo), mejorando la precisión del modelo.
 
        * El proceso opera en dos fases iterativas:
        
        * Propagación hacia Adelante (Forward Pass): La entrada se propaga desde la primera capa hasta la capa de salida para calcular la predicción de la red. La función de pérdida cuantifica la discrepancia entre la salida predicha y la salida deseada (valor real).
        
        * Propagación hacia Atrás (Backward Pass): El error calculado se propaga desde la capa de salida hacia atrás. Usando la regla de la cadena del cálculo diferencial, el algoritmo calcula el gradiente (la tasa a la que cada peso y sesgo afecta la pérdida general).
        
        * Este gradiente es crucial, ya que indica la dirección en la que deben ajustarse los pesos y sesgos para reducir el error de la red.

    * b) Optimizadores
 
        * Los optimizadores son algoritmos que utilizan la información del gradiente, calculada por la retropropagación, para realizar la actualización efectiva de los pesos de la red a lo largo del tiempo. Su meta es encontrar el conjunto óptimo de pesos que minimice la función de pérdida.
        
        * Descenso del Gradiente (Gradient Descent): Es la base de todos los optimizadores. Mueve los pesos en la dirección opuesta al gradiente (la pendiente más pronunciada hacia el "valle" de la pérdida). La Tasa de Aprendizaje es un hiperparámetro clave que determina el tamaño de los pasos dados en esta dirección.
        
        * Optimizadores Comunes: El Descenso de Gradiente Estocástico (SGD), Adam y RMSProp son variantes avanzadas que ajustan dinámicamente la tasa de aprendizaje o incorporan momentos anteriores para acelerar la convergencia y evitar problemas como la convergencia lenta o caer en mínimos locales.
 
    * En conjunto, la retropropagación proporciona el gradiente del error, y los optimizadores lo utilizan para guiar el proceso de aprendizaje supervisado, permitiendo el entrenamiento de complejas arquitecturas de deep learning.

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

- W. S. McCulloch y W. Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity". Disponible en: https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity, 2024.

- Angelvillazon.com, "Historia de las redes neuronales en la Inteligencia Artificial," 2025. [Online]. Available: https://www.angelvillazon.com/inteligencia-artificial-robotica/historia-de-las-redes-neuronales-en-la-inteligencia-artificial/

- Lamaquinaoraculo.com, "Neuronas de McCulloch y Pitts - Artículo de LMO," 2025. [Online]. Available: https://lamaquinaoraculo.com/deep-learning/el-modelo-neuronal-de-mcculloch-y-pitts/
  
---

### Licencia

Este proyecto usa la licencia **MIT**. Ver [LICENSE](LICENSE) para detalles.

---
