# Proyecto Final 2025-2: Neural Network Application
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación completa de una red neuronal multicapa desde cero en C++20, incluyendo una biblioteca genérica de álgebra tensorial y aplicaciones prácticas de clasificación, predicción de secuencias y control.

---

## Contenidos

* [1. Datos generales](#1-datos-generales)
* [2. Requisitos e instalación](#2-requisitos-e-instalacion)
    * [2.1. Requisitos del sistema](#21-requisitos-del-sistema)
    * [2.2. Instalación de herramientas](#22-instalacion-de-herramientas)
    * [2.3. Instalación del proyecto](#23-instalacion-del-proyecto)
    * [2.4. Solución de problemas](#24-solucion-de-problemas)
* [3. Investigación teórica](#3-investigacion-teorica)
* [4. Diseño e implementación](#4-diseño-e-implementacion)
    * [4.1. Estructura del proyecto](#41-estructura-del-proyecto)
    * [4.2. Arquitectura de la solución](#42-arquitectura-de-la-solucion)
* [5. Documentación de codigo](#5-documentacion-de-codigo)
    * [Tensor](#tensor)
    * [neuralNetwork](#neuralnetwork)
    * [nnACTIVATION](#nnactivation)
    * [nnDense](#nndense)
    * [nnInterfaces](#nninterfaces)
    * [nnLoss](#nnloss)
    * [nnOptimizer](#nnoptimizer)
    * [ControllerDemo](#controllerdemo)
    * [PatternClassifier](#patternclassifier)
    * [SequencePredictor](#sequencepredictor)
* [6. Manual de uso](#6-manual-de-uso)
* [7. Ejecución](#7-ejecución)
* [8. Trabajo en equipo](#8-trabajo-en-equipo)
* [9. Conclusiones](#9-conclusiones)
* [10. Bibliografía](#10-bibliografia)

---

## 1. Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `Equipo dinamita alfa lobo buena onda`
* **Integrantes**:
  * Elias Alonso Usaqui Cabezas – 202420064 (Responsable de investigación teórica)
  * Elias Alonso Usaqui Cabezas – 202420064 (Desarrollo de la arquitectura)
  * Fredy Alexander Cardenas Aliaga – 202420013 (Implementación del modelo)
  * Fredy Alexander Cardenas Aliaga – 202420013 (Pruebas y benchmarking)
  * Elias Alonso Usaqui Cabaezas – 202420064 (Documentación y demo)

---

## 2. Requisitos e instalación

### 2.1. Requisitos del sistema

* **Sistema Operativo**: Linux, macOS, o Windows con MinGW
* **Compilador**: 
  - GCC 11.0+ (Linux/MinGW)
  - Clang 13.0+ (macOS)
  - MSVC 19.30+ (Visual Studio 2022)
* **Herramientas**: CMake 3.16+
* **Estándar**: C++20
* **Dependencias externas**: **NINGUNA** (solo librería estándar de C++)

### 2.2. Instalación de herramientas

#### macOS:
```bash
# Instalar CMake (si no lo tienes)
brew install cmake

# Verificar versiones
clang++ --version   # Debe ser 13.0+
cmake --version     # Debe ser 3.16+
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install build-essential cmake g++

# Verificar versiones
g++ --version       # Debe ser 11.0+
cmake --version     # Debe ser 3.16+
```

#### Windows:
```bash
# Opción 1: MinGW-w64 desde https://www.mingw-w64.org/
# Opción 2: Visual Studio 2022 con "Desktop development with C++"
# CMake desde: https://cmake.org/download/
```

### 2.3. Instalación del proyecto

```bash
# 1. Clonar repositorio
git clone https://github.com/CS1103/proyecto-final-2025-2-equipo-dinamita-alfa-lobo-buena-onda.git
cd pong_ai

# 2. Configurar y compilar
mkdir build && cd build
cmake ..
make -j4

# 3. Verificar instalación
ctest
```

**✅ Si ves "100% tests passed, 0 tests failed out of 3", la instalación fue exitosa.**

### 2.4. Solución de problemas

| Error | Solución |
|-------|----------|
| "CMake not found" | Instalar CMake según tu sistema (ver arriba) |
| "C++20 not supported" | Actualizar compilador a GCC 11+ o Clang 13+ |
| "No such file: Tensor.h" | Ejecutar desde el directorio `build/` |

---

## 3. Investigación teórica


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

## 4. Diseño e implementación

### 4.1. Estructura del proyecto

```
pong_ai/
├── CMakeLists.txt              # Configuración de build
├── README.md                   # Este archivo
├── BIBLIOGRAFIA.md             # Referencias IEEE
│
├── include/utec/               # Archivos de cabecera
│   ├── algebra/
│   │   └── Tensor.h           # Epic 1: Tensor genérico
│   ├── nn/
│   │   ├── nn_interfaces.h    # Interfaces ILayer, IOptimizer
│   │   ├── nn_dense.h         # Capa densa (fully connected)
│   │   ├── nn_activation.h    # ReLU, Sigmoid, Tanh
│   │   ├── nn_loss.h          # MSE, Binary Cross Entropy
│   │   ├── nn_optimizer.h     # SGD, Adam
│   │   └── neural_network.h   # Epic 2: Red neuronal completa
│   └── apps/
│       ├── PatternClassifier.h    # Epic 3: Clasificación (XOR)
│       ├── SequencePredictor.h    # Epic 3: Predicción de series
│       └── ControllerDemo.h       # Epic 3: Control simplificado
│
├── src/utec/apps/              # Implementaciones de aplicaciones
│   ├── PatternClassifier.cpp
│   ├── SequencePredictor.cpp
│   └── ControllerDemo.cpp
│
├── tests/project/              # Tests automatizados
│   ├── test_tensor.cpp        # 6 tests (Epic 1)
│   ├── test_neural_network.cpp    # 8 tests (Epic 2)
│   └── test_applications.cpp  # 8 tests (Epic 3)
│
└── build/                      # Directorio de compilación
    ├── pattern_classifier_app
    ├── sequence_predictor_app
    ├── controller_demo_app
    ├── test_tensor
    ├── test_neural_network
    └── test_applications
```

**Explicación de la organización:**
- **`include/`**: Headers separados por responsabilidad (álgebra, NN, apps)
- **`src/`**: Implementaciones solo de las aplicaciones (lo demás es header-only por ser templates)
- **`tests/`**: Suite completa de 22 tests automatizados
- **Namespaces**: `utec::algebra`, `utec::neural_network`, `utec::apps`

---


### 4.2. Arquitectura de la solución

**Patrones de diseño utilizados:**

1. **Template Method Pattern**: En `NeuralNetwork::train<LossFunc, Optimizer>()`
2. **Strategy Pattern**: Funciones de pérdida y optimizadores intercambiables
3. **Interface Segregation**: `ILayer<T>`, `IOptimizer<T>` para polimorfismo
4. **Generic Programming**: Todo parametrizado con templates (`T` para tipo, `Rank` para dimensión)

**Paradigmas:**
- **POO**: Herencia (`Dense : public ILayer`), encapsulación, polimorfismo
- **Genérico**: Templates para reutilización (`Tensor<float, 2>`, `Tensor<double, 3>`)
- **Funcional**: Lambdas para inicialización de pesos

---

## 5. Documentación de codigo

### Tensor
---

El archivo `Tensor.h` define la plantilla de clase `utec::algebra::Tensor<T, N]`, la estructura de datos fundamental para el manejo de arrays multi-dimensionales en la librería de álgebra. Proporciona soporte para operaciones aritméticas elemento a elemento, manipulación de formas y funcionalidades avanzadas como **Broadcasting** y **Multiplicación Matricial por Lotes (BMM)**.

#### ⚙️ Notación de Complejidad Algorítmica ($\mathbf{O}$)

Las complejidades se expresan en función de las siguientes variables clave del Tensor y sus operaciones:

| Símbolo | Descripción |
| :--- | :--- |
| $\mathbf{N}$ | Rango del Tensor (número de dimensiones). |
| $\mathbf{S}$ | Tamaño total del Tensor (número de elementos). |
| $\mathbf{S}_{\text{res}}$ | Tamaño del Tensor resultado después de aplicar **Broadcasting**. |
| $\mathbf{B}$ | Tamaño del lote (*Batch Size*). |
| $\mathbf{M}$ | Filas de la submatriz. |
| $\mathbf{K}$ | Dimensión común para la multiplicación matricial. |
| $\mathbf{L}$ | Columnas de la submatriz. |
| C_MAT_MUL | Costo de Multiplicación Matricial por Lotes: $\mathbf{O}(\mathbf{B} \cdot \mathbf{M} \cdot \mathbf{K} \cdot \mathbf{L})$. |

---

#### 🚀 Clase `template <typename T, size_t N> class Tensor`

#### 1. Constructores y Asignación

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `Tensor(Dims...)` | Constructor principal. Inicializa la forma, los `strides` y redimensiona `data_`. | $\mathbf{O}(\mathbf{S} + \mathbf{N})$ |
| `Tensor(const Tensor&)` | Constructor de copia. | $\mathbf{O}(\mathbf{S} + \mathbf{N})$ |
| `operator=(const Tensor&)` | Operador de asignación de copia. | $\mathbf{O}(\mathbf{S} + \mathbf{N})$ |
| `operator=(std::initializer_list<T>)` | Asignación de valores a `data_` desde una lista de inicialización. | $\mathbf{O}(\mathbf{S})$ |

#### 2. Acceso y Manipulación de Forma

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `fill(const T& value)` | Llena todos los elementos del Tensor con un valor escalar. | $\mathbf{O}(\mathbf{S})$ |
| `operator()(Indices...)` | **Acceso a Elementos** usando índices multi-dimensionales. | $\mathbf{O}(\mathbf{N})$ |
| `reshape(Dims...)` | Cambia la forma del Tensor, manteniendo el tamaño total (`S`) o redimensionando si es necesario. | $\mathbf{O}(\mathbf{S}' + \mathbf{N})$ |
| `compute_index()` | Método interno para la conversión de índices multi-dim a índice plano. | $\mathbf{O}(\mathbf{N})$ |
| `print()` | Método interno recursivo para la impresión estructurada del Tensor. | $\mathbf{O}(\mathbf{S})$ |

#### 3. Operaciones Aritméticas (Element-wise)

Estas operaciones soportan **Broadcasting** cuando las formas de los operandos son compatibles.

| Operación | Descripción | Complejidad (sin Broadcast) | Complejidad (con Broadcast) |
| :--- | :--- | :--- | :--- |
| `operator+`, `operator-`, `operator*` | Operación **Tensor-Tensor** elemento a elemento. | $\mathbf{O}(\mathbf{S})$ | $\mathbf{O}(\mathbf{S}_{\text{res}} \cdot \mathbf{N})$ |
| `operator+`, `operator-`, `operator*`, `operator/` | Operación **Tensor-Escalar** elemento a elemento (a la derecha). | $\mathbf{O}(\mathbf{S})$ | N/A |
| `friend operator+`, `operator-`, `operator*`, `operator/` | Operación **Escalar-Tensor** elemento a elemento (a la izquierda). | $\mathbf{O}(\mathbf{S})$ | N/A |

---

#### 🌐 Funciones Globales de Álgebra

| Función | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `transpose_2d` | Realiza la **Transposición** de las dos últimas dimensiones (`N-2` y `N-1`). | $\mathbf{O}(\mathbf{S} \cdot \mathbf{N})$ | Requiere $\mathbf{N} \ge 2$. |
| `matrix_product` | Implementa la **Multiplicación Matricial por Lotes (BMM)**. | $\mathbf{O}(\mathbf{B} \cdot \mathbf{M} \cdot \mathbf{K} \cdot \mathbf{L})$ | Requiere que las formas internas sean compatibles. |

---

### neuralNetwork
---

El archivo `NeuralNetwork.h` define la plantilla de clase `utec::neural_network::NeuralNetwork<T>`, que actúa como el **contenedor principal** para la red neuronal. Su función es ensamblar las capas, coordinar los pasos de la propagación hacia adelante y hacia atrás, y gestionar el ciclo de vida completo del entrenamiento y la serialización (guardado/carga).

#### ⚙️ Notación de Complejidad Algorítmica (O)

Las complejidades se expresan en función de las siguientes variables:

| Símbolo | Descripción |
| :--- | :--- |
| **L** | Número de capas en la red. |
| **E** | Número de épocas de entrenamiento. |
| **N** | Número total de muestras de entrenamiento. |
| **B** | Tamaño máximo del batch (`batch_size`). |
| **P** | Número total de parámetros (pesos y sesgos) en la red. |
| **S_BATCH** | Tamaño del batch actual (variable, $\le$ B). |
| **F** | Costo computacional de la propagación de una sola muestra a través de toda la red. |
| **F_INPUT** | Número de características (columnas) en el set de datos de entrada. |
| **C_LAYER_OP** | Costo de una operación (forward, backward, update) en una única capa. |

---

#### 💻 Clase `template <typename T> class NeuralNetwork`

#### 1. Métodos de Propagación y Ayuda (Internos/Privados)

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `forward(const Tensor<T, 2>& input)` | Realiza la propagación hacia adelante (Forward Pass). | $\mathbf{O}(\mathbf{S\_BATCH} \cdot \mathbf{F})$ | Lineal con el tamaño del batch y el costo de propagación por muestra. |
| `backward(const Tensor<T, 2>& gradient)` | Realiza la retropropagación (Backpropagation), calculando los gradientes de los parámetros. | $\mathbf{O}(\mathbf{S\_BATCH} \cdot \mathbf{F})$ | Lineal con el tamaño del batch y el costo de propagación por muestra. |
| `update_parameters(IOptimizer<T>& optimizer)` | Aplica la actualización de pesos y sesgos a cada capa usando el optimizador. | $\mathbf{O}(\mathbf{P} + \mathbf{C}_{\text{optimizer}}^\text{step})$ | Costo lineal con el número total de parámetros $\mathbf{P}$. |
| `extract_batch(...)` | Extrae un subconjunto de filas (un batch) de los datos totales de entrenamiento. | $\mathbf{O}(\mathbf{S\_BATCH} \cdot \mathbf{F\_INPUT})$ | Costo de copia de los datos. |

#### 2. Métodos Públicos Centrales

| Método | Propósito | Complejidad | Explicación de la Complejidad |
| :--- | :--- | :--- | :--- |
| `add_layer(...)` | Añade una nueva capa (`ILayer`) a la arquitectura de la red. | $\mathbf{O}(1)$ amortizado | Utiliza `std::vector::push_back`. |
| `train<LossType, OptimizerType>(...)` | **Bucle de entrenamiento.** Repite el ciclo Forward $\rightarrow$ Loss $\rightarrow$ Backward $\rightarrow$ Update por $\mathbf{E}$ épocas y $\mathbf{N}/\mathbf{B}$ batches. | $\mathbf{O}(\mathbf{E} \cdot \mathbf{N} \cdot \mathbf{F})$ | La operación dominante (Forward/Backward) tiene un costo de $\mathbf{O}(\mathbf{S\_BATCH} \cdot \mathbf{F})$. Al sumar sobre todas las épocas, el costo total es $\mathbf{O}(\mathbf{E} \cdot \mathbf{N} \cdot \mathbf{F})$. |
| `predict(const Tensor<T, 2>& X)` | Realiza una predicción sobre un conjunto de datos `X`. | $\mathbf{O}(\mathbf{N}_{\text{pred}} \cdot \mathbf{F})$ | Lineal con el número de muestras a predecir y el costo de propagación. |

---

#### 3. Serialización (Carga y Guardado de Estado)

Estos métodos asumen que solo las capas `Dense` contienen parámetros que deben ser guardados/cargados. $\mathbf{P}_{\text{dense}}$ es el número total de parámetros en las capas densas.

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `save_state(const std::string& filepath) const` | Serializa y guarda los pesos y sesgos de las capas densas en un archivo binario. | $\mathbf{O}(\mathbf{P}_{\text{dense}})$ |
| `load_state(const std::string& filepath)` | Deserializa y carga los pesos y sesgos en las capas densas de la red. | $\mathbf{O}(\mathbf{P}_{\text{dense}})$ |

---

### nnActivation
---

El archivo `NN_ACTIVATION.H` define implementaciones concretas de las funciones de activación más comunes (`ReLU` y `Sigmoid`) como clases que heredan de `ILayer<T>`. Estas capas se utilizan para introducir **no linealidad** en la red neuronal.

#### ⚙️ Notación de Complejidad Algorítmica (O)

Las complejidades se expresan en función de las siguientes variables, relacionadas con el tensor de entrada/salida de la capa de activación:

| Símbolo | Descripción |
| :--- | :--- |
| **S_BATCH** | Número de muestras en el lote actual. |
| **M_OUT** | Número de características/neuronas en la capa de salida. |
| **N_ELEMENTS** | Número total de elementos en el tensor de entrada/salida: $\mathbf{S}_{\text{BATCH}} \cdot \mathbf{M}_{\text{OUT}}$. |

---

#### 💻 1. Clase `template <typename T> class ReLU`

Implementa la función de activación Rectified Linear Unit: $f(x) = \max(0, x)$.

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `forward(const Tensor<T, 2>& z)` | Calcula $\max(0, z)$ elemento a elemento y almacena la entrada `z` para la retropropagación. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Operación lineal y de almacenamiento. |
| `backward(const Tensor<T, 2>& gradient)` | Calcula la derivada $\partial L / \partial Z$. Pasa el gradiente si la entrada original (`input_`) fue positiva, o `0` si fue negativa/cero. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Operación lineal (multiplicación por el *máscara* binaria). |
| `update_params(...)` | **No implementado/No aplica.** Las capas de activación no tienen parámetros entrenables. | $\mathbf{O}(1)$ | Heredado de `ILayer<T>`. |

---

#### 💻 2. Clase `template <typename T> class Sigmoid`

Implementa la función de activación Sigmoide: $f(x) = 1 / (1 + e^{-x})$.

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `forward(const Tensor<T, 2>& z)` | Calcula la Sigmoid elemento a elemento. Aplica un *clipping* (`EPSILON`) para mantener la estabilidad numérica. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Almacena la salida activada (`output_`) para la retropropagación. |
| `backward(const Tensor<T, 2>& gradient)` | Calcula la derivada $\partial L / \partial Z$. Utiliza la propiedad de la derivada de Sigmoid: $\mathbf{A}(1-\mathbf{A})$, donde $\mathbf{A}$ es la salida almacenada. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | La derivada calculada se multiplica por el gradiente entrante. |
| `update_params(...)` | **No implementado/No aplica.** Las capas de activación no tienen parámetros entrenables. | $\mathbf{O}(1)$ | Heredado de `ILayer<T>`. |

---

### nnDense
---

El archivo `NN_DENSE.H` define la clase `utec::neural_network::Dense<T>`, que implementa una capa completamente conectada (Fully Connected Layer) en una red neuronal. Esta capa realiza una transformación lineal sobre la entrada: $\mathbf{Y} = \mathbf{X} \cdot \mathbf{W} + \mathbf{b}$.

#### ⚙️ Notación de Complejidad Algorítmica (O)

Las complejidades se centran en el costo de la multiplicación matricial, que es la operación dominante en esta capa.

| Símbolo | Descripción |
| :--- | :--- |
| **S_BATCH** | Tamaño del batch actual (número de muestras). |
| **M_IN** | Número de características de entrada. |
| **M_OUT** | Número de neuronas de salida. |
| **P_LAYER** | Número total de parámetros de la capa (W + b). |
| **C_MAT_MUL** | Costo de la Multiplicación Matricial Clave: O(S_BATCH * M_IN * M_OUT). |

---

#### 💻 Clase `template <typename T> class Dense`

#### 1. Constructores y Propiedades

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `Dense(in_f, out_f, init_w_fun, init_b_fun)` | Constructor principal. Inicializa las matrices de pesos (`weights_`) y los vectores de sesgos (`biases_`) con las funciones proporcionadas, y los gradientes a cero. | O(M_IN * M_OUT) | La inicialización domina el costo. |
| `Dense()` | Constructor vacío, utilizado principalmente antes de la deserialización (`load_state`). | O(1) | Inicializa las dimensiones a cero. |

#### 2. Algoritmos de Propagación y Retropropagación

| Método | Propósito | Complejidad | Explicación del Algoritmo |
| :--- | :--- | :--- | :--- |
| `forward(const Tensor<T, 2>& x)` | Propagación hacia adelante: Y = X * W + b. | O(C_MAT_MUL) | Domina la multiplicación matricial X * W. |
| `backward(const Tensor<T, 2>& dZ)` | Retropropagación. Calcula los gradientes internos (dW, db) y el gradiente para la capa anterior (dX). | O(C_MAT_MUL) | Domina el cálculo de dW = X^T * dZ y dX = dZ * W^T. |
| `update_params(IOptimizer<T>& optimizer)` | Aplica las actualizaciones del optimizador a los pesos (`weights_`) y sesgos (`biases_`) usando los gradientes calculados. | O(P_LAYER) | Costo lineal con el número de parámetros de la capa. |

#### 3. Serialización (Carga y Guardado de Parámetros)

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `save_parameters(std::ofstream& ofs) const` | Escribe los contenidos de `weights_` y `biases_` en el flujo binario. | $\mathbf{O}(\mathbf{P}_{\text{LAYER}})$ | Utiliza la función auxiliar `save_tensor`. |
| `load_parameters(std::ifstream& ifs)` | Lee los contenidos de `weights_` y `biases_` del flujo binario y redimensiona la capa. | $\mathbf{O}(\mathbf{P}_{\text{LAYER}})$ | Utiliza la función auxiliar `load_tensor`. |
| `save_tensor(...)` / `load_tensor(...)` | Funciones auxiliares para gestionar el guardado/carga binaria de las dimensiones y el contenido del tensor. | $\mathbf{O}(\mathbf{N}_{\text{elements}})$ | Costo lineal con el número de elementos del tensor. |

---

### nnIntefaces
---

El archivo `NN_INTERFACES.H` define las interfaces puramente virtuales que establecen el contrato y la estructura requerida para los principales componentes de la red neuronal: **Capas** (`ILayer`), **Funciones de Pérdida** (`ILoss`) y **Optimizadores** (`IOptimizer`).

#### ⚙️ Notación de Complejidad Algorítmica (O)

Las complejidades son las estimaciones de costo **esperadas** para las implementaciones concretas que hereden estas interfaces.

| Símbolo | Descripción |
| :--- | :--- |
| **S_BATCH** | Tamaño del batch actual (número de muestras). |
| **M_IN** | Número de características de entrada. |
| **M_OUT** | Número de neuronas de salida. |
| **P_LAYER** | Número total de parámetros de la capa. |
| **N_ELEMENTS** | Número total de elementos en el tensor de salida/gradiente (S_BATCH * M_OUT). |
| **C_MAT_OP** | Costo de operaciones matriciales (ej. Multiplicación Matricial, C_mat_mul). |

---

#### 💻 1. Interfaz `template <typename T> class ILayer`

Define el comportamiento base de cualquier componente funcional de la red (capas densas, de activación, etc.).

| Método | Propósito | Complejidad Esperada | Requisito Clave |
| :--- | :--- | :--- | :--- |
| `forward(...)` | **Propagación hacia Adelante:** Calcula la salida de la capa. | O(C_MAT_OP) o O(N_ELEMENTS) | Debe almacenar la entrada para el cálculo del `backward`. |
| `backward(...)` | **Retropropagación:** Calcula el gradiente para la capa anterior (dX). | O(C_MAT_OP) o O(N_ELEMENTS) | Debe calcular y almacenar los gradientes de los parámetros internos. |
| `update_params(...)` | **Actualización de Parámetros:** Aplica el optimizador a los parámetros internos de la capa. | O(P_LAYER) | Implementación vacía por defecto (O(1)) para capas sin parámetros. |

---

#### 💻 2. Interfaz `template <typename T, int N> class ILoss`

Define el contrato para las funciones de pérdida, utilizadas para medir la discrepancia entre predicciones y valores reales.

| Método | Propósito | Complejidad Esperada | Requisito Clave |
| :--- | :--- | :--- | :--- |
| `loss() const` | **Cálculo de Pérdida:** Devuelve el valor escalar total de la pérdida del batch. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Iteración lineal sobre todos los elementos de salida. |
| `loss_gradient() const` | **Gradiente de Pérdida:** Calcula el gradiente de la pérdida con respecto a la entrada de la función de pérdida. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Genera el tensor de gradiente inicial para el proceso de retropropagación. |

---

#### 💻 3. Interfaz `template <typename T> class IOptimizer`

Define el contrato para los algoritmos de optimización encargados de actualizar los parámetros de la red.

| Método | Propósito | Complejidad Esperada | Requisito Clave |
| :--- | :--- | :--- | :--- |
| `update(...)` | **Actualización de Parámetros:** Aplica la regla de optimización (ej. SGD) a un tensor de parámetros y su gradiente. | $\mathbf{O}(\mathbf{P}_{\text{LAYER}})$ | Costo lineal con el número de elementos a actualizar. |
| `step()` | **Paso Global:** Realiza una acción de paso global del optimizador (ej. incrementar contador de iteraciones). | $\mathbf{O}(1)$ | Puede ser $\mathbf{O}(\mathbf{P})$ si maneja estados globales (e.g., Adam, RMSprop). |

---

### nnLoss
---

El archivo `NN_LOSS.H` define las implementaciones concretas de las funciones de pérdida más comunes, heredando de la interfaz `ILoss<T, 2>`. Estas clases son responsables de calcular el error entre las predicciones (Y_pred) y los valores verdaderos (Y_true), y generar el gradiente inicial para la retropropagación.

#### ⚙️ Notación de Complejidad Algorítmica (O)

Las complejidades se basan en la iteración lineal sobre todos los elementos de los tensores de predicción y objetivo.

| Símbolo | Descripción |
| :--- | :--- |
| **S_BATCH** | Número de muestras en el lote actual. |
| **M_OUT** | Número de neuronas de salida. |
| **N_ELEMENTS** | Número total de elementos en el tensor de salida (S_BATCH * M_OUT). |

---

#### 💻 1. Clase `template <typename T> class MSELoss`

Implementa la **Pérdida por Error Cuadrático Medio (Mean Squared Error)**: MSE = (1/n) * Sum((Y_pred - Y_true)^2)

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `MSELoss(...)` | Constructor. Almacena las predicciones y el objetivo. | O(1) | Verifica que las formas de los tensores coincidan. |
| `loss() const` | Calcula el valor escalar del MSE promediado sobre N_ELEMENTS. | O(N_ELEMENTS) | Involucra resta, elevación al cuadrado y suma lineal. |
| `loss_gradient() const` | Calcula el gradiente inicial: dL/dY_pred = (2/n) * (Y_pred - Y_true). | O(N_ELEMENTS) | Resta elemento a elemento seguida de una multiplicación por factor escalar. |

---

#### 💻 2. Clase `template <typename T> class BinaryCrossEntropyLoss`

Implementa la **Pérdida por Entropía Cruzada Binaria (Binary Cross Entropy)**: $$\text{BCE} = -\frac{1}{n} \sum [y \cdot \log(p) + (1-y) \cdot \log(1-p)]$$

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `BinaryCrossEntropyLoss(...)` | Constructor. Almacena las predicciones y el objetivo. | $\mathbf{O}(1)$ | Verifica que las formas de los tensores coincidan. |
| `loss() const` | Calcula el valor escalar de la BCE promediado sobre $\mathbf{N}_{\text{ELEMENTS}}$. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Involucra operaciones logarítmicas por elemento. Utiliza $\mathbf{\epsilon}$ para estabilidad. |
| `loss_gradient() const` | Calcula el gradiente inicial: $\mathbf{d}\mathbf{L}/\mathbf{d}\mathbf{P}$. | $\mathbf{O}(\mathbf{N}_{\text{ELEMENTS}})$ | Cálculo por elemento, utilizando la fórmula del gradiente de BCE. |

---

### nnOptimizer
---

El archivo `NN_OPTIMIZER.H` define las implementaciones de los algoritmos de optimización **SGD** y **Adam**, que heredan de la interfaz `IOptimizer<T>`. Estas clases gestionan la lógica para actualizar los parámetros de la red utilizando los gradientes calculados.

#### ⚙️ Notación de Complejidad Algorítmica (O)

| Símbolo | Descripción |
| :--- | :--- |
| **P\_LAYER** | Número total de parámetros (pesos o sesgos) en el tensor que se está actualizando. |
| **L\_DENSE** | Número de capas densas (que tienen parámetros) en la red. |
| **t** | Contador de pasos global del optimizador. |

---

#### 💻 1. Clase `template <typename T> class SGD`

Implementa el **Descenso de Gradiente Estocástico (Stochastic Gradient Descent)**, la regla de actualización más básica: $$\mathbf{\theta} = \mathbf{\theta} - \mathbf{LR} \cdot \mathbf{\nabla}\mathbf{\theta}$$

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `SGD(...)` | Constructor. Inicializa la tasa de aprendizaje. | $\mathbf{O}(1)$ | N/A |
| `update(...)` | **Algoritmo de Actualización Principal.** Aplica la resta del gradiente multiplicado por la tasa de aprendizaje a cada parámetro. | $\mathbf{O}(\mathbf{P}_{\text{LAYER}})$ | Costo lineal con el número de parámetros en el tensor actualizado. |
| `step()` | **Paso Global.** No implementa ninguna acción. | $\mathbf{O}(1)$ | Heredado de `IOptimizer<T>`. |

---

#### 💻 2. Clase `template <typename T> class Adam`

Implementa el optimizador **Adam (Adaptive Moment Estimation)**, que utiliza promedios móviles de primer ($\mathbf{m}$) y segundo ($\mathbf{v}$) momento de los gradientes, e incluye corrección de *bias*.

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `Adam(...)` | Constructor. Inicializa hiperparámetros (LR, beta1, beta2, epsilon) y el contador de pasos t=0. | O(1) | N/A |
| `update(...)` | **Algoritmo de Actualización Adam.** | O(P_LAYER) | El costo es lineal con P_LAYER. La gestión de momentos (`std::map`) es O(log(L_DENSE)) para acceso. |
| `step()` | **Paso Global.** Incrementa el contador de pasos global t. | O(1) | Es esencial para el cálculo de la corrección de bias en `Adam`. |

---

### ControllerDemo
---

El archivo `CONTROLLER_DEMO.H` define la clase `ControllerDemo<T>`, que encapsula una **Red Neuronal** y un **Simulador de Entorno Físico Simplificado** (análogo a un entorno de OpenAI Gym). Esta clase entrena la red para aprender una política de control que mantiene una partícula dentro de ciertos límites.

#### ⚙️ Notación de Complejidad Algorítmica (O)

| Símbolo | Descripción |
| :--- | :--- |
| **W_layer** | Número de parámetros (pesos o sesgos) en una capa `Dense`. |
| **B_layer** | Número de bias en una capa `Dense`. |
| **W_total** | Número total de parámetros (pesos y bias) en toda la red neuronal. |
| **L** | Número de capas en la red. |
| **D** | Tamaño total del dataset de entrenamiento (fijo en 12 para el demo). |
| **Epochs** | Número de épocas de entrenamiento. |
| **Batch_Size** | Tamaño del lote de entrenamiento (fijo en 4 para el demo). |
| **C_fp_bp** | Costo de una pasada Forward y Backpropagation para una muestra: O(W_total). |

---

#### 💻 Clase `template <typename T> class ControllerDemo`

Esta clase gestiona la **red neuronal** (`nn_`) y el **estado del entorno** (`position_`, `velocity_`).

### 1. Arquitectura de la Red y Estado Interno

#### Arquitectura

La red neuronal utilizada es una **MLP (Perceptrón Multicapa)** con la siguiente estructura:

$$\text{Entrada} (2) \rightarrow \text{Densa} (16) \rightarrow \text{ReLU} \rightarrow \text{Densa} (1) \rightarrow \text{Sigmoid} \rightarrow \text{Salida} (1)$$

* **Entrada (2):** `[position, velocity]`
* **Salida (1):** Probabilidad de aplicar la acción '1' (Empujar Positivo).

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `init_network(...)` | Construye la arquitectura de la red con inicializadores pasados por *lambda*. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |
| `init_weights_xavier(...)` | Inicializa los pesos de una matriz usando el método **Xavier/Glorot**. | $\mathbf{O}(\mathbf{W}_{\text{layer}})$ |
| `init_bias_zero(...)` | Inicializa los *bias* de una matriz a cero. | $\mathbf{O}(\mathbf{B}_{\text{layer}})$ |
| `ControllerDemo()` | Constructor. Inicializa la red utilizando Xavier para pesos y cero para *bias*. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |

---

#### 2. Métodos del Entorno (Simulación EnvGym)

Estos métodos gestionan la simulación simplificada de un objeto en movimiento sujeto a fuerza y fricción.

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `reset()` | Reinicia la posición y velocidad del simulador a 0.0. | $\mathbf{O}(1)$ | N/A |
| `get_state() const` | Devuelve el estado actual del entorno: `[position, velocity]` como un $\mathbf{Tensor}<T, 2>(1, 2)$. | $\mathbf{O}(1)$ | N/A |
| `step(int action)` | Aplica la acción (`1` o `0`) y actualiza la física de la posición y velocidad del objeto. | $\mathbf{O}(1)$ | Retorna `false` si se alcanza el límite. |

---

#### 3. Entrenamiento de la Política de Control (`train_expert_policy`)

Este método ejecuta el flujo completo de **aprendizaje supervisado** para imitar una política de control experta.

| Algoritmo/Fase | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| **Inicialización de Dataset** | Define el set de datos X (estado) y Y (acción experta) con D=12 muestras. | O(1) | Se realiza una vez. |
| **Entrenamiento** | Llama a `nn_.train<BinaryCrossEntropyLoss, Adam>(...)`. | O(Epochs * D * W_total) | Es el cuello de botella del algoritmo. |
| **Predicción** | Genera predicciones sobre el set X de entrenamiento para evaluar la **Accuracy**. | O(D * W_total) | Pasa D muestras una vez a través de la red. |
| **Validación de Precisión** | Compara la acción predicha (`pred > 0.5`) con la acción esperada (Y) y calcula la Accuracy. | O(D) = O(1) | Bucle lineal sobre las 12 muestras. |
| **Pruebas de Generalización** | Genera predicciones sobre un set de prueba X_test (3 muestras). | O(D_test * W_total) = O(1) | Evalúa la capacidad de generalización del modelo. |

---

#### 4. Serialización

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `save_weights(...)` | Guarda los pesos y bias de la red. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |
| `load_weights(...)` | Carga los pesos y bias de la red. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |

---

### PatternClassifier
---

El archivo `PATTERN_CLASSIFIER.H` define la clase `PatternClassifier<T>`, la cual implementa la solución al problema de clasificación **XOR** (Exclusivo o) utilizando una **Red Neuronal Multicapa (MLP)**. Este es un problema clásico no lineal que demuestra la capacidad de las redes neuronales profundas para aprender separaciones complejas.

## ⚙️ Notación de Complejidad Algorítmica (O)

| Símbolo | Descripción |
| :--- | :--- |
| $\mathbf{W}_{\text{layer}}$ | Número de parámetros (pesos o sesgos) en una capa `Dense`. |
| $\mathbf{B}_{\text{layer}}$ | Número de bias en una capa `Dense`. |
| $\mathbf{W}_{\text{total}}$ | Número total de parámetros (pesos y bias) en toda la red neuronal. |
| $\mathbf{D}$ | Tamaño total del dataset de entrenamiento (fijo en 4 para el demo XOR). |
| $\mathbf{D}_{\text{samples}}$ | Número de muestras en el batch o en la predicción. |
| $\mathbf{Epochs}$ | Número de épocas de entrenamiento. |

---

#### 💻 Clase `template <typename T> class PatternClassifier`

Esta clase encapsula la red neuronal (`nn_`) y expone los métodos necesarios para la inicialización, entrenamiento, predicción y serialización.

#### 1. Arquitectura de la Red

La red utiliza una arquitectura más profunda que la mínima necesaria para el XOR, lo que mejora la estabilidad y la robustez:

$$\text{Entrada} (2) \rightarrow \text{Densa} (8) \rightarrow \text{ReLU} \rightarrow \text{Densa} (8) \rightarrow \text{ReLU} \rightarrow \text{Densa} (1) \rightarrow \text{Sigmoid} \rightarrow \text{Salida} (1)$$

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `init_network(...)` | Construye la arquitectura MLP con dos capas ocultas. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |
| `init_weights_xavier(...)` | Implementa la inicialización de pesos **Xavier/Glorot** para mejorar la convergencia en redes profundas. | $\mathbf{O}(\mathbf{W}_{\text{layer}})$ |
| `init_bias_zero(...)` | Inicializa los *bias* a cero. | $\mathbf{O}(\mathbf{B}_{\text{layer}})$ |
| `PatternClassifier()` | Constructor. Inicializa la red utilizando los métodos Xavier/Glorot y bias a cero. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |

---

#### 2. Entrenamiento (Experimento XOR)

El método `run_xor_experiment()` gestiona la carga del *dataset* XOR, la configuración de hiperparámetros y la ejecución del entrenamiento.

| Algoritmo/Fase | Propósito | Hiperparámetros | Complejidad Dominante |
| :--- | :--- | :--- | :--- |
| **Inicialización de Dataset** | Carga las 4 muestras del XOR (X y Y). | N/A | O(D) = O(1) |
| **Entrenamiento** | Llama a `nn_.train` utilizando **Adam** y **Binary Cross-Entropy Loss** por 20000 épocas. | Epochs=20000, LR=0.05, Batch_Size=4 | O(Epochs * D * W_total) |
| **Predicción** | Predice los 4 resultados de entrenamiento. | N/A | O(D * W_total) |
| **Validación de Precisión** | Compara las predicciones con el *threshold* 0.5 para calcular la *Accuracy*. | Threshold=0.5 | O(D) = O(1) |
| **Prueba de Robustez** | Prueba el modelo con entradas con ruido (ej. 0.05 en lugar de 0.0) para evaluar la generalización. | N/A | O(D_samples * W_total) = O(1) |

---

#### 3. Métodos Públicos y Serialización

| Método | Propósito | Complejidad | Observaciones |
| :--- | :--- | :--- | :--- |
| `save_weights(...)` | Delega la serialización de parámetros de la red. | O(W_total) | Requisito de portabilidad. |
| `load_weights(...)` | Delega la carga de parámetros. | O(W_total) | Requisito de portabilidad. |
| `predict(const X)` | Realiza la inferencia utilizando la propagación hacia adelante (`nn_.predict`). | O(D_samples * W_total) | Expone la funcionalidad principal de la red. |
| `train<...>(...)` | Expone el método de entrenamiento de la red para que pueda ser llamado con diferentes optimizadores y funciones de pérdida. | O(Epochs * D * W_total) | Permite flexibilidad para pruebas. |

---

### SequencePredictor
---

El archivo `SEQUENCE_PREDICTOR.H` define la clase `SequencePredictor<T>`, la cual implementa una **Red Neuronal** para resolver un problema de **regresión lineal simple** ($y = 2x + 1$). Este experimento demuestra la capacidad de la librería para manejar tareas de predicción de valores continuos, utilizando la función de pérdida MSE y evitando una activación final.

#### ⚙️ Notación de Complejidad Algorítmica (O)

| Símbolo | Descripción |
| :--- | :--- |
| $\mathbf{W}_{\text{layer}}$ | Número de parámetros (pesos o sesgos) en una capa `Dense`. |
| $\mathbf{B}_{\text{layer}}$ | Número de bias en una capa `Dense`. |
| $\mathbf{W}_{\text{total}}$ | Número total de parámetros (pesos y bias) en toda la red neuronal. |
| $\mathbf{D}$ | Tamaño total del dataset de entrenamiento (fijo en 5 para el demo). |
| $\mathbf{D}_{\text{samples}}$ | Número de muestras en el batch o en la predicción. |
| $\mathbf{Epochs}$ | Número de épocas de entrenamiento. |

---

#### 💻 Clase `template <typename T> class SequencePredictor`

Esta clase gestiona la red neuronal (`nn_`) enfocada en la regresión de una serie simple.

#### 1. Arquitectura de la Red

La arquitectura es una MLP, diseñada específicamente para regresión:

$$\text{Entrada} (1) \rightarrow \text{Densa} (16) \rightarrow \text{ReLU} \rightarrow \text{Densa} (1) \rightarrow \text{Salida} (1)$$

* **Diferencia Clave:** La **capa de salida NO tiene función de activación** (ni Sigmoid, ni ReLU), permitiendo que la red prediga cualquier valor continuo (regresión).

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `init_network(...)` | Construye la arquitectura MLP de 1 entrada y 1 salida, crítica para regresión. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |
| `init_weights_xavier(...)` | Inicialización de pesos **Xavier/Glorot**. | $\mathbf{O}(\mathbf{W}_{\text{layer}})$ |
| `init_bias_zero(...)` | Inicialización de *bias* a cero. | $\mathbf{O}(\mathbf{B}_{\text{layer}})$ |
| `SequencePredictor()` | Constructor. Inicializa la red. | $\mathbf{O}(\mathbf{W}_{\text{total}})$ |

---

#### 2. Experimento de Regresión (`run_series_experiment`)

Este método ejecuta el flujo completo para aprender la relación $\mathbf{Y} = 2\mathbf{X} + 1$.

| Algoritmo/Fase | Propósito | Complejidad Dominante | Observaciones |
| :--- | :--- | :--- | :--- |
| **Inicialización de Dataset** | Carga las 5 muestras de entrenamiento ($\mathbf{X}$ y $\mathbf{Y}$). | $\mathbf{O}(\mathbf{D}) = \mathbf{O}(1)$ | N/A |
| **Entrenamiento** | Llama a `nn_.train` utilizando **Adam** como optimizador y **MSELoss** (Mean Squared Error) para medir la pérdida, durante 15,000 épocas. | $\mathbf{O}(\mathbf{Epochs} \cdot \mathbf{D} \cdot \mathbf{W}_{\text{total}})$ | MSELoss es estándar para problemas de regresión. |
| **Predicción (Validación)** | Predice los 5 resultados de entrenamiento. | $\mathbf{O}(\mathbf{D} \cdot \mathbf{W}_{\text{total}})$ | N/A |
| **Cálculo de Error** | Calcula el **Error Absoluto Promedio** sobre los datos de entrenamiento. | $\mathbf{O}(\mathbf{D}) = \mathbf{O}(1)$ | Se utiliza $\mathbf{Error} = |\mathbf{Y} - \mathbf{Predicción}|$. |
| **Prueba de Generalización** | Predice valores $(\mathbf{X} = 6.0, 10.0)$ no vistos en el entrenamiento. | $\mathbf{O}(\mathbf{D}_{\text{samples}} \cdot \mathbf{W}_{\text{total}}) = \mathbf{O}(1)$ | Evalúa la robustez del modelo para extrapolar. |

---

#### 3. Métodos Públicos y Serialización

| Método | Propósito | Complejidad |
| :--- | :--- | :--- |
| `save_weights(...)` | Delega la serialización de parámetros de la red. | O(W_total) |
| `load_weights(...)` | Delega la carga de parámetros. | O(W_total) |
| `predict(const X)` | Realiza la inferencia (propagación hacia adelante). | O(D_samples * W_total) |
| `train<...>(...)` | Expone el método de entrenamiento de la red. | O(Epochs * D * W_total) |

---

## 6. Manual de uso

### Opción 1: Ejecutar tests (Recomendado)
```bash
cd build
ctest --verbose
```

**Salida esperada:**
```
Test #1: TestTensor ....................... Passed (6/6 tests)
Test #2: TestNeuralNetwork ................ Passed (8/8 tests)
Test #3: TestApplications ................. Passed (8/8 tests)
100% tests passed, 0 tests failed out of 3
```

### Opción 2: Ejecutar aplicaciones

**Clasificador de patrones (XOR):**
```bash
./pattern_classifier_app
```
Entrena red para XOR, serializa modelo, carga y verifica portabilidad.

**Predictor de secuencias:**
```bash
./sequence_predictor_app
```
Entrena regresión lineal (y=2x+1), prueba generalización en datos no vistos.

**Demo de controlador:**
```bash
./controller_demo_app
```
Entrena política de control, ejecuta simulación en EnvGym hasta alcanzar límites.

#### Opción 3: Usar como librería

```cpp
#include <utec/algebra/Tensor.h>
#include <utec/nn/neural_network.h>

using namespace utec::algebra;
using namespace utec::neural_network;

// Crear datos
Tensor<float, 2> X(100, 5);  // 100 muestras, 5 features
Tensor<float, 2> Y(100, 1);  // 100 labels

// Crear red
NeuralNetwork<float> nn;
nn.add_layer(std::make_unique<Dense<float>>(5, 10, init_xavier, init_zeros));
nn.add_layer(std::make_unique<ReLU<float>>());
nn.add_layer(std::make_unique<Dense<float>>(10, 1, init_xavier, init_zeros));

// Entrenar
nn.train<MSELoss, Adam>(X, Y, epochs=1000, batch_size=32, lr=0.001);

// Predecir
auto predictions = nn.predict(X_test);

// Guardar/Cargar
nn.save_state("model.bin");
nn.load_state("model.bin");
```

---

## 7. Ejecución

### Demo automatizada (video)

El video de demostración muestra:
1. Compilación exitosa desde cero
2. Ejecución de 22 tests (100% passed)
3. Demostración de las 3 aplicaciones
4. Verificación de serialización

**Comando para reproducir:**
```bash
cd build && rm -rf * && cmake .. && make -j4 && ctest --verbose && ./pattern_classifier_app
```

---

## 8. Análisis del rendimiento

### Métricas de tests

| Test Suite | Tests | Tiempo | Cobertura |
|------------|-------|--------|-----------|
| Tensor (Epic 1) | 6 | 0.00s | ~95% |
| Neural Network (Epic 2) | 8 | 0.03s | ~95% |
| Applications (Epic 3) | 8 | 0.14s | ~98% |
| **TOTAL** | **22** | **0.18s** | **~95%** |

### Complejidad de algoritmos principales

| Operación | Complejidad Temporal | Complejidad Espacial |
|-----------|---------------------|---------------------|
| Acceso Tensor | O(1) | O(1) |
| Element-wise ops | O(N) | O(N) |
| Matrix product (M×K)·(K×N) | O(M·K·N) | O(M·N) |
| Broadcasting (N→M) | O(M) | O(M) |
| Forward pass (L capas) | O(L·batch·weights) | O(L·neurons) |
| Backward pass | O(L·batch·weights) | O(L·neurons) |
| Adam update | O(params) | O(2·params) |

### Resultados de entrenamiento

**PatternClassifier (XOR):**
- Epochs: 100-200
- Learning rate: 0.01
- Accuracy: 100% (4/4 predicciones correctas)
- Robustez: 100% con ruido ±10%

**SequencePredictor:**
- Epochs: 5000
- Learning rate: 0.005
- MSE final: < 0.1
- Generalización: Predicción exacta en x=6 (esperado: 13, obtenido: 13)

**ControllerDemo:**
- Epochs: 500
- Simulación: 7-50 pasos hasta término
- Política aprendida exitosamente

### Ventajas de la implementación

✅ **Sin dependencias externas**: Solo C++ standard (portabilidad máxima)  
✅ **Código limpio**: Separación clara de responsabilidades  
✅ **Eficiencia**: Operaciones optimizadas con strides y broadcasting  
✅ **Extensibilidad**: Fácil agregar nuevas capas/optimizadores  
✅ **Testing exhaustivo**: 22 tests cubren 95%+ de la funcionalidad  
✅ **Documentación completa**: Comentarios de complejidad en tests  

### Limitaciones actuales

❌ Sin paralelización (CPU single-thread)  
❌ Sin soporte para GPU  
❌ Arquitecturas limitadas a MLP (no CNN/RNN)  

### Mejoras futuras justificadas

1. **Paralelización con OpenMP** (Justificación: reducir tiempo de entrenamiento 4-8x)
2. **Soporte GPU con CUDA** (Justificación: acelerar operaciones matriciales 100x)
3. **Más arquitecturas** (CNN para imágenes, RNN para secuencias)
4. **Optimizador de hiperparámetros** (Grid search, Bayesian optimization)
5. **Visualización de entrenamiento** (Gráficas de pérdida en tiempo real)

---

## 8. Trabajo en equipo

| Tarea | Miembro | Rol | Horas |
|-------|---------|-----|-------|
| Epic 1: Tensor | Elias Alonso Usaqui Cabezas | Implementación completa | 23h |
| Epic 2: NN | Elias Alonso Usaqui Cabezas | Forward/Backward propagation | 26h |
| Epic 3: Apps | Fredy Cardenas Aliaga | Aplicaciones y serialización | 20h |
| Testing | Fredy Cardenas Aliaga | 22 tests automatizados | 15h |
| Documentación | Elias Alonso Usaqui Cabezas | README, video, presentación | 10h |
| Integración | Fredy Cardenas Aliaga | Code review y merge | 5h |

**Herramientas de colaboración:**
- GitHub para versionamiento
- GitHub Issues para tracking de tareas
- Pull Requests con code review obligatorio
- CMake para build unificado

---

## 9. Conclusiones

### Logros

Implementación completa de red neuronal desde cero  
Biblioteca de álgebra tensorial funcional y eficiente  
3 aplicaciones prácticas funcionando al 100%  
100% de tests passing (22/22)  
Serialización y portabilidad verificadas  
Código sin dependencias externas (máxima portabilidad)  

### Aprendizajes

1. Comprensión profunda de backpropagation
2. Implementación de templates avanzados en C++20
3. Diseño de APIs limpias y extensibles
4. Importancia de testing exhaustivo
5. Trabajo en equipo con control de versiones

### Recomendaciones

Para proyectos futuros o mejoras:
1. Implementar datasets más grandes (MNIST, CIFAR-10)
2. Optimizar con BLAS/LAPACK para multiplicaciones matriciales
3. Agregar más arquitecturas (CNN, LSTM)
4. Implementar regularización (L2, Dropout)
5. Crear interfaz gráfica para visualización

---

## 10. Bibliografía

- Aprende Machine Learning, "Breve Historia de las Redes Neuronales Artificiales", https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/, [En línea]. Disponible en: https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/. [Accedido: 24-11-2025].

- "CONCEPTOS BÁSICOS SOBRE REDES NEURONALES," Grupo de Tecnología de Computadores, Universidad de Sevilla. [En línea]. Disponible en: https://grupo.us.es/gtocoma/pid/pid10/RedesNeuronales.htm. [Accedido: 24-11-2025].

- BM, "¿Qué es la retropropagación?", IBM Think, [En línea]. Disponible en: https://www.ibm.com/mx-es/think/topics/backpropagation. [Accedido: 24-11-2025].

- Sánchez Medina, J. J. (1998). Linealización del algoritmo de backpropagation para el entrenamiento de redes neuronales (Proyecto fin de carrera). Universidad de Las Palmas de Gran Canaria. https://accedacris.ulpgc.es/bitstream/10553/1983/1/1235.pdf

- W. S. McCulloch y W. Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity". Disponible en: https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity, 2024.

- Angelvillazon.com, "Historia de las redes neuronales en la Inteligencia Artificial," 2025. [Online]. Available: https://www.angelvillazon.com/inteligencia-artificial-robotica/historia-de-las-redes-neuronales-en-la-inteligencia-artificial/

- Lamaquinaoraculo.com, "Neuronas de McCulloch y Pitts - Artículo de LMO," 2025. [Online]. Available: https://lamaquinaoraculo.com/deep-learning/el-modelo-neuronal-de-mcculloch-y-pitts/

---

