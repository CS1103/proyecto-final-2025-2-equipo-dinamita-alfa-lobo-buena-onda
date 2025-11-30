# Proyecto Final 2025-2: Neural Network Application
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación completa de una red neuronal multicapa desde cero en C++20, incluyendo una biblioteca genérica de álgebra tensorial y aplicaciones prácticas de clasificación, predicción de secuencias y control.

---

## Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Investigación teórica](#1-investigación-teórica)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Diseño e implementación](#2-diseño-e-implementación)
6. [Ejecución](#3-ejecución)
7. [Análisis del rendimiento](#4-análisis-del-rendimiento)
8. [Trabajo en equipo](#5-trabajo-en-equipo)
9. [Conclusiones](#6-conclusiones)
10. [Bibliografía](#7-bibliografía)

---

## Datos generales

* **Tema**: Redes Neuronales en AI
* **Grupo**: `Equipo dinamita alfa lobo buena onda`
* **Integrantes**:
  * Elias Alonso Usaqui Cabezas – 202420064 (Responsable de investigación teórica)
  * Elias Alonso Usaqui Cabezas – 202420064 (Desarrollo de la arquitectura)
  * Fredy Alexander Cardenas Aliaga – 202420013 (Implementación del modelo)
  * Fredy Alexander Cardenas Aliaga – 202420013 (Pruebas y benchmarking)
  * Elias Alonso Usaqui Cabaezas – 202420064 (Documentación y demo)

---

## Requisitos e instalación

### Requisitos del sistema

* **Sistema Operativo**: Linux, macOS, o Windows con MinGW
* **Compilador**: 
  - GCC 11.0+ (Linux/MinGW)
  - Clang 13.0+ (macOS)
  - MSVC 19.30+ (Visual Studio 2022)
* **Herramientas**: CMake 3.16+
* **Estándar**: C++20
* **Dependencias externas**: **NINGUNA** (solo librería estándar de C++)

### Instalación de herramientas

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

### Instalación del proyecto (3 pasos)

```bash
# 1. Clonar repositorio
git clone https://github.com/TU_USUARIO/TU_REPO.git
cd pong_ai

# 2. Configurar y compilar
mkdir build && cd build
cmake ..
make -j4

# 3. Verificar instalación
ctest
```

**✅ Si ves "100% tests passed, 0 tests failed out of 3", la instalación fue exitosa.**

### Solución de problemas

| Error | Solución |
|-------|----------|
| "CMake not found" | Instalar CMake según tu sistema (ver arriba) |
| "C++20 not supported" | Actualizar compilador a GCC 11+ o Clang 13+ |
| "No such file: Tensor.h" | Ejecutar desde el directorio `build/` |

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

## Estructura del proyecto

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


### Álgebra Tensorial

Implementación de `Tensor<T, Rank>` que soporta:
- Operaciones element-wise (suma, resta, multiplicación)
- Broadcasting (estilo NumPy)
- Multiplicación matricial
- Transposición

---

## 2. Diseño e implementación

### 2.1 Arquitectura de la solución

**Patrones de diseño utilizados:**

1. **Template Method Pattern**: En `NeuralNetwork::train<LossFunc, Optimizer>()`
2. **Strategy Pattern**: Funciones de pérdida y optimizadores intercambiables
3. **Interface Segregation**: `ILayer<T>`, `IOptimizer<T>` para polimorfismo
4. **Generic Programming**: Todo parametrizado con templates (`T` para tipo, `Rank` para dimensión)

**Paradigmas:**
- **POO**: Herencia (`Dense : public ILayer`), encapsulación, polimorfismo
- **Genérico**: Templates para reutilización (`Tensor<float, 2>`, `Tensor<double, 3>`)
- **Funcional**: Lambdas para inicialización de pesos

### 2.2 Componentes principales

#### Epic 1: Tensor<T, Rank>
```cpp
// Ejemplo de uso
Tensor<float, 2> matrix(3, 4);  // Matriz 3x4
matrix(1, 2) = 5.0;              // Acceso variádico
auto transposed = transpose_2d(matrix);
auto result = matrix_product(A, B);  // Multiplicación matricial
```

**Características:**
- Almacenamiento eficiente con `std::vector<T>`
- Cálculo de strides para acceso O(1)
- Broadcasting automático
- Soporte para operaciones batch

#### Epic 2: Red Neuronal
```cpp
NeuralNetwork<float> nn;
nn.add_layer(std::make_unique<Dense<float>>(2, 4, init_xavier, init_zeros));
nn.add_layer(std::make_unique<ReLU<float>>());
nn.add_layer(std::make_unique<Dense<float>>(4, 1, init_xavier, init_zeros));

nn.train<BinaryCrossEntropyLoss, Adam>(X, Y, epochs=1000, batch_size=4, lr=0.01);
```

**Forward Propagation (O(L·N·M)):**
```
Para cada capa l de 1 a L:
    Z[l] = W[l] · A[l-1] + b[l]
    A[l] = activation(Z[l])
```

**Backward Propagation (O(L·N·M)):**
```
Para cada capa l de L a 1:
    dZ[l] = dA[l] ⊙ activation'(Z[l])
    dW[l] = dZ[l] · A[l-1]^T
    db[l] = sum(dZ[l])
    dA[l-1] = W[l]^T · dZ[l]
```

#### Epic 3: Aplicaciones

1. **PatternClassifier**: Resuelve XOR (problema no linealmente separable)
2. **SequencePredictor**: Aprende patrones aritméticos (y = 2x + 1)
3. **ControllerDemo**: Política de control basada en estado (posición, velocidad)

### 2.3 Manual de uso

#### Opción 1: Ejecutar tests (Recomendado)
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

#### Opción 2: Ejecutar aplicaciones

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

## 3. Ejecución

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

## 4. Análisis del rendimiento

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

## 5. Trabajo en equipo

| Tarea | Miembro | Rol | Horas |
|-------|---------|-----|-------|
| Epic 1: Tensor | Fredy Cardenas Aliaga | Implementación completa | 20h |
| Epic 2: NN | Fredy Cardenas Aliaga | Forward/Backward propagation | 25h |
| Epic 3: Apps | Fredy Cardenas Aliaga | Aplicaciones y serialización | 20h |
| Testing | Fredy Cardenas Aliaga | 22 tests automatizados | 15h |
| Documentación | Fredy Cardenas Aliaga | README, video, presentación | 10h |
| Integración | Fredy Cardenas Aliaga | Code review y merge | 5h |

**Herramientas de colaboración:**
- GitHub para versionamiento
- GitHub Issues para tracking de tareas
- Pull Requests con code review obligatorio
- CMake para build unificado

---

## 6. Conclusiones

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

## 7. Bibliografía

- Aprende Machine Learning, "Breve Historia de las Redes Neuronales Artificiales", https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/, [En línea]. Disponible en: https://www.aprendemachinelearning.com/breve-historia-de-las-redes-neuronales-artificiales/. [Accedido: 24-11-2025].

- "CONCEPTOS BÁSICOS SOBRE REDES NEURONALES," Grupo de Tecnología de Computadores, Universidad de Sevilla. [En línea]. Disponible en: https://grupo.us.es/gtocoma/pid/pid10/RedesNeuronales.htm. [Accedido: 24-11-2025].

- BM, "¿Qué es la retropropagación?", IBM Think, [En línea]. Disponible en: https://www.ibm.com/mx-es/think/topics/backpropagation. [Accedido: 24-11-2025].

- Sánchez Medina, J. J. (1998). Linealización del algoritmo de backpropagation para el entrenamiento de redes neuronales (Proyecto fin de carrera). Universidad de Las Palmas de Gran Canaria. https://accedacris.ulpgc.es/bitstream/10553/1983/1/1235.pdf

- W. S. McCulloch y W. Pitts, "A Logical Calculus of the Ideas Immanent in Nervous Activity". Disponible en: https://en.wikipedia.org/wiki/A_Logical_Calculus_of_the_Ideas_Immanent_in_Nervous_Activity, 2024.

- Angelvillazon.com, "Historia de las redes neuronales en la Inteligencia Artificial," 2025. [Online]. Available: https://www.angelvillazon.com/inteligencia-artificial-robotica/historia-de-las-redes-neuronales-en-la-inteligencia-artificial/

- Lamaquinaoraculo.com, "Neuronas de McCulloch y Pitts - Artículo de LMO," 2025. [Online]. Available: https://lamaquinaoraculo.com/deep-learning/el-modelo-neuronal-de-mcculloch-y-pitts/

---

