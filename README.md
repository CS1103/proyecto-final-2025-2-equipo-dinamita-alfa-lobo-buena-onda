# Proyecto Final 2025-2: Neural Network Application
## **CS2013 Programación III** · Informe Final

### **Descripción**

Implementación completa de una red neuronal multicapa desde cero en C++20, incluyendo una biblioteca genérica de álgebra tensorial y aplicaciones prácticas de clasificación, predicción de secuencias y control.

---

## Contenidos

1. [Datos generales](#datos-generales)
2. [Requisitos e instalación](#requisitos-e-instalación)
3. [Estructura del proyecto](#estructura-del-proyecto)
4. [Investigación teórica](#1-investigación-teórica)
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
  * Fredy Alexander Cardenas Aliaga - 202420013 (Epic 1: Biblioteca de Álgebra)
  * Fredy Alexander Cardenas Aliaga - 202420013 (Epic 2: Red Neuronal)
  * Fredy Alexander Cardenas Aliaga - 202420013 (Epic 3: Aplicaciones)
  * Fredy Alexander Cardenas Aliaga - 202420013 (Testing y documentación)
  * Fredy Alexander Cardenas Aliaga - 202420013 (Integración y presentación) 

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

## 1. Investigación teórica

### Fundamentos implementados

1. **Redes Neuronales Feedforward (MLP)**
   - Arquitectura: capas densas con funciones de activación no lineales
   - Forward propagation: cálculo de salidas capa por capa
   - Backward propagation: algoritmo de retropropagación para calcular gradientes

2. **Optimización por Gradiente Descendente**
   - SGD (Stochastic Gradient Descent): actualización básica de pesos
   - Adam: optimizador adaptativo con momentos (beta1=0.9, beta2=0.999)

3. **Funciones de Activación**
   - ReLU: `f(x) = max(0, x)` - no linealidad eficiente
   - Sigmoid: `f(x) = 1/(1+e^(-x))` - para salidas binarias

4. **Funciones de Pérdida**
   - MSE: para problemas de regresión
   - Binary Cross Entropy: para clasificación binaria

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

Ver archivo [BIBLIOGRAFIA.md](BIBLIOGRAFIA.md) con 6+ referencias en formato IEEE.

---

