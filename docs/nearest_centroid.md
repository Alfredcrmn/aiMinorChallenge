# Clasificador de centroide más cercano

Implementación individual en `nearest_centroid_from_scratch.py`, ejecutable con
Python 3.10 o superior. El modelo y sus pruebas no requieren NumPy, pandas ni
bibliotecas de ML. El CSV compartido no se modifica.

## Ejecución desde la raíz del repositorio

```bash
./.venv/bin/python nearest_centroid_from_scratch.py
./.venv/bin/python nearest_centroid_from_scratch.py --examples 5 --show-centroids
```

También se puede utilizar cualquier intérprete Python 3.10+ en lugar del entorno
virtual. La ruta predeterminada del dataset se resuelve respecto al archivo del
programa, no al directorio de ejecución. `--csv /ruta/features.csv` permite elegir
otro archivo con el mismo esquema.

- `--examples N`: muestra las primeras N predicciones de prueba. Por defecto son
  10; 0 oculta los ejemplos. Las métricas siempre se calculan sobre todas las
  ventanas de prueba, independientemente de N. Si N excede las ventanas
  disponibles, muestra todas.
- `--show-centroids`: muestra las cantidades por clase y las primeras tres
  coordenadas de cada centroide. El modelo conserva las 72 coordenadas completas.

El flujo es lectura del CSV → ajuste del estandarizador con `train` → promedio de
las features estandarizadas por clase → predicciones sobre ventanas de `test`.
`fit_nearest_centroid()` no consulta valores ni etiquetas de prueba.

`predict(model, features)` recibe un vector **sin estandarizar**, en el orden de
features del CSV. Aplica el estandarizador almacenado una sola vez y selecciona
la menor distancia euclidiana al cuadrado: `sum((z_j - centroide_j) ** 2)`.
Los empates exactos se resuelven por el menor identificador de clase. Las
distancias no son probabilidades ni porcentajes de confianza.

La salida indica grabación, ventana, clase real, clase predicha y distancia al
centroide elegido. Las primeras ventanas siguen el orden del CSV y no son una
muestra representativa de todas las clases. No se guardan modelos/resultados en disco.

## Interpretación de la salida

- `Accuracy`: porcentaje de ventanas de prueba correctamente clasificadas:
  `100 * aciertos / total_de_ventanas_test`. No se calcula sobre entrenamiento
  ni únicamente sobre los ejemplos impresos.
- `Correct predictions`: aciertos y total de ventanas evaluadas.
- `Precision (macro)`: para cada clase, qué proporción de sus predicciones fue
  correcta: `TP / (TP + FP)`. Después promedia las 16 clases con igual peso.
- `Recall (macro)`: para cada clase, qué proporción de sus ventanas reales se
  identificó correctamente: `TP / (TP + FN)`. Después promedia las 16 clases.
- `F1 score (macro)`: promedio de los F1 individuales por clase, calculados como
  `2 * TP / (2 * TP + FP + FN)`. No es el F1 calculado a partir de los promedios
  macro de precision y recall.
- `actual` es la clase verdadera; `predicted` es la clase elegida. Si coinciden,
  la predicción de esa ventana es correcta.
- `squared_distance`: distancia al cuadrado al centroide elegido en el espacio
  estandarizado. Una distancia menor indica mayor cercanía; no expresa confianza
  ni tiene un umbral universal de acierto.

La accuracy es global por ventana, no por grabación ni por paciente. Las clases
con más ventanas pesan más y las ventanas de una misma grabación están
relacionadas. No demuestra el desempeño por movimiento ni generalización a
pacientes nuevos. Precision, recall y F1 macro dan igual peso a cada clase,
independientemente de su cantidad de ventanas. Cuando un denominador es cero,
la métrica de esa clase vale cero; todas las clases configuradas se incluyen en
el promedio. Se calculan manualmente y se imprimen como porcentajes.
No se muestran todavía una matriz de confusión ni un desglose por clase.

## Resultados obtenidos
- Entrenamiento: 29,544 ventanas, utilizadas para ajustar el estandarizador y
  calcular los 16 centroides de 72 dimensiones.
- Prueba: 7,384 ventanas, sin participación en el ajuste del modelo.
- Predicciones correctas: 3,603 de 7,384.

| Métrica | Resultado | Forma de cálculo |
| --- | ---: | --- |
| Accuracy | 48.79% | Porcentaje global de ventanas correctamente clasificadas. |
| Precision (macro) | 50.86% | Promedio aritmético de las precisiones de las 16 clases. |
| Recall (macro) | 48.33% | Promedio aritmético de los recalls de las 16 clases. |
| F1 score (macro) | 47.99% | Promedio aritmético de los F1 de las 16 clases. |

**Precision, recall y F1 score se calculan mediante promedios macro:** primero
se obtiene cada métrica por movimiento y después se suman sus 16 valores y se
divide entre 16. Todos los movimientos tienen el mismo peso, aunque tengan
distintas cantidades de ventanas. En particular, el F1 macro promedia los F1
individuales; no se obtiene combinando la precision macro y el recall macro.

Los porcentajes se muestran redondeados a dos decimales. La accuracy indica que
el modelo reconoce correctamente aproximadamente 49 de cada 100 ventanas de
prueba. Estos resultados describen la versión actual, no garantizan el mismo
desempeño con pacientes nuevos ni sustituyen el análisis por clase.

Una limitación conocida es que el dataset derivado incluye ventanas parcial o
completamente ubicadas en el relleno final con ceros de las grabaciones
originales. Los resultados anteriores incluyen todas esas ventanas: todavía no
se ha aplicado una corrección del relleno ni se han excluido casos de prueba.

Para reproducir el resumen sin imprimir ejemplos individuales:

```bash
./.venv/bin/python nearest_centroid_from_scratch.py --examples 0
```

## Pruebas

Suite completa, incluyendo preparación compartida (requiere NumPy):

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Pruebas del modelo individual sin cargar bibliotecas externas:

```bash
./.venv/bin/python -S -m unittest discover -s tests -p 'test_centroid_*.py' -v
./.venv/bin/python -S -m unittest discover -s tests -p 'test_standardization.py' -v
```

Se verifican lectura, ausencia de influencia de `test`, estandarización, features
constantes, centroides, distancias conocidas, empates, entradas inválidas y la
ejecución real por consola desde otro directorio.
