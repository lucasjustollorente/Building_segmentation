## Building Footprints - Segmentación de Imágenes Satelitales
Este repositorio contiene código para la segmentación de imágenes satelitales, donde cada píxel se clasifica en una de las dos categorías posibles: fondo (background) o edificio (building). Se emplea la red neuronal UNet y se utiliza la biblioteca TensorFlow/Keras para el desarrollo.

Archivos
UnetTensorflow.ipynb
UnetTensorflowflujo2.ipynb
Características
Clases de Etiquetas:
Fondo (Background): Representado por 0 ((r,g,b)=(0,0,0)).
Edificio (Building): Representado por 1 ((r,g,b)=(255,255,255)).
Descripción de los Flujos de Entrenamiento
Se han realizado dos flujos completos de entrenamiento y pruebas utilizando la arquitectura UNet con diferentes configuraciones y preprocesamientos de imágenes.

Fichero UnetTensorflow.ipynb
En este flujo de trabajo se realiza el preprocesamiento de las imágenes, estableciendo un tamaño de 224x224 píxeles. Se emplea una arquitectura UNet sin capas de dropout, con 40 épocas de entrenamiento y un tamaño de lote (batch size) de 16. El modelo se entrena y se valida utilizando datos de prueba y sus respectivas máscaras. La precisión (accuracy) obtenida en el conjunto de prueba es del 89%.

Modelo Entrenado: modeldef.h5

Fichero UnetTensorflowflujo2.ipynb
En este segundo flujo de trabajo, se añade un hiperparámetro adicional durante el preprocesamiento de las imágenes: order=0. Este parámetro se utiliza para reducir la resolución de los píxeles en las imágenes. Se mantiene la misma arquitectura de red UNet, pero se reduce el número de épocas a 30 y el tamaño de lote a 32. La precisión obtenida en el conjunto de prueba es del 83%. Se menciona que, con más tiempo y recursos, se podría haber utilizado una red más profunda para mejorar el modelo.

Modelo Entrenado: modelflujo2.h5

Resumen
Ambos flujos de trabajo proporcionan resultados aceptables en términos de precisión en la segmentación de imágenes satelitales. El uso de la red neuronal UNet ha demostrado ser efectivo para este tipo de tareas. Sin embargo, la configuración específica de la red y los hiperparámetros pueden influir en el rendimiento del modelo. En futuras iteraciones, se podría explorar la optimización de la arquitectura de la red y la selección de hiperparámetros para mejorar aún más la precisión del modelo.



Fichero ResnetUnetmodel.ipynb

Se implementaron las clases Dataset y DataLoader para facilitar el preprocesamiento de imágenes durante el entrenamiento de un modelo de segmentación. Se introdujeron nuevas métricas, como Intersection over Union (IoU), para evaluar el rendimiento del modelo. Se utilizó una arquitectura UNet que emplea un "backbone" como encoder. El backbone en este caso se cargó con pesos preentrenados de ImageNet, lo que significa que el modelo aprovechó las representaciones aprendidas por una red neuronal entrenada en un conjunto de datos grande como ImageNet. Esta estrategia puede mejorar el rendimiento y la convergencia del modelo para la tarea específica de segmentación de imágenes.

Fichero ResnetUnetmodel+preprocess.ipynb

Se extendió el flujo de trabajo de la pasada semana al incorporar un paso adicional de preprocesamiento. Este paso eliminó las imágenes con espacios en blanco para reducir el ruido en los datos utilizados para el entrenamiento y la validación. Esta mejora condujo a una notable mejora en los resultados de validación, especialmente en los límites de los edificios en las imágenes segmentadas.

Conclusiones
La principal diferencia entre utilizar un backbone preentrenado como ResNet50 con encoder_weights de ImageNet y entrenar una red UNet desde cero radica en la capacidad de transferencia de aprendizaje. Al utilizar un backbone preentrenado, el modelo puede beneficiarse de las representaciones aprendidas en tareas previas, lo que a menudo conduce a una mejor generalización y un rendimiento más sólido en comparación con iniciar el entrenamiento desde cero, donde las representaciones se aprenden desde datos específicos de la tarea actual.

