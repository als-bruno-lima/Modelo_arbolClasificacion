Modelo de clasificacion
 
Este proyecto emplea el algoritmo de Decision Tree para clasificar como "Diabetes" o "No diabetes" una muestra de pacientes. Basandose en un dataset con datos de Glucosa, Presion sanguinea, Espesor de piel, Insulina, IMC y Edad de cada paciente. El modelo está desarrollado en Python usando la librería scikit-learn, pandas y matplotlib. El proyecto crea un modelo de arbol y se le establece un hiper parametro de maximo de profundidad de 5, esto para evitar que sea muy grande en caso de no especificar una profundidad fija. Los datos son particionados en dos partes, una para entrenamiento y otra para realizar las pruebas. 
 
# Dependencias
 
Para instalar las dependencias del proyecto ejecutar: 
 
```terminal
python -m pip install -r ./librerias.txt

# Ejecutar proyecto
Para correr el proyecto ejecutar:
``` terminal
python modelo.py
