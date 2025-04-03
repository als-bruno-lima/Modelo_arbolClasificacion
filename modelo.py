import pandas as pd
#lee dataset de pacientes
df = pd.read_csv('./diabetes.csv')
df

df.info()
df.describe()

#ver distribucion de categorias
df['Resultado'].value_counts()

#crea set de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1]
Y=df.iloc[:,-1]

x_train, y_test, y_train, x_test = train_test_split(X,Y,test_size=0.3,random_state=123,stratify=Y)


print("Tamaño de set de entrenamiento:", x_train.shape, y_train.shape)
print("Tamaño de set de prueba:", x_test.shape, y_test.shape)

print("Tamaño de set de categorias de dataset original: ", Y.value_counts(normalize=True))
print("Tamaño de set de categorias de dataset de entrenamiento: ", y_train.value_counts(normalize=True))
print("Tamaño de set de categorias de dataset de prueba: ", y_test.value_counts(normalize=True))

#creamos arbol de clasificacion y lo entrenamos.
from sklearn.tree import DecisionTreeClassifier

#creamos instancia del arbol
arbol = DecisionTreeClassifier(max_depth=5, min_samples_leaf=4,random_state=123)
arbol.fit(x_train,y_train)


#graficar el arbol
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(30,15))
plot_tree(arbol, feature_names=x_train.columns, class_names=["No diabetes","Diabetes"], filled=True,rounded=True,);

#guardar el arbol graficado como PNG
plt.savefig("Arbol_clasificacion.png",dpi=700)

print("profundidad del arbol: ", arbol.get_depth())
print("cantidad de hojas: ", arbol.get_n_leaves())

#validacion de datos
from sklearn.model_selection import StratifiedKFold, cross_val_score

#creo particiones estratificadas
skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
scores=cross_val_score(arbol, x_train, y_train, cv=skf, scoring="f1_weighted")

#realiza validacion cruzada

print("exactitud de entrenamiento: ", scores.mean())

#arbol con:
#max_depth = 5
#min_samples_leaf=4
arbol=DecisionTreeClassifier(max_depth=5,min_samples_leaf=4,random_state=123)

#entreno el arbol
arbol.fit(x_train,y_train)

#puntaje del set de prueba
score_test = arbol.score(y_test, x_test)
print("exactitud de prueba: ", score_test)




