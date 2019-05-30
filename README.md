# Clasificación de patrones

### Supervisado
+ LDA

### No supervisado
+ Kmeans
+ EM

### Análisis del trabajo 

El caso de entrenamiento supervisado es poco común, dado que el problema de clasificar manualmente las muestras, o mismo generarlas, puede ser muy dificultoso aunque de disponer de dichas muestras, el entrenamiento es muy bueno.

Comúnmente en _machine learning_ se tiene un gran volumen de información pero la misma no se encuentra clasificada. Por lo tanto las técnicas de _kmeans_ y _EM_ son de gran importancia.

En particular el algoritmo _EM_ está respaldado por una teoría mientras que _kmeans_ tiene un enfoque iterativo dado que es un método no paramétrico de hacer _clustering_. 

En conclusión si los _clusters_ están bien separados, cualquier método funciona. Sin embargo ante la aparición de _outliers_ y _clusters_ cercanos, el método de _kmeans_ es el mejor dado que se genera una decisión tipo **hard** asignandole un _cluster_ a los _outliers_ mientras que en _EM_ con el tipo de decisión **soft** no se le asignaría ningun _cluster_ dado que las responsabilidades serán nulas de todas las clases.
