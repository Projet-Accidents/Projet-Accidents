ACCIDENTS ROUTIERS EN FRANCE

L’objectif de ce projet est d’essayer de prédire la gravité des accidents routiers en France en se basant sur les données historiques recensées en France entre 2005 et 2023.

Nous avons à notre disposition des fichiers excel regroupant des données recueillies par l'unité des forces de l'ordre qui est intervenue sur le lieu de l'accident, ainsi que la description des bases de données anuelles.

Les fichiers sont séparés en 4 rubriques:

La rubrique CARACTERISTIQUES qui décrit les circonstances générales de l’accident
La rubrique LIEUX qui décrit le lieu principal de l’accident même si celui-ci s’est déroulé à une intersection
La rubrique VEHICULES impliqués
La rubrique USAGERS impliqués

Après réflexion collective, nous décidons de créer un modèle dont l'objectif serait de permettre à un centre d'appel de secours de déterminer le niveau de gravité d'un accident qui vient de se produire, afin de déclencher le nombre et le type de secours adapté. La varaible cible est donc la gravité.
Nous définirons aussi la gravité de l’accident comme la gravité maximale constatée lors de cet accident.

Après l'agrégation et le nettoyage des différents fichiers, nous analyserons ces données, puis testerons des modèles de Machine Learning pour prédire au mieux les besoins de secours. 

Les datasets etant trop lourds (273226 lignes), ils ne sont pas stockés dans le Github. Voici cependant le lien pour y accéder

Caractéristiques de 2005 à 2023:

Lieux de 2005 à 2023:

Usagers de 2005 à 2023:

Véhicules de 2005 à 2023:

Concaténation caractéristiques de 2019 à 2023: 

Concaténation lieux de 2019 à 2023:

Concaténation usagers de 2019 à 2023:

Concaténation Véhicules de 2019 à 2023:

Dataframe final:

Dataframe machine learning: 


