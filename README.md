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

Caractéristiques de 2005 à 2023: https://drive.google.com/drive/folders/1sZes2NkcsUA0cxTwFXgmMNBedvp5SjGi?usp=sharing

Lieux de 2005 à 2023: https://drive.google.com/drive/folders/1uOsHWUWYnw-0CiAcg3vh5zxfm7BfzrtX?usp=sharing

Usagers de 2005 à 2023: https://drive.google.com/drive/folders/1h6knGV_1aji_GENWUOuia96dxAPA4kVf?usp=sharing

Véhicules de 2005 à 2023: https://drive.google.com/drive/folders/1QyoQubXr21dQCzDSkAPSKHB1t0q5nefc?usp=sharing

all_data_caractéristiques de 2019 à 2023: https://drive.google.com/file/d/16-ccVuQu2EnpLzghV223rXGyst-2SjQu/view?usp=sharing

all_data_lieux de 2019 à 2023: https://drive.google.com/file/d/155xpqFW4bpgoN_d15qGUnmwES9g6983-/view?usp=sharing

all_data_usagers de 2019 à 2023: https://drive.google.com/file/d/1OegcguvHBBTn6kCltXhkVdhujvxRvUZ-/view?usp=sharing

all_data__véhicules 2019 à 2023: https://drive.google.com/file/d/1RbaJlB0pRC5mhFy6cnTk303jbBDjpnMf/view?usp=sharing

Dataframe final df_total_final: https://drive.google.com/file/d/1MoTCvFEn78pcS05pssfep4LaNM-weB-5/view?usp=sharing

Dataframe machine learning df_machine_learning : https://drive.google.com/file/d/1JLQIGfRc9HLQqccd5PhnHjt7BuE4YjK-/view?usp=sharing



