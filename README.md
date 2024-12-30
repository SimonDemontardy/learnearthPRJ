# learnearthPRJ
application of ia tools on an earthquake dataset following a problematic. 

Titre: Prédiction de la population impacté par un séisme partir de données sismiques
Résumé: Le nombre de tremblements de terre par an est considérable: en ce moment
même le chiffre est de 974 530 depuis le 1er janvier 2024. Néanmoins la classification de
ces tremblements de terre a son importance car tous ne sont pas à la même échelle. En
effet, sur les 1 millions de tremblements annuels, seuls 1000 d’entre eux sont capables de
faire des dégâts. Nous nous sommes donc intéressés à un dataset contenant uniquement
les tremblements significatifs (magnitude ≥ 5.5). Au cours de nos recherches nous avons
remarqué une colonne type d’alerte. Initialement nous voulions donc entraîner une
intelligence artificielle qui prédirait l’alerte en fonction des autres données sismiques.
Cependant, trop peu de données contiennent cette information (700).
À travers l'utilisation d'algorithmes de classification supervisée, nous tenterons donc de
catégoriser les séismes en fonction de l'ampleur de leur impact sur une population alentour,
en introduisant nos propres classes d'alerte telles que "bénin", "problématique", et
"dangereux". Avant cela, nous commencerons par une analyse de clustering pour explorer
les relations entre les caractéristiques des séismes, telles que la magnitude et la
profondeur… Cela nous permettra de mieux comprendre les groupes de séismes en
fonction de leur impact potentiel, afin d'affiner la définition de l'impact avant de construire un
modèle de classification supervisée.Notre nouveau jeu de données recense près de 23 000
tremblements survenus entre 1965 et 2016.
Définition du problème:
- Problématique: Comment prédire l'impact d'un séisme sur une population en fonction
de ses caractéristiques sismiques ?
Nous souhaiterions réaliser comme approche préliminaire un clustering à partir des
données physiques sismiques afin de mieux définir notre notion d’impact. Puis nous
passerons à la classification supervisée afin de prédire le possible impact du séisme.
Description méthodologique:
- Nettoyage des données :
→ Enlever les exemples ayant des données manquantes
→ Enlever des variables inutiles (comme ‘id’, ‘source’, ‘location source’...)
- Analyse des données :
→ Ensuite, nous définissons le rayon d'étude à partir de l'épicentre, et la population
susceptible d'être impactée. Nous choisissons un rayon de 50 km, car c'est la
distance moyenne à laquelle un séisme modéré peut causer des dégâts à la
population. Ce rayon est également considéré comme le rayon minimum pour les
séismes de plus forte magnitude.
→ Nous allons ensuite créer un objet geopanda qui correspond à cette zone, et
l’utilisation de l’api Overpass API nous permet d'accéder à une estimation de
population dans la zone à partir des données de OpenStreetMap.
→ Ainsi on obtient la densité de population touchée par le séisme à partir de la
longitude et la latitude
- Nous pouvons par la suite réaliser notre analyse préliminaire par clustering en
utilisant la densité touchée, la magnitude et la profondeur du séisme. Puis nous
utiliserons les mêmes données pour réaliser la classification supervisée.
- Évaluation des résultats, avec l’accuracy, matrices de confusions, AUC…
