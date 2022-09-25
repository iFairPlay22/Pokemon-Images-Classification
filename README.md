# Deep learning classification

Création d'un réseau de Deep Learning permettant de classifier des images. En clair, le réseau permet de détecter à quelle classe appartient une image, suite à un entrainement de type supervisé

## Exemple de classification utilisé

L'exemple actuel utilise un dataset associant des images contenant un pokémon à son nom. Ainsi, suite à un entrainement, l'intelligence artificielle est capable de détecter le nom d'un pokémon, à partir d'une photo de celui-ci.

## Système générique

Le système a été créé dans l'objectif d'être générique, et ainsi de pouvoir fonctionner avec différents datasets. Il est ainsi possible de prédire autre chose que des noms de pokémons... Afin de tester avec un dataset personnalisé, il suffit juste de déployer le datasets concerné dans les répertoires suivants :

> `datasets/tv_dataset` pour le training & validation ;

> `datasets/t_dataset` pour les tests ;

Les fichiers seront automatiquement classifiés en fonction de leur organisation en sous répertoire. Le nom de sous répertoire correspondra ainsi au nom de la classe, et les fichiers y étant inclus à des exemples de fichier appartenant à cette classe.