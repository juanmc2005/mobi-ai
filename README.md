# Instructions de préparation
- Télécharger le jeu de données depuis https://www.kaggle.com/pavansanagapati/urban-sound-classification
- Décompresser le fichier `urban-sound-classification.zip`
- Dans le répertoire `urban-sound-classification`, décompresser le fichier `train.zip`
- Placer le dossier `train` généré dans le répertoire principal du projet

# Instructions de lecture des données
La lecture et préparation du dataset peut être exécuté sans l'entraînement d'aucun modèle. Pour obtenir un objet DataSet avec les partitions de X et Y pour l'entraînement et le test, il suffit d'exécuter le suivant dans la console Julia :
- `cd("<project dir>")`
- `include("model-commons.jl")`
- `dataset = create_data_splits(n, m)` pour que les spectrogrammes aient une dimension de nxm, les valeurs par défaut sont n=150 et m=100

L'objet DataSet est définit de la façon suivante :

```julia
struct DataSet
  X_train::Array{Float32, 2}
  Y_train::Flux.OneHotMatrix
  X_test::Array{Float32, 2}
  Y_test::Flux.OneHotMatrix
end
```

# Instructions d'entraînement des modèles

### ANN
- `cd("<project dir>")`
- `include("ann.jl")`
- `dataset = create_data_splits(n, m)` pour des valeurs de n et m
- `train_ann(dataset)`

Définition de la fonction `train_ann` pour d'autres configurations :

```julia
function train_ann(d::DataSet; hidden=750, epochs=epochs, eta=eta, plotloss=true)
```

### CNN
- `cd("<project dir>")`
- `include("cnn.jl")`

L'inclusion du fichier exécute automatiquement la création du dataset, puis l'entraînement du réseau avec les paramètres optimaux.
