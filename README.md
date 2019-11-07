# FoodRecognitionModel

W tym repozytorium zamieszczam wyniki mojego pierwszego projektu związanego z deep learningiem - model klasyfikujący zdrowe i niezdrowe jedzenie.

## Dataset

W projekcie skorzystałem z datasetu [Food-101](https://www.kaggle.com/kmader/food41), wybierając z niego 20 klas, które później podzieliłem na zdrowe i niezdrowe:

- hamburgers        &#127828;
- donuts            &#127828;
- waffles           &#127828;
- pizza             &#127828;
- red_velvet_cake   &#127828;
- french_fries      &#127828;
- chocolate_mousse  &#127828;
- eggs_benedict     &#127828;
- chicken_wings     &#127828;
- hot_dog           &#127828;
- sushi             &#127823;
- curry             &#127823;
- greek_salad       &#127823;
- seaweed_salad     &#127823;
- grilled_salomon   &#127823;
- edamame           &#127823;
- guacamole         &#127823;
- falafel           &#127823;
- escargots         &#127823;
- lobster_bisque    &#127823;

Każda klasa zawiera po 1000 obrazków.  
Tak skonstruowany dataset podzieliłem na validation data (po +/-100 obrazków na klasę) i na train data (po +/-900 obrazków na klasę).  
Obrazy skalowałem do rozmiaru 128x128.  

## Klasyfikacja obrazów z 20 klas

Przed podziałem na 2 główne kategorie, najpierw wytrenowałem model na domyślnych 20 kategoriach.
Jako podstawę modelu wziąłem VGG16

```python
model = applications.VGG16(include_top=False, weights='imagenet')
```

wygenerowałem bottle neck, który potem załadowałem do głównego modelu  

**bootle_neck.py**
```python

def trainTopModel():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = to_categorical(train_generator.classes, 20)

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels =to_categorical(val_generator.classes, 20)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(20, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)
```

**main.py**
```python
base_model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dense(20, activation='softmax'))

    top_model.load_weights(top_model_weights_path)

    model = Model(input= base_model.input, output= top_model(base_model.output))
```

W modelu zafreezowałem pierwsze 15 warstw, tak aby zostawić ostatni conv block VGG16.  
Ustaliłem 160 epochów i 16 batchów, optymalizator adagrad, po czym użyłem fit_generatora do wytrenowania modelu. Następnie go zapisałem.      
Pierwsze wyniki przestawiały się tak:

![](https://i.imgur.com/xBER3nM.png)
![](https://i.imgur.com/d7M2UJO.png)

po ewaluacji modelu val_accuracy wynosiło około 61%, a val_loss oscylowało nieco poniżej jedynki.  
Po niesatysfakcjonujących wynikach ponownie wczytałem zapisany model, ustawiłem agresywniejsze data augmentation i spróbowałem jeszcze raz.  
Zmieniłem również optymalizator na adam.  

Kolejne wyniki wyglądały dużo lepiej:

![](https://i.imgur.com/3AjsIeF.png)
![](https://i.imgur.com/ygrKhY4.png)

val_accuracy na poziomie około 81% i val_loss na poziomie około 0.55%

## Klasyfikacja obrazów z 2 głównych klas

Podzieliłem dataset na 2 główne klasy.  
Do wcześniej stworzonego modelu dodałem nową warstę wyjścia i wytrenowałem na nim nowy model klasyfikujący obrazy jedzenia zdrowego i niezdrowego.

**main.py**
```python
 base_model = load_model('model_saved.h5')
    ll = base_model.output
    ll = Dense(20,activation="softmax")(ll)

    model = Model(inputs=base_model.input,outputs=ll)
```

Wykresy:

![](https://i.imgur.com/yvEkqAq.png) 
![](https://i.imgur.com/BuV3FDV.png)  
*ostatnie 20 epochów ucięte omyłkowo przez przybliżenie*  

Udało się podciągnąć val_accuracy do około 85% i val_loss do około 0.45%.  
Myślę, że z większą liczbą sampli, rozdzielczością obrazów, lub innym podejściem do data augmentation udałoby się wyciągnąć więcej, ale jestem zadowolony :)

## Kilka przykładowych predictów:

**same zdrowe:**
https://i.imgur.com/uqhcSxC.png  

**same niezdrowe:**
https://i.imgur.com/E06QTeV.png





