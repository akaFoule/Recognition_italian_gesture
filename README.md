# Riconoscimento di gesti umani con le mani (italiani) usando il framework Hand MediaPipe e una rete neurale LSTM

Riconoscimento di gesti umani con le mani usando il framework Hand MediaPipe e una rete neurale LSTM (Long Short Term Memory) con [MediaPipe Hand tracking](https://google.github.io/mediapipe/solutions/hands) on desktop (CPU)

### ISTRUZIONI:

- Registra dei video, nei quali fai il gesto ch vuoi far riconoscere o apprendere alla rete
- Ottieni le caratteristiche dei punti chiave della mano per ogni fotogramma al secondo (fps) per ongi video dato in input per una parola e trasformalo in un file txt

## 1. Installazione Hand MediaPipe framework

- Clona il repository
- Per il build con Bazel esegui, apri la cartella nel terminale ed esegui i comandi presenti nel file txt Comandi

## 2. Create your own training data

Crea una cartella **trainvideoset** dove all'interno sono archiviati in cartelle i video relativi al gesto. Un gesto = una cartella con N video. Avvia **build_model.py** per ottenere i file txt e video di output mp4 con tracciamento della mano. Devi avere almeno 150 video per una parola (un gesto) per addestrare correttamente la rete. Se si vuole usarne di meno bisogna modificare 157 del file build_model

```
python3 build.py --input_data_path=[INPUT_PATH] --output_data_path=[OUTPUT_PATH]

```

Per esempio: *input_data_path=/.../trainvideoset/*  and *output_data_path=/.../traintxtset/* 

```
trainvideoset
|-- Bombazza
|	|-- Bombazza1.mp4
|	|-- Bombazza2.mp4
|	|-- Bombazza3.mp4
|	...
|	|-- Bombazza150.mp4
|
|-- Bacio
|	|-- Bacio1.mp4
|	|-- Bacio2.mp4
|	|-- Bacio3.mp4
|	...
|	|-- Bacio150.mp4
|...
```

La cartella di output sarà inizialmente vuota poi, quando il build sarà completato, conterrà un acartella con i file mp4 archiviati in base al gesto e 2 cartelle 
Absolute e Relative, nella prima ci sono i punti normalizzati nella seconda invece le coordinate trasformate in un intero (-1, 0, 1)

Per esempio:

```
traintxtset
|-- Absolute
|	|-- Bombazza
|		|-- Bombazza1.txt
|		|-- Bombazza2.txt
|		|-- Bombazza3.txt
|		...
|		|-- Bombazza150.txt
|	...
||-- Relative
|	|-- Bombazza
|		|-- Bombazza1.txt
|		|-- Bombazza2.txt
|		|-- Bombazza3.txt
|		...
|		|-- Bombazza150.txt
|	...
|-- Bombazza
|	|-- Bombazza1.mp4
|	|-- Bombazza2.mp4
|	|-- Bombazza3.mp4
|	...
|	|-- Bombazza150.mp4
|
|-- Bacio
|	|-- Bacio1.mp4
|	|-- Bacio2.mp4
|	|-- Bacio3.mp4
|	...
|	|-- Bacio150.mp4
|...
```
**Importante**: Assegna un nome alla cartella con attenzione poiché il nome della cartella sarà l'etichetta stessa per i dati video. (NON utilizzare la barra spaziatrice o "_" per il nome della cartella, ad esempio * train_videos_set * o * train video set *)

## 3. Addestramento della rete LSTM 

La rete viene addestrata, e ne restituisce un modello durante la fase di build. E salva all'interno della cartella di output il grafico dell' accuratezza, della loss e la matrice di confusione

## 4. Test

Per la fase di test avviare 
```
python3 main.py
```

Il risultato è mostrato in a GUI with Tkinter library

- Riconosci un gesto

![alt]()

- Oppure una sequenza di 2 o 3 gesti

![alt]()
