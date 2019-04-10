# PROGETTO BIG DATA 2018/19
Codice per il progetto del corso di big data 18/19
## Requisiti
Il codice è pensato per essere eseguito tramite [Docker](https://www.docker.com/) per gestire automaticamente le dipendenze verso i pacchetti python e spark. Se Spark è già installato nella macchina o su un cluster raggiungibile, si può modificare il Dockerfile, per usare un'immagine che non includa spark, installare localmente tutti i pacchetti python necessari, o specificare nel codicer il master per Spark.

Per utilizzare il progetto tramite Docker, il primo step è l'installazione di Docker, spiegata dettagliatamente nella guida di installazione ufficiale per [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-using-the-repository) o [Mac](https://docs.docker.com/docker-for-mac/install/). A questo punto, nella cartella contenente il progetto, è necessario modificare il file start.sh, sostituendo  ```/home/filippo/bigdata/:/home/jovyan/``` con ```path_della_cartella_contenente_il_progetto:/home/jovyan/```. Ora possiamo eseguire il file ```build.sh``` e il file ```start.sh```. 
Nota: La porta su cui jupyterlab e sparkUI ascoltano è rimappata nel comando di start, quindi o si modifica anche quello, o si usano gli indirizzi [http://localhost:9000/lab?]() per jupyterlab e [http://localhost:9001/]() per sparkUI.

Nel caso di un'installazione senza Docker, è necessario jupyterlab per eseguire i notebook, pyspark, numpy e pandas. Soddisfando tutte le dipendenze, è possibile lanciare jupyterlab con ```jupyter lab``` dalla cartella contenente il progetto.

## Dataset
Il dataset utilizzato è disponibile, su richiesta, [qui](https://zenodo.org/record/1489920). Il codice del progetto si aspetta di trovare i file del dataset nella cartella ```./dataset/```.

## File dei word embedding
I word embedding utilizzati sono scaricabili [qui](http://www.maurodragoni.com/research/opinionmining/dranziera/embeddings-evaluation.php). Per l'esecuzione del progetto sono da inserire nella cartella ```./dataset/wordembeddings/```.

## Esecuzione del codice

### Preprocessing del dataset
Eseguire il file ```./work/preprocessarticles.sh``` per formattare correttamente i file xml. A seguire, per generare i dataframe di spark, è necessario eseguire il codice del notebook ```./work/ParseXml.ipynb```. In particolare è necessario eseguire l'ultima cella per tutti i file da parsare. Infine eseguire il notebook ```./work/DataMerger.ipynb```.

### Test con TF e TF-IDF
Per eseguire i test sul dataset ottenuto al passo precedente, è necessario eseguire il file execTest.py. Per effettuare test con classificatori differenti è sufficiente definire la nuova pipeline e la sua descrizione (viene inserita nel file dei risultati).
I test vengono effettuati usando il dataset degli articoli etichettati per publisher come training set, e quelli etichettati singolarmente come test set. 
I risultati vengono scritti sul file ```./work/results```, e vengono riportate le metriche: accuracy, precision, recall, e f1. I risultati contengono sia la media della cross-validation sul training set, sia il risultato che il modello ha sul test set (trainato sull'intero training set).

### Test con word embedding
Per eseguire i test con i word embedding è necessario seguire i seguenti passi:
1. Calcolare la lista di tutte le parole presenti nel nostro dataset. Per farlo, è sufficiente eseguire le celle del notebook ```./work/wordEmb/WordListComputing.ipynb```. Questo genererà un file nella stessa cartella del notebook.
2. Usare la precedente lista per ridurre la dimensione dei file contenenti i word-embedding. Il notebook ```./work/wordEmb/WEReducer.ipynb``` contiene il codice per fare ciò, e ha sia una cella per esegurire il procedimento su tutti i file, sia una cella in cui è possibile modificare il nome e eseguirlo quindi su un file alla volta.
3. I file per effettivamente eseguire i test con i wordembedding sono contenuti nella cartella ```./work/wordEmb/final/```. Nello specifico, è sufficiente modificare la variabile indices del file ```./work/wordEmb/final/startTest``` con gli indici dei word embedding embedding da testare. I valori validi sono da 1 a 10, e le corrispondenze sono indicate nel file ```./work/wordEmb/final/Index2WE.txt```. 


Come nel caso di TF e TFIDF i test vengono effettuati usando il dataset degli articoli etichettati per publisher come training set, e quelli etichettati singolarmente come test set. I test sono però eseguiti in due modi diversi, ovvero applicando i wordembedding su tutte le parole dell'articolo, e applicandol i wordembedding solo dopo aver filtrato le ''stop-words'', parole come "I", ''the'' etc. che potrebbero avere poca importanza e confondere il modello. I risultati vengono scritti sui file ```./work/wordEmb/final/resultsRaw{i}``` nel caso della procedura senza filtri sulle parole, e nel file ```./work/wordEmb/final/resultsFiltered{i}```, i è in entrambi i casi l'indice del wordEmbedding correntemente usato. Vengono riportate le metriche: accuracy, precision, recall, e f1. I risultati contengono sia la media della cross-validation sul training set, sia il risultato che il modello ha sul test set (trainato sull'intero training set).
Per testare differente tecniche di combinazione dei wordembedding (somma, media, etc.) è sufficiente modificare il file ```./work/wordEmb/final/weTest.py``` alla riga 58.
