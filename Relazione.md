#Algoritmi per il Calcolo Parallelo
##Marco Tieghi

----

###Richiesta
L'esercizio richiedeva la realizzazione di un programma in grado di effettuare una fattorizzazione QR ricorrendo all'algoritmo di Gram-Schmidt modificato, usando prima un approccio in seriale (scrivendo cioè un programma che possa eseguire in seriale il codice), poi un approccio in parallelo (scrivendo un programma in grado di ripartire i calcoli su più unità di calcolo in contemporanea). Per l'approccio in parallelo si richiedeva l'uso degli strumenti messi a disposizione dalle librerie di CUDA, il sistema di calcolo tramite GPU sviluppato da NVIDIA.

###Approccio

Prima di tutto si è ideato e sviluppato il programma in seriale, in quanto ha permesso di comprendere meglio che algoritmo si doveva applicare. Si è scritto il codice in C, usando gli strumenti di questo linguaggio per strutturare il programma, e si sono definite le funzioni secondo il prototipo fornito con il testo dell'esercitazione. Da queste basi si è potuto scivere così il programma in seriale.

In seguito si è cercato di comprendere meglio i meccanismi di CUDA esposti a lezione, cercando di capire la struttura di un programma CUDA e l'approccio verso questo sistema.
Compresi perciò i meccanismi, si è potuto modificare il codice in seriale per adattarlo alla struttura CUDA, modificando le funzioni in modo tale che potessero lavorare con i threads, definendo i diversi kernel del programma e calcolando la giusta struttura di blocchi e threads che si sarebbero usati.

Nel codice ogni funzione e ogni passo è descritto così da avere ben chiaro cosa si sta eseguendo.
