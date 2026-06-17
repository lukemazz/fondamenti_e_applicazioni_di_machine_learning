# Documentazione del Progetto — CDC Diabetes Health Indicators
**Corso:** Fondamenti e Applicazioni del Machine Learning
**Dataset:** CDC Diabetes Health Indicators (UCI ML Repository, ID 891)
**Studente:** singolo → 1 classificatore manuale

---

## Descrizione del Dataset

Il dataset proviene dal CDC (Centers for Disease Control and Prevention) ed è basato
sul sondaggio BRFSS (Behavioral Risk Factor Surveillance System) del 2015.
Contiene **253.680 righe** e **22 colonne**, di cui 21 sono attributi di input e 1 è
la variabile target `Diabetes_binary`.

**Variabile target:** `Diabetes_binary`
- 0 = nessun diabete
- 1 = diabete o pre-diabete

**Problema di classificazione binaria** con forte sbilanciamento delle classi:
circa l'86% delle istanze appartiene alla classe 0 e il 14% alla classe 1.

### Tipologie di attributi

| Tipo | Attributi |
|---|---|
| Nominale/Binario (0 o 1) | HighBP, HighChol, CholCheck, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, DiffWalk, Sex |
| Ordinale | GenHlth (1–5), Age (1–13), Education (1–6), Income (1–8) |
| Numerico continuo | BMI (12–98) |
| Numerico discreto (0–30) | MentHlth, PhysHlth |

---

## Task 1 — Caricamento e Trasformazione del Dataset

### Obiettivo
Preparare il dataset in un formato adatto ai classificatori trattati a lezione,
che lavorano con attributi nominali, ed estrarre i due file CSV richiesti dalla
specifica di progetto.

### 1.1 Caricamento del dataset originale

Il dataset viene caricato da file CSV tramite `pandas.read_csv()`.
Su Google Colab, il pacchetto `ucimlrepo` permette di scaricarlo direttamente
da UCI ML Repository senza dover gestire manualmente il download del file.
Il dataset viene poi salvato localmente in CSV per essere riutilizzato
nelle celle successive senza dover riscaricarlo ogni volta.

### 1.2 Discretizzazione degli attributi numerici

I classificatori come Naïve Bayes lavorano naturalmente con attributi nominali.
Gli attributi `BMI`, `MentHlth` e `PhysHlth` sono numerici e devono essere
convertiti in categorie ordinali tramite **discretizzazione**.

**BMI** viene suddiviso secondo le soglie cliniche standard dell'OMS:
- `sottopeso` → BMI < 18.5
- `normale`   → 18.5 ≤ BMI < 25
- `sovrappeso`→ 25 ≤ BMI < 30
- `obeso`     → BMI ≥ 30

Queste soglie non sono arbitrarie: corrispondono alle categorie utilizzate in
medicina per classificare il peso corporeo in relazione al rischio di malattie
metaboliche, tra cui il diabete.

**MentHlth e PhysHlth** (giorni di malessere nell'ultimo mese, 0–30) vengono
suddivisi in quattro categorie clinicamente ragionevoli:
- `nessuno`  → 0 giorni
- `lieve`    → 1–7 giorni (circa una settimana)
- `moderato` → 8–14 giorni (metà mese)
- `grave`    → > 14 giorni (più della metà del mese)

Le colonne numeriche originali vengono rimosse dopo la discretizzazione per
evitare ridondanza nel dataset.

**Motivazione:** mantenere le colonne originali dopo la discretizzazione
introdurrebbe informazione duplicata che potrebbe falsare i modelli,
in particolare Naïve Bayes che assume indipendenza tra gli attributi.

### 1.3 Estrazione di `manuale.csv`

Vengono selezionate **14 righe** (7 per classe) con campionamento casuale
stratificato per classe, usando `random_state=42` per garantire la
riproducibilità. Le righe vengono poi mescolate in ordine casuale prima
di salvare il file, per evitare che il classificatore manuale benefici
di un ordinamento artificiale.

**Motivazione del bilanciamento:** estrarre lo stesso numero di campioni
per entrambe le classi è essenziale per poter calcolare significativamente
le probabilità condizionali di Naïve Bayes su un insieme così piccolo.
Se si seguisse la distribuzione originale (86/14), su 14 righe si otterrebbero
circa 12 istanze classe 0 e solo 2 classe 1, rendendo impossibile stimare
le probabilità per la classe diabetica.

### 1.4 Estrazione di `training.csv`

Il dataset completo (253k righe) è troppo grande per le fasi manuali ma ideale
per Scikit-Learn. Viene estratto un campione di **5.000 righe** usando
`train_test_split` con il parametro `stratify=y`.

**Motivazione della stratificazione:** la stratificazione garantisce che
la proporzione originale 86/14 tra le classi venga rispettata anche nel
campione estratto. Senza stratificazione, un campionamento casuale potrebbe
per pura sfortuna produrre una distribuzione diversa da quella attesa,
invalidando le analisi successive.

---

## Task 2 — Classificatore Naïve Bayes Manuale su `manuale.csv`

### Obiettivo
Implementare da zero (senza Scikit-Learn) un classificatore Naïve Bayes,
addestrarlo su `manuale.csv` e valutarne le prestazioni sullo stesso file.

### Scelta del classificatore

È stato scelto **Naïve Bayes** per i seguenti motivi:
1. La maggior parte degli attributi del dataset è già binaria (0/1), il che rende
   il calcolo delle probabilità condizionali molto diretto (conteggi semplici).
2. Il calcolo a mano è fattibile su 14 campioni.
3. Naïve Bayes è robusto alla presenza di attributi ridondanti, che sono
   frequenti in dataset di tipo sanitario.

### Fondamento teorico

Il classificatore si basa sulla **regola di Bayes** con l'assunzione di
**indipendenza condizionale** tra gli attributi dato la classe:

```
P(classe | a₁, ..., aₖ) ∝ P(classe) × P(a₁|classe) × ... × P(aₖ|classe)
```

Si sceglie la classe con il valore più alto di questo prodotto.

### 2.1 Separazione attributi / target

Il dataframe viene diviso in `X_man` (attributi) e `y_man` (target).
Le costanti `TARGET` e `CLASSI` vengono definite globalmente per essere
riutilizzate nei task successivi senza duplicazione di codice.

### 2.2 Addestramento con stimatore di Laplace

La funzione `train_naive_bayes(X, y, classi)` calcola:

**Probabilità a priori** per ogni classe con Laplace:
```
P(classe) = (count(classe) + 1) / (n_totale + n_classi)
```

**Probabilità condizionali** per ogni attributo e ogni valore possibile,
con stimatore di Laplace:
```
P(attr=v | classe) = (count(attr=v, classe) + 1) / (count(classe) + n_valori_distinti)
```

**Motivazione dello stimatore di Laplace:** su un dataset piccolo come
`manuale.csv` (14 righe), è molto probabile che alcuni valori di un
attributo non compaiano mai in una certa classe. Senza Laplace, quella
probabilità sarebbe 0, e il prodotto di tutte le probabilità condizionali
sarebbe anch'esso 0, rendendo impossibile classificare. Laplace aggiunge
un conteggio fittizio di 1 a ogni combinazione (attributo, valore, classe),
garantendo che nessuna probabilità sia mai zero.

### 2.3 Predizione con log-probabilità

La funzione `predict_naive_bayes()` calcola il prodotto delle probabilità
in **spazio logaritmico**:
```
LogScore(classe) = log P(classe) + log P(a₁|classe) + log P(a₂|classe) + ...
```

**Motivazione:** moltiplicare molte probabilità piccole (tutte tra 0 e 1)
porta rapidamente a underflow numerico — il numero diventa così piccolo
che il computer lo arrotonda a zero. Lavorare con i logaritmi trasforma
le moltiplicazioni in somme, eliminando il problema. La classe con il
log-score più alto corrisponde comunque alla classe con la probabilità più
alta, perché il logaritmo è una funzione monotona crescente.

### 2.4 Valutazione su `manuale.csv`

Il classificatore viene applicato a tutte le righe di `manuale.csv`
usando il modello addestrato sull'intero stesso file (train = test).

Vengono calcolate le seguenti metriche dalla confusion matrix (TP, TN, FP, FN):

| Metrica | Formula | Significato |
|---|---|---|
| Accuracy | (TP+TN) / totale | Percentuale di predizioni corrette |
| Precision | TP / (TP+FP) | Dei predetti diabetici, quanti lo sono davvero |
| Recall | TP / (TP+FN) | Dei diabetici reali, quanti vengono identificati |
| F1-score | 2·P·R / (P+R) | Media armonica di precision e recall |

**Nota:** con train = test lo stesso file, l'accuracy tende ad essere alta
perché il modello ha "visto" i dati su cui viene valutato. Questo non è
un problema in questa fase: lo scopo del Task 2 è verificare la correttezza
del calcolo manuale, non misurare la capacità di generalizzazione.

---

## Task 3 — Analisi Esplorativa di `training.csv`

### Obiettivo
Verificare la qualità dei dati (data cleaning) e produrre visualizzazioni
che descrivano la distribuzione degli attributi e le loro relazioni con il target.

### 3.1 Data Cleaning

Vengono effettuati due controlli sistematici:

**Valori mancanti:** `isnull().sum()` conta i NaN per ogni colonna.
Il dataset CDC è in genere pulito, ma la verifica è necessaria perché
valori mancanti non gestiti portano errori silenziosi nei modelli.

**Valori fuori range:** per ogni attributo viene definito l'insieme dei
valori ammissibili e si contano le righe anomale. Ad esempio, `GenHlth`
deve essere intero tra 1 e 5; `BMI_cat` deve essere una delle quattro
etichette definite in fase di discretizzazione.

**Motivazione:** osservazioni errate (es. BMI = 999, o un attributo binario
con valore 2) potrebbero distorcere le probabilità calcolate da Naïve Bayes
e le distanze calcolate da k-NN. La verifica va fatta prima di qualsiasi analisi.

### 3.2 Distribuzione delle classi

Si stampa il conteggio assoluto e percentuale delle classi per confermare
che lo sbilanciamento del dataset originale (~86/14) sia stato preservato
anche nel campione estratto grazie alla stratificazione.

### 3.3 Boxplot

Per visualizzare la distribuzione di BMI, MentHlth e PhysHlth separatamente
per i due gruppi (diabetici e non diabetici), le colonne categoriche vengono
temporaneamente riconvertite in numeri interi ordinali (0, 1, 2, 3)
usando dizionari di mappatura.

I boxplot mostrano mediana, quartili e outlier per ogni gruppo, permettendo
di vedere a colpo d'occhio se i due gruppi differiscono su questi attributi.
**Atteso:** BMI medio più alto nel gruppo diabetico (categoria obeso più frequente).

### 3.4 Pairplot

La scatter matrix mostra le relazioni tra coppie di attributi principali
(BMI, MentHlth, PhysHlth, GenHlth, Age), con i punti colorati per classe.

**Motivazione:** permette di vedere se le due classi sono linearmente separabili
in qualche coppia di attributi, cosa che suggerirebbe l'uso di classificatori
lineari (Regressione Logistica, Perceptrone). Se i punti delle due classi si
sovrappongono molto, i classificatori non lineari (Random Forest, k-NN) saranno
probabilmente più adatti.

La palette usa `'Non diabetico': 'steelblue'` e `'Diabetico': 'tomato'`
con chiavi stringa perché la colonna `hue` contiene le etichette testuali
prodotte dal `.map()`.

### 3.5 Matrice di correlazione

Si calcola la correlazione di Pearson tra tutte le coppie di colonne numeriche
del dataset. Le colonne categoriche (`BMI_cat`, `MentHlth_cat`, `PhysHlth_cat`)
vengono escluse perché le loro versioni numeriche (`BMI_num`, ecc.) sono già
presenti dopo la conversione per i boxplot.

La heatmap evidenzia:
- quali attributi correlano di più con `Diabetes_binary` (candidati utili)
- quali attributi correlano fortemente tra loro (ridondanza, potenziale
  violazione dell'assunzione di indipendenza di Naïve Bayes)

La colonna della correlazione con il target viene stampata in ordine
decrescente per facilitare la feature selection nel Task 4.

---

## Task 4 — Valutazione del Classificatore Manuale su `training.csv`

### Obiettivo
Valutare il Naïve Bayes implementato nel Task 2 su un dataset di dimensioni
reali (5.000 righe) e ottimizzarne le prestazioni tramite feature selection.

### 4.1 Split interno 80/20

`training.csv` viene diviso in train interno (80%, 4.000 righe) e test
interno (20%, 1.000 righe) con stratificazione sulla classe target.

**Motivazione:** a differenza del Task 2 dove train = test, qui serve una
valutazione onesta della capacità di generalizzazione. Il test set non
viene mai usato in fase di addestramento.

### 4.2 Addestramento

Si riusa la funzione `train_naive_bayes()` definita nel Task 2,
questa volta sul train interno del `training.csv` (4.000 righe).
Non viene ridefinita per evitare duplicazione di codice.

### 4.3 Predizione

Le predizioni vengono generate con `predict_naive_bayes()` su ogni riga
del test set. Si usa `iterrows()` per scorrere il DataFrame riga per riga,
convertendo ogni riga in dizionario con `.to_dict()` prima di passarla
alla funzione di predizione.

### 4.4 Metriche di valutazione

Vengono calcolate le stesse metriche del Task 2 (accuracy, precision, recall,
F1-score) dalla confusion matrix.

Con classi sbilanciate (86/14), l'**F1-score è la metrica più importante**:
un classificatore banale che predice sempre "Non diabetico" otterrebbe
accuracy ~86% ma F1 = 0. Il F1-score penalizza questo comportamento perché
combina precision e recall, e il recall sulla classe minoritaria sarebbe 0.

### 4.5 Feature selection basata sulla correlazione

Si selezionano solo gli attributi con correlazione assoluta con il target
superiore a 0.1, filtrati in modo da includere solo quelli presenti
in `X_tr` (escludendo le colonne numeriche ausiliarie `BMI_num`, ecc.
create solo per i grafici nel Task 3).

Il modello viene riaddestrato con il sottoinsieme di attributi selezionati
e rivalutato sullo stesso test set, producendo una tabella comparativa
con i risultati di entrambe le configurazioni.

**Motivazione:** Naïve Bayes assume indipendenza condizionale tra gli
attributi, ma attributi ridondanti o non correlati con il target
introducono rumore nelle probabilità condizionali. Rimuoverli può
migliorare le prestazioni, specialmente su dataset con molti attributi
binari poco informativi.

---

## Task 5 — Addestramento con Scikit-Learn

### Obiettivo
Addestrare e confrontare cinque classificatori di Scikit-Learn su `training.csv`,
ottimizzarne gli iperparametri con cross-validation, e salvare il modello
migliore per l'utilizzo su `real_settings.csv` all'esame.

### 5.1 Encoding delle colonne categoriche

`LabelEncoder` converte le stringhe `'sottopeso'`, `'normale'`, ecc. in
interi (0, 1, 2, 3). Questo è necessario perché tutti i modelli di Scikit-Learn
richiedono input numerici.

**Nota:** si usa `LabelEncoder` e non `OneHotEncoder` perché le categorie
`BMI_cat`, `MentHlth_cat` e `PhysHlth_cat` sono **ordinali** (c'è un ordine
naturale: sottopeso < normale < sovrappeso < obeso), e la codifica intera
preserva questa relazione d'ordine. OneHotEncoder sarebbe più appropriato
per attributi nominali senza ordine.

### 5.2 Classificatori scelti e motivazioni

| Classificatore | Classe Scikit-Learn | Motivazione |
|---|---|---|
| Naïve Bayes | `BernoulliNB` | Baseline; adatto per attributi binari; confronto con l'implementazione manuale |
| Decision Tree | `DecisionTreeClassifier` | Interpretabile; produce regole leggibili; buon punto di partenza per dataset misti |
| Logistic Regression | `LogisticRegression` | Modello lineare robusto; buona baseline per problemi binari; veloce da addestrare |
| k-NN | `KNeighborsClassifier` | Non parametrico; nessuna assunzione sulla distribuzione dei dati |
| Random Forest | `RandomForestClassifier` | Ensemble di alberi; gestisce bene attributi ridondanti e interazioni non lineari |

**Gestione dello sbilanciamento:** `class_weight='balanced'` viene usato
dove supportato (Decision Tree, Logistic Regression, Random Forest).
Questo parametro fa sì che gli errori sulla classe minoritaria (diabetici,
14%) vengano penalizzati proporzionalmente di più durante l'addestramento,
impedendo al modello di ignorare la classe rara.

### 5.3 Ottimizzazione con GridSearchCV

Per ogni classificatore viene eseguita una ricerca a griglia sugli
iperparametri con `GridSearchCV`:

- **Cross-validation:** `StratifiedKFold` con 5 fold, con shuffle e
  `random_state=42` per riproducibilità. La stratificazione garantisce
  che ogni fold mantenga la proporzione 86/14 delle classi.

- **Metrica di scoring:** `f1` (F1-score sulla classe positiva, cioè diabetico).
  Si ottimizza F1 e non accuracy per le stesse ragioni discusse nel Task 4.

- **`n_jobs=-1`:** usa tutti i core disponibili per parallelizzare la ricerca,
  riducendo il tempo di esecuzione.

Per ogni modello viene stampato il progresso e al termine vengono salvati
i best params e le metriche sul test set.

### 5.4 Tabella comparativa

I risultati di tutti i modelli vengono raccolti in un DataFrame e stampati
come tabella riassuntiva con accuracy, precision, recall e F1-score.
I best params trovati da GridSearchCV vengono stampati separatamente
per ogni modello.

### 5.5 Selezione e salvataggio del classificatore finale

Il modello con il **miglior F1-score sul test set** viene selezionato come
classificatore finale. Viene poi riaddestrato sull'intero `training.csv`
(non solo sull'80% di train interno) con i best params trovati dalla
cross-validation, per massimizzare le informazioni usate in fase di training.

Il modello viene salvato su disco con `joblib.dump()` nel file
`classificatore_finale.pkl`. Questo file potrà essere caricato all'esame
con `joblib.load()` e applicato direttamente su `real_settings.csv`.

**Motivazione del riaddestramento su tutto il dataset:** durante la fase
di selezione, l'80% serve per addestrare e il 20% per valutare onestamente.
Una volta scelto il modello migliore, non c'è più bisogno di tenere da
parte il test set: riutilizzare tutto il dataset per il training finale
migliora la qualità del modello produzione.

---

## Struttura dei file prodotti

```
/
├── diabetes_binary_health_indicators_BRFSS2015.csv   ← dataset originale
├── manuale.csv                                        ← 14 campioni (Task 1 e 2)
├── training.csv                                       ← 5000 campioni (Task 1, 3, 4, 5)
├── boxplot.png                                        ← grafici (Task 3)
├── pairplot.png                                       ← grafici (Task 3)
├── correlazione.png                                   ← grafici (Task 3)
├── classificatore_finale.pkl                          ← modello per l'esame (Task 5)
└── code/
    └── main.py                                        ← tutto il codice
```

---

## Riferimenti alle slide del corso

| Argomento trattato nel codice | Slide |
|---|---|
| Tipi di attributi, discretizzazione | PDF 3 — Input del Machine Learning |
| Data cleaning, preparazione dell'input | PDF 3 — Input del Machine Learning |
| Naïve Bayes, stimatore di Laplace | PDF 5 — Algoritmi Parte I |
| Alberi Decisionali (ID3, gain ratio) | PDF 6 — Algoritmi Parte II |
| Regressione Logistica, Perceptrone, k-NN | PDF 7 — Algoritmi Parte III |
| Accuracy, precision, recall, F1, confusion matrix | PDF 8 — Valutazione |
| Cross-validation, GridSearchCV, Scikit-Learn | PDF 9 — Introduzione a Scikit-Learn |
