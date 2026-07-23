# Job Matching Engine
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gabrieleafferni/job-matching-engine/blob/main/job_matching_engine.ipynb)
**Multi-source job aggregation with hybrid semantic scoring and LLM re-ranking.**

A retrieval-and-ranking pipeline that collects job postings from five job boards via their official APIs, normalises them into a common schema, applies configurable eligibility filters, and ranks them against a candidate profile using multilingual sentence embeddings combined with an LLM-as-judge stage.

No scraping — public, documented APIs only.

**Not tied to any profile or country.** Roles, skills, geographic areas, eligibility thresholds and the candidate text are all parameters in a single configuration cell; adapting it to a different search requires no code changes. The bundled configuration — junior data analyst in Piedmont, Italy — is a working example, not a constraint.

*[Versione italiana più sotto ⬇](#italiano)*

---

## Pipeline

```
Adzuna · JSearch · Jooble · Careerjet · Arbeitnow
                 │
                 ▼  normalisation into a shared record schema
     cross-source deduplication (canonical URL + title|company)
                 │
                 ▼
     eligibility filters
       · reserved-category postings (Italian L. 68/99)
       · seniority and required years of experience (regex extraction)
       · work arrangement and geographic scope
                 │
                 ▼
     hybrid scoring — cosine similarity on multilingual
     embeddings (0.8) + saturating keyword match (0.2)
                 │
                 ▼
     LLM re-ranking on top-N — structured JSON output
     (score, rationale, main gap)
                 │
                 ▼
     export — CSV (local) or Google Sheets (Colab)
```

## Design decisions

**Multilingual embeddings.** Italian job postings routinely mix Italian and English within the same description — a role titled *Data Analyst* whose requirements are written in Italian. `paraphrase-multilingual-MiniLM-L12-v2` projects both languages into a shared vector space, so *"analisi delle serie storiche"* and *"time series forecasting"* land close together. The model is small enough (~470 MB) to run on CPU.

**Hybrid rather than purely semantic scoring.** Semantic similarity alone rewards postings that are broadly "about data" while missing the specific technologies required. Keyword matching alone is brittle against synonyms. The 80/20 combination corrects both failure modes; keyword contribution saturates after eight matches so that keyword-stuffed postings gain no advantage.

**Retrieval, then precision.** Running the LLM across every posting would be slow and expensive. Semantic scoring acts as the retrieval stage, narrowing hundreds of postings to a shortlist; the LLM then evaluates only those, judging *eligibility* rather than textual similarity — a distinction embeddings cannot make.

**Score rescaling.** Raw cosine similarity between long texts clusters in a narrow band (roughly 0.10–0.55), producing a compressed and unreadable ranking. Scores are linearly rescaled across that empirically calibrated band into a 0–100 range.

**Failure isolation.** Each source is wrapped independently: a rate-limited or failing board does not halt the run. LLM calls retry with exponential backoff and jitter on transient status codes (429, 500, 502, 503, 529).

**Auditable filters.** Every filtered-out posting is recorded with its exclusion reason rather than silently dropped, so the rules can be inspected and tuned against real outcomes.

## Running it

```bash
git clone https://github.com/gabrieleafferni/job-matching-engine.git
cd job-matching-engine
pip install -r requirements.txt
cp .env.example .env      # then fill in your keys
jupyter notebook job_matching_engine.ipynb
```

On Google Colab, use **Colab Secrets** (🔑 in the sidebar) instead of the `.env` file. The notebook detects its environment and loads credentials accordingly.

No key is mandatory: the pipeline enables only the sources it finds credentials for, and Arbeitnow requires no authentication at all. The notebook itself documents how to obtain each key.

## Adapting it to your own search

Everything that defines **who** is searching, **what** for and **where** lives in a single configuration cell. No other cell needs to change.

| § | Defines | Change it to |
|---|---|---|
| 1 | Candidate profile | Your own CV in prose form |
| 2 | Target roles | Your job titles, in any number of rotating blocks |
| 3 | Bonus keywords | The technologies of your field |
| 4 | Country, locale, geography | ISO country code, locale, and your own areas with their cities |
| 5 | Eligibility filters | Experience threshold, seniority markers, filters to disable |
| 6 | Technical parameters | Pages per source, posting freshness, scoring weights |

Three worked examples:

```python
# Junior data analyst, Turin — the bundled example
PAESE, LOCALE = "it", "it_IT"
AREE_AMMESSE, MAX_ANNI_ESPERIENZA = ["Piemonte"], 1

# Mid-level frontend developer, Berlin
PAESE, LOCALE = "de", "de_DE"
AREE_GEOGRAFICHE = {"Berlin-Brandenburg": ["berlin", "potsdam", "brandenburg"]}
AREE_AMMESSE, MAX_ANNI_ESPERIENZA = ["Berlin-Brandenburg"], 5
FILTRO_CATEGORIE_PROTETTE = False     # Italy-specific regulation
BLOCCHI = {"Frontend": ["frontend developer", "react developer", "ui engineer"]}

# Remote-only, no geographic constraint
LUOGHI, AREE_AMMESSE, SOLO_REMOTO = [""], [], True
```

Searching for a senior role rather than a junior one means emptying `MARCATORI_SENIORITY` and setting `MAX_ANNI_ESPERIENZA = None` — the filter switches off without touching any code. Assertions at the end of the configuration cell catch the common mistakes (an area listed but never defined, weights that do not sum to one) with a message pointing at the offending section.

One caveat: the rescaling band `SIM_MIN`/`SIM_MAX` is calibrated for the bundled profile. After the first run with a different one, inspect the `score_semantico` distribution — if scores bunch up near 0 or 100, move the bounds. The scoring cell prints mean and maximum precisely to make this check immediate.

## Known limitations

- Deduplication on (title, company) is aggressive: two genuinely distinct openings sharing a title within the same company collapse into one.
- The experience-years filter is heuristic. A 40-character context window around each number reduces false positives but does not eliminate them.
- The rescaling band is calibrated for one profile and needs retuning for substantially different ones.
- The LLM scores postings individually, without comparative context, so scores are consistent on average but not strictly comparable to one another.

## Stack

Python · pandas · sentence-transformers · Anthropic API · gspread · requests

---

<a name="italiano"></a>

# Job Matching Engine — versione italiana

**Aggregazione multi-fonte di annunci di lavoro con scoring semantico ibrido e re-ranking LLM.**

Pipeline di retrieval e ranking che raccoglie offerte da cinque job board tramite API ufficiali, le normalizza in uno schema comune, applica filtri di eleggibilità configurabili e le ordina per affinità con un profilo candidato, combinando embedding multilingue e una valutazione LLM-as-judge.

Nessuno scraping: solo API pubbliche e documentate.

**Non è legata a un profilo o a un paese.** Ruoli, skill, aree geografiche, soglie di eleggibilità e testo del candidato sono parametri di un'unica cella di configurazione: adattarla a un'altra ricerca non richiede modifiche al codice. La configurazione inclusa — analista dati junior in Piemonte — è un esempio funzionante, non un vincolo.

## Scelte tecniche

**Embedding multilingue.** Gli annunci italiani mescolano abitualmente italiano e inglese nella stessa descrizione. `paraphrase-multilingual-MiniLM-L12-v2` proietta le due lingue nello stesso spazio vettoriale, così *"analisi delle serie storiche"* e *"time series forecasting"* finiscono vicini. Il modello è abbastanza leggero (~470 MB) da girare su CPU.

**Scoring ibrido invece che puramente semantico.** Il solo semantico premia annunci genericamente "di area data ma privi delle tecnologie richieste; il solo keyword matching è fragile rispetto ai sinonimi. La combinazione 80/20 corregge entrambi i difetti, e il contributo keyword satura dopo otto match perché gli annunci infarciti di buzzword non ne traggano vantaggio.

**Prima retrieval, poi precisione.** Interrogare l'LLM su tutti gli annunci sarebbe lento e costoso. Lo scoring semantico fa da stadio di retrieval e riduce centinaia di annunci a una shortlist; l'LLM valuta solo quelli, giudicando l'*idoneità alla candidatura* più che la somiglianza testuale — una distinzione che gli embedding non sanno fare.

**Riscalatura dei punteggi.** La similarità coseno grezza fra testi lunghi si concentra in una banda stretta (circa 0.10–0.55), producendo una classifica schiacciata e illeggibile. I punteggi vengono riscalati linearmente su quella banda, calibrata empiricamente, in un intervallo 0–100.

**Isolamento dei guasti.** Ogni fonte è incapsulata singolarmente: una board in errore o in rate limit non interrompe la run. Le chiamate all'LLM ritentano con backoff esponenziale e jitter sui codici transitori.

**Filtri verificabili.** Ogni annuncio scartato viene registrato con il motivo dell'esclusione invece di sparire silenziosamente, così le regole restano ispezionabili e tarabili sui risultati reali.

## Come si esegue

```bash
git clone https://github.com/gabrieleafferni/job-matching-engine.git
cd job-matching-engine
pip install -r requirements.txt
cp .env.example .env      # poi inserisci le chiavi
jupyter notebook job_matching_engine.ipynb
```

Su Google Colab si usano i **Colab Secrets** (icona 🔑 nella barra laterale) al posto del file `.env`. Il notebook riconosce l'ambiente e carica le credenziali di conseguenza.

Nessuna chiave è obbligatoria: la pipeline attiva solo le fonti per cui trova una credenziale, e Arbeitnow non richiede autenticazione. Il notebook stesso documenta come ottenere ciascuna chiave.

## Adattarlo alla propria ricerca

Tutto ciò che definisce **chi** cerca, **cosa** e **dove** sta in un'unica cella di configurazione. Nessun'altra cella va toccata.

| § | Definisce | Come cambiarlo |
|---|---|---|
| 1 | Profilo del candidato | Il proprio CV in forma discorsiva |
| 2 | Ruoli da cercare | Le proprie mansioni, in un numero libero di blocchi a rotazione |
| 3 | Skill che valgono un bonus | Le tecnologie del proprio settore |
| 4 | Paese, lingua, geografia | Codice ISO, locale e le proprie aree con le relative città |
| 5 | Filtri di eleggibilità | Soglia di esperienza, marcatori di seniority, filtri da disattivare |
| 6 | Parametri tecnici | Pagine per fonte, freschezza degli annunci, pesi dello scoring |

Tre esempi:

```python
# Analista dati junior a Torino — la configurazione di esempio inclusa
PAESE, LOCALE = "it", "it_IT"
AREE_AMMESSE, MAX_ANNI_ESPERIENZA = ["Piemonte"], 1

# Frontend developer mid-level a Berlino
PAESE, LOCALE = "de", "de_DE"
AREE_GEOGRAFICHE = {"Berlin-Brandenburg": ["berlin", "potsdam", "brandenburg"]}
AREE_AMMESSE, MAX_ANNI_ESPERIENZA = ["Berlin-Brandenburg"], 5
FILTRO_CATEGORIE_PROTETTE = False     # normativa specifica italiana
BLOCCHI = {"Frontend": ["frontend developer", "react developer", "ui engineer"]}

# Solo remoto, nessun vincolo geografico
LUOGHI, AREE_AMMESSE, SOLO_REMOTO = [""], [], True
```

Chi cerca un ruolo senior anziché junior svuota `MARCATORI_SENIORITY` e porta `MAX_ANNI_ESPERIENZA` a `None`: il filtro si spegne senza toccare il codice. Alcune assertion in coda alla configurazione intercettano gli errori più comuni (un'area citata ma mai definita, pesi che non sommano a 1) indicando il paragrafo da correggere.

Un avvertimento: la banda `SIM_MIN`/`SIM_MAX` è calibrata sul profilo di esempio. Dopo la prima run con un profilo diverso, guarda la distribuzione di `score_semantico`: se i punteggi sono schiacciati verso 0 o verso 100, sposta i due estremi.

## Limiti noti

- La deduplica su (titolo, azienda) è aggressiva: due posizioni realmente distinte con lo stesso titolo nella stessa azienda vengono collassate in una.
- Il filtro sugli anni di esperienza è euristico: la finestra di contesto di 40 caratteri riduce i falsi positivi ma non li elimina.
- La banda di riscalatura è calibrata su un profilo specifico e va ritarata per profili molto diversi.
- L'LLM valuta un annuncio alla volta, senza contesto comparativo: i punteggi sono coerenti in media ma non strettamente confrontabili.

---

Progetto personale. Le API sono utilizzate nel rispetto dei rispettivi termini di servizio; nessun dato degli annunci è ridistribuito in questo repository.
