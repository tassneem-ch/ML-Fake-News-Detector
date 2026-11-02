# Fake News Detector

An educational NLP project in R: train a neural network to tell fake news
articles from real ones, then expose it as an interactive Shiny app.

Two scripts do the work. `fake_news_nn.R` trains and saves the model;
`app.R` loads it and lets you paste text and get a verdict.

## How it works

**1. Pre-processing** — the text is lowercased, stripped of punctuation,
numbers and English stopwords, then stemmed (`running` → `run`).

**2. TF-IDF** — a document-term matrix weighted by term frequency against
inverse document frequency, so rare words like *conspiracy* carry more
weight than common ones like *news*. Only the **top 500 terms** are kept to
keep the feature space manageable.

**3. Neural network** — a single hidden-layer network trained with `nnet`
(10 hidden units, 200 iterations, weight decay `1e-4`) on an 80/20
stratified split with `set.seed(42)` for reproducibility.

**4. Shiny app** — the trained model, its top terms and the IDF weights are
saved together to `model_data.rds`, so the app can rebuild the exact same
TF-IDF features for any new input.

## Files

| File | Purpose |
|------|---------|
| `fake_news_nn.R` | Training script: load data, clean, vectorise, train, evaluate, save |
| `app.R` | Shiny app (`TruthLens`) for interactive classification |
| `model_data.rds` | Saved model + `top_terms` + `idf_weights`, consumed by `app.R` |
| `Fake.csv`, `True.csv` | The two labelled datasets (~116 MB combined) |
| `confusion_matrix.png` | Held-out confusion matrix plot |
| `wordcloud_fake.png`, `wordcloud_true.png` | Most frequent words per class |
| `Rplots.pdf` | Auto-generated plot output from running the script |

## Running it

```r
# 1. Train and save the model (writes model_data.rds)
Rscript fake_news_nn.R

# 2. Launch the interactive app
Rscript app.R        # then open http://localhost:5000
```

Both scripts install any missing packages on first run from CRAN:

`tidyverse`, `tidytext`, `tm`, `caret`, `nnet`, `neuralnet`, `e1071`,
`Matrix`, `SnowballC`, `wordcloud`, `RColorBrewer`, plus `shiny`, `bslib`,
`thematic` and `shinycssloaders` for the app.

Run the training script **first** — `app.R` checks for `model_data.rds` and
will tell you if it is missing rather than silently guessing.

## What it will and won't do

This is a teaching project, and it behaves like one:

- **It learns word frequencies, not meaning.** It has no understanding of
  the claims in an article, only of which terms tend to co-occur in the
  training data.
- **The verdict is a pattern match, not a fact check.** The app says so
  itself, in a permanent note under every result.
- **No accuracy figure is claimed here** because none is asserted — the
  confusion matrix is produced on your own run, against your own data.
- **The datasets are large** (about 116 MB in total), which is why they
  live directly in the repository rather than being regenerated.

Always verify with multiple sources. That advice is not a disclaimer
bolted on at the end — it is the reason this project exists.

## Limitations and possible next steps

- A bag-of-words model cannot detect sarcasm, negation or context.
- 500 terms is a deliberately small vocabulary; raising it trades
  precision for speed and memory.
- Reasonable extensions: a held-out cross-validation pass, comparing
  against a linear model, or moving the pipeline to `text2vec` for speed.
