# ============================================================
#  Fake News Detection — NLP + Neural Network in R
#  Data: Fake.csv and True.csv 
# ============================================================

# ── 1. Install & load packages ────────────────────────────────
packages <- c("tidyverse", "tidytext", "tm", "caret",
              "nnet", "neuralnet", "e1071", "Matrix",
              "SnowballC", "wordcloud", "RColorBrewer")

installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]
if (length(to_install) > 0) install.packages(to_install, dependencies = TRUE)


library(tidyverse)
library(tidytext)
library(tm)
library(caret)
library(nnet)      
library(e1071)
library(Matrix)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)

set.seed(42)

# ── 2. Load data ──────────────────────────────────────────────

fake <- read_csv("Fake.csv") %>% mutate(label = "FAKE")
true <- read_csv("True.csv") %>% mutate(label = "TRUE")

df <- bind_rows(fake, true) %>%
  mutate(
    label    = as.factor(label),
    doc_id   = row_number(),
    content  = paste(title, text, sep = " ")
  ) %>%
  select(doc_id, content, label)

cat("Dataset size:", nrow(df), "\n")
cat("Label distribution:\n"); print(table(df$label))

# ── 3. Text pre-processing (tm pipeline) ─────────────────────
corpus <- VCorpus(VectorSource(df$content))

corpus <- corpus %>%
  tm_map(content_transformer(tolower)) %>%
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords("en")) %>%
  tm_map(stemDocument) %>%          
  tm_map(stripWhitespace)

# ── 4. Build TF-IDF Document-Term Matrix ─────────────────────
dtm_tfidf <- DocumentTermMatrix(
  corpus,
  control = list(
    weighting  = function(x) weightTfIdf(x, normalize = TRUE),
    bounds     = list(global = c(5, Inf))   
  )
)

cat("DTM dimensions (docs x terms):", dim(dtm_tfidf), "\n")

# Keep only the top-N most frequent terms to limit dimensionality
top_n_terms <- 500
freq   <- slam::col_sums(dtm_tfidf)
top_terms <- names(sort(freq, decreasing = TRUE)[1:min(top_n_terms, length(freq))])
dtm_reduced <- dtm_tfidf[, top_terms]

# Convert to plain matrix
feature_matrix <- as.matrix(dtm_reduced)
feature_df     <- as.data.frame(feature_matrix)

# Make column names syntactically valid
colnames(feature_df) <- make.names(colnames(feature_df))

# Attach label
feature_df$label <- df$label

# ── 5. Train / Test split  (80 / 20) ─────────────────────────
train_idx  <- createDataPartition(feature_df$label, p = 0.80, list = FALSE)
train_data <- feature_df[ train_idx, ]
test_data  <- feature_df[-train_idx, ]

cat("\nTrain rows:", nrow(train_data), " | Test rows:", nrow(test_data), "\n")

# ── 6. Neural Network — nnet (single hidden layer) ───────────
cat("\nTraining neural network …\n")

# nnet params
hidden_units <- 10   
max_iters    <- 200   

nn_model <- nnet(
  label ~ .,
  data       = train_data,
  size       = hidden_units,
  maxit      = max_iters,
  decay      = 1e-4,     
  MaxNWts    = 100000,  
  trace      = FALSE
)

cat("Neural network trained.\n")
cat("Final value:", nn_model$value, "\n")

# ── 7. Predictions & Confusion Matrix ────────────────────────
pred_class <- predict(nn_model, newdata = test_data, type = "class")
pred_class <- factor(pred_class, levels = levels(test_data$label))

cat("\n========== CONFUSION MATRIX ==========\n")
cm <- confusionMatrix(pred_class, test_data$label, positive = "FAKE")
print(cm)

# ── 8. Detailed metrics ───────────────────────────────────────
cat("\n========== KEY METRICS ==========\n")
cat(sprintf("Accuracy   : %.4f\n", cm$overall["Accuracy"]))
cat(sprintf("Kappa      : %.4f\n", cm$overall["Kappa"]))
cat(sprintf("Sensitivity: %.4f\n", cm$byClass["Sensitivity"]))
cat(sprintf("Specificity: %.4f\n", cm$byClass["Specificity"]))
cat(sprintf("Precision  : %.4f\n", cm$byClass["Pos Pred Value"]))
cat(sprintf("F1-Score   : %.4f\n", cm$byClass["F1"]))

# ── 9. Plot confusion matrix ──────────────────────────────────
cm_table <- as.data.frame(cm$table)
colnames(cm_table) <- c("Predicted", "Actual", "Freq")

ggplot(cm_table, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile(color = "white", linewidth = 1) +
  geom_text(aes(label = Freq), size = 8, fontface = "bold") +
  scale_fill_gradient(low = "#d6eaf8", high = "#1a5276") +
  labs(
    title    = "Confusion Matrix — Fake News Neural Network",
    subtitle = sprintf("Accuracy: %.2f%%  |  F1: %.4f",
                       cm$overall["Accuracy"] * 100,
                       cm$byClass["F1"]),
    x = "Actual Label",
    y = "Predicted Label"
  ) +
  theme_minimal(base_size = 14) +
  theme(legend.position = "right")

ggsave("confusion_matrix.png", width = 6, height = 5, dpi = 150)
cat("\nConfusion matrix plot saved as confusion_matrix.png\n")

# ── 10. Word clouds  ───────────────────────────
# Most common words in FAKE vs TRUE news
fake_words <- df %>%
  filter(label == "FAKE") %>%
  unnest_tokens(word, content) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 100)

true_words <- df %>%
  filter(label == "TRUE") %>%
  unnest_tokens(word, content) %>%
  anti_join(stop_words, by = "word") %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 100)

png("wordcloud_fake.png", width = 600, height = 600)
wordcloud(fake_words$word, fake_words$n,
          max.words = 80, random.order = FALSE,
          colors = brewer.pal(8, "Reds"), scale = c(4, 0.5))
title("Top Words — FAKE News")
dev.off()

png("wordcloud_true.png", width = 600, height = 600)
wordcloud(true_words$word, true_words$n,
          max.words = 80, random.order = FALSE,
          colors = brewer.pal(8, "Blues"), scale = c(4, 0.5))
title("Top Words — TRUE News")
dev.off()

cat("Word cloud images saved.\n")

# ── 11. Save model for Shiny app ─────────────────────────────
# We save the model, top_terms, and the IDF weights so the app 
# can reproduce the exact TF-IDF features for new input.
cat("\nSaving model and terms for the demo app...\n")

# Calculate IDF for the top_terms
# Document frequency for each term
df_counts <- slam::col_sums(dtm_reduced > 0)
n_docs    <- nrow(dtm_reduced)
idf_weights <- log2(n_docs / df_counts)

saveRDS(list(
  model       = nn_model,
  top_terms   = top_terms,
  idf_weights = idf_weights
), "model_data.rds")

cat("Model data saved to model_data.rds ✓\n")
cat("\nDone! ✓\n")
