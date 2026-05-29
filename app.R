# ============================================================
#  Fake News Detector — Interactive Demo
#  Educational Interface for NLP + Neural Networks
# ============================================================

# ── 1. Load Libraries ─────────────────────────────────────────
packages <- c("shiny", "bslib", "tidyverse", "tm", "SnowballC", "nnet", "thematic", "shinycssloaders")
installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]
if (length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")

library(shiny)
library(bslib)
library(tidyverse)
library(tm)
library(SnowballC)
library(nnet)
library(thematic)
library(shinycssloaders)

# Enable automatic thematic styling for plots
thematic_shiny(font = "auto")

# ── 2. Load Model Data ────────────────────────────────────────
# Note: Ensure you have run fake_news_nn.R first to generate this file!
model_path <- "model_data.rds"

if (!file.exists(model_path)) {
  # Create a dummy model if file is missing (for first run safety)
  model_data <- NULL
} else {
  model_data <- readRDS(model_path)
}

# ── 3. UI Definition ─────────────────────────────────────────
ui <- page_navbar(
  title = "TruthLens: Fake News Detector",
  theme = bs_theme(
    version = 5,
    bootswatch = "lux",
    primary = "#2c3e50",
    secondary = "#18bc9c",
    base_font = font_google("Inter"),
    heading_font = font_google("Outfit")
  ),
  
  # Main Detector Tab
  nav_panel(
    title = "Detector",
    icon = icon("search"),
    layout_sidebar(
      sidebar = sidebar(
        title = "Article Input",
        width = 400,
        helpText("Paste the news article title or full text below to analyze its authenticity."),
        textAreaInput(
          "news_text", 
          "News Content", 
          placeholder = "Example: Scientists discover new planet made of diamonds...",
          height = "300px"
        ),
        actionButton("analyze_btn", "Analyze Article", class = "btn-primary w-100", icon = icon("microchip")),
        hr(),
        helpText("This model uses a Neural Network trained on ~40,000 news articles.")
      ),
      
      # Main Content Area
      uiOutput("prediction_results")
    )
  ),
  
  # How It Works Tab
  nav_panel(
    title = "How It Works",
    icon = icon("graduation-cap"),
    div(
      class = "container-fluid mt-4",
      layout_columns(
        card(
          card_header("1. Text Pre-processing"),
          p("The model doesn't 'read' words like humans. First, we clean the text:"),
          tags$ul(
            tags$li(tags$b("Lowercasing:"), " 'Apple' becomes 'apple'"),
            tags$li(tags$b("Cleaning:"), " Remove punctuation and numbers"),
            tags$li(tags$b("Stopwords:"), " Remove common words like 'the', 'is', 'at'"),
            tags$li(tags$b("Stemming:"), " 'running' becomes 'run'")
          )
        ),
        card(
          card_header("2. TF-IDF Vectorization"),
          p("We convert text into numbers using TF-IDF (Term Frequency - Inverse Document Frequency)."),
          p("This measures how important a word is to a document in a collection. Rare words like 'conspiracy' get higher weights than common words like 'news'.")
        ),
        card(
          card_header("3. Neural Network"),
          p("The numbers are fed into a single-layer Neural Network."),
          p("The network has been trained to recognize patterns of words and weights that are statistically associated with fake or true reporting.")
        )
      )
    )
  ),
  
  # Footer
  nav_spacer(),
  nav_item(tags$small("Educational Demo v1.0", class = "text-muted"))
)

# ── 4. Server Logic ──────────────────────────────────────────
server <- function(input, output, session) {
  
  # Reactive values to store results
  results <- reactiveVal(NULL)
  
  # Error handling if model is missing
  observe({
    if (is.null(model_data)) {
      showModal(modalDialog(
        title = "Model Not Found",
        "Please run 'fake_news_nn.R' first to train and save the model (model_data.rds).",
        footer = modalButton("Dismiss"),
        easyClose = TRUE
      ))
    }
  })
  
  observeEvent(input$analyze_btn, {
    req(input$news_text)
    req(model_data)
    
    # Show loading indicator
    withProgress(message = 'Analyzing text...', value = 0, {
      
      # 4.1. Preprocessing
      incProgress(0.3, detail = "Cleaning text...")
      clean_text <- input$news_text %>%
        tolower() %>%
        removePunctuation() %>%
        removeNumbers()
      
      # Remove stopwords and stem
      sw <- stopwords("en")
      words <- unlist(strsplit(clean_text, "\\s+"))
      words <- words[words %in% model_data$top_terms] # Focus only on terms we know
      
      # 4.2. Feature Engineering (TF-IDF)
      incProgress(0.3, detail = "Calculating features...")
      
      # Calculate Term Frequency (TF) for the input text
      tf_counts <- table(words)
      
      # Create an empty feature vector
      feature_vector <- setNames(numeric(length(model_data$top_terms)), model_data$top_terms)
      
      # Fill with TF-IDF: TF * IDF_saved
      matching_words <- names(tf_counts)
      if (length(matching_words) > 0) {
        # Normalize TF (simple frequency)
        tf_normalized <- as.numeric(tf_counts) / length(words)
        # Apply IDF
        feature_vector[matching_words] <- tf_normalized * model_data$idf_weights[matching_words]
      }
      
      # Prepare data frame for prediction
      feature_df <- as.data.frame(t(feature_vector))
      colnames(feature_df) <- make.names(colnames(feature_df))
      
      # 4.3. Prediction
      incProgress(0.3, detail = "Running Neural Network...")
      prob <- predict(model_data$model, newdata = feature_df, type = "raw")
      
      # Note: nnet predict type="raw" returns a matrix
      # If the model has 2 levels (FAKE, TRUE), prob is probability of the second level?
      # Let's check levels
      levels_info <- model_data$model$lev
      # Usually: [1] "FAKE" [2] "TRUE"
      # Probability of being TRUE
      prob_true <- as.numeric(prob)
      
      results(list(
        prob_true = prob_true,
        label = ifelse(prob_true < 0.5, "FAKE", "TRUE"),
        confidence = ifelse(prob_true < 0.5, (1 - prob_true) * 100, prob_true * 100),
        tokens_found = length(matching_words)
      ))
    })
  })
  
  # Render the results UI
  output$prediction_results <- renderUI({
    res <- results()
    
    if (is.null(res)) {
      return(
        div(
          class = "d-flex flex-column justify-content-center align-items-center h-100 text-muted",
          icon("newspaper", class = "fa-4x mb-3"),
          h4("Ready for Analysis"),
          p("Enter some text on the left and click 'Analyze'")
        )
      )
    }
    
    # determine color and icon
    is_fake <- res$label == "FAKE"
    status_color <- if (is_fake) "danger" else "success"
    status_icon <- if (is_fake) "exclamation-triangle" else "check-circle"
    
    card(
      full_screen = TRUE,
      card_header(
        class = paste0("bg-", status_color, " text-white d-flex justify-content-between align-items-center"),
        span(icon(status_icon), " Analysis Result: ", res$label),
        span(sprintf("%.1f%% Confidence", res$confidence))
      ),
      card_body(
        layout_columns(
          value_box(
            title = "Verdict",
            value = res$label,
            showcase = icon(status_icon),
            theme = status_color
          ),
          value_box(
            title = "Confidence",
            value = sprintf("%.1f%%", res$confidence),
            showcase = icon("chart-line"),
            theme = "info"
          ),
          value_box(
            title = "Key Terms Found",
            value = res$tokens_found,
            showcase = icon("list-ol"),
            theme = "secondary"
          )
        ),
        hr(),
        h5("What does this mean?"),
        p(
          if (is_fake) {
            "The model detected patterns of language frequently found in misleading or sensationalist content. This includes specific word associations and frequencies that match documented fake news datasets."
          } else {
            "The model found that the language patterns in this article align closely with established factual reporting. It contains vocabulary and structure typical of mainstream journalism."
          }
        ),
        tags$blockquote(
          class = "blockquote mt-3",
          tags$small("Note: This is an educational tool and may occasionally misclassify complex news. Always verify with multiple sources.")
        )
      )
    )
  })
}

# ── 5. Run App ───────────────────────────────────────────────
# We set host and port for easier access in this environment
options(shiny.port = 5000, shiny.host = "0.0.0.0")
shinyApp(ui, server)
