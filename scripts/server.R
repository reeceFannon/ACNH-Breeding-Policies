library(shiny)
library(tidyverse)

server <- function(input, output, session) {
  
  # Add label + id for selection
  flowers_labeled <- reactive({
    flowers %>%
      mutate(
        id    = row_number(),
        label = sprintf("%s | %s | %s",
                        flower, phenotype, genotype)
      )
  })
  
  # Apply species filter
  filtered_flowers <- reactive({
    df <- flowers_labeled()
    req(input$species)
    
    if (input$species == "all") {
      df
    } else {
      df %>% filter(flower == input$species)
    }
  })
  
  # Update the multi-select choices whenever species changes
  observeEvent(filtered_flowers(), {
    df <- filtered_flowers()
    choices_vec <- setNames(df$id, df$label)  # values = id, labels = label
    
    updateSelectizeInput(
      session,
      inputId  = "parents",
      choices  = choices_vec,
      server   = TRUE,
      selected = character(0)
    )
  })
  
  # Get full rows for the selected parents
  selected_parents <- reactive({
    req(input$parents)
    df_all <- flowers_labeled()
    ids <- as.integer(input$parents)
    df_all %>% filter(id %in% ids)
  })
  
  # Warning / status text
  output$warning <- renderUI({
    parents <- selected_parents()
    n_sel   <- length(input$parents)
    
    if (n_sel == 0) {
      HTML("<p style='color:#666;'>Choose two flowers to see a Punnett square.</p>")
    } else if (n_sel < 2) {
      HTML("<p style='color:red;'>Select one more flower.</p>")
    } else {
      # Keep parents in the order they appear in the selectize
      df_all <- flowers_labeled()
      ids <- as.integer(input$parents)
      parents <- df_all %>% filter(id %in% ids)
      parents <- parents[match(ids, parents$id), ]
      parents <- parents[1:2, ]
      
      if (length(unique(parents$flower)) > 1) {
        HTML("<p style='color:red;'>Selected flowers are different species. Please choose two of the same species.</p>")
      } else if (n_sel > 2) {
        HTML("<p style='color:#b35;'>More than two parents selected; using the first two in the list.</p>")
      } else {
        # No warning
        HTML("")
      }
    }
  })
  
  # Punnett square plot
  punnett_data <- reactive({
    parents <- selected_parents()
    n_sel   <- length(input$parents)
    if (n_sel < 2) return(NULL)
    
    # Align and take first two
    df_all <- flowers_labeled()
    ids <- as.integer(input$parents)
    parents <- df_all %>% filter(id %in% ids)
    parents <- parents[match(ids, parents$id), ]
    parents <- parents[1:2, ]
    
    # If different species, don't plot (warning handles message)
    if (length(unique(parents$flower)) > 1) return(NULL)
    
    species <- parents$flower[1]
    
    df_species <- flowers %>%
      filter(flower == species) %>%
      select(genotype, phenotype)
    
    punnetSquare(parent1 = parents$genotype[1], parent2 = parents$genotype[2], df = df_species)
  })
  
  # Plot
  output$punnett_plot <- renderPlot({
    genes <- punnett_data()
    req(genes)
    plotPunnetSquare(genes)
  })
  
  output$geno_table <- renderTable({
    genes <- punnett_data()
    req(genes)
    
    genes %>%
      count(offspring, name = "count") %>%
      mutate(prop = count / sum(count)) %>%
      arrange(desc(prop))
  }, digits = 2)
  
  # Phenotype-only distribution
  output$pheno_table <- renderTable({
    genes <- punnett_data()
    req(genes)
    
    genes %>%
      count(phenotype, name = "count") %>%
      mutate(prop = count / sum(count)) %>%
      arrange(desc(prop))
  }, digits = 2)
}
