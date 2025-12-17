# scripts/server.R
library(shiny)
library(tidyverse)
library(reticulate)

# Point reticulate at the Python env you use in Colab / locally, if needed:
# use_virtualenv("your-venv-name")
# use_condaenv("your-conda-env-name")

# Import the Python bridge
source_python("scripts/python_wrapper.py")

server <- function(input, output, session) {

  ## ---- Existing Punnett-square logic ----

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

  # Punnett square underlying data
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

    punnettSquare(
      parent1 = parents$genotype[1],
      parent2 = parents$genotype[2],
      df      = df_species
    )
  })

  # Plot
  output$punnett_plot <- renderPlot({
    genes <- punnett_data()
    req(genes)
    plotPunnettSquare(genes)
  })

  # Genotype distribution
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


  ## ---- New: Episode planner logic (Tab 2) ----

  # Helpers for parsing episode planner inputs
  parse_targets <- function(txt) {
    if (is.null(txt) || txt == "") return(NULL)
    lines <- strsplit(txt, "\n")[[1]]
    lines <- trimws(lines)
    lines <- lines[lines != ""]
    if (length(lines) == 0) return(NULL)

    lapply(lines, function(line) {
      parts <- strsplit(line, ",")[[1]]
      parts <- trimws(parts)
      parts[parts != ""]
    })
  }

  parse_root_state <- function(txt) {
    if (is.null(txt) || txt == "") return(character(0))
    parts <- strsplit(txt, ",")[[1]]
    parts <- trimws(parts)
    parts[parts != ""]
  }

  episode_result <- eventReactive(input$ep_run, {
    targets_list   <- parse_targets(input$ep_targets_raw)
    root_state_vec <- parse_root_state(input$ep_root_state_raw)

    # Call the Python wrapper
    run_episode_for_shiny(
      species                      = input$ep_species,
      targets                      = targets_list,
      root_state                   = root_state_vec,
      root_n_simulations           = input$ep_root_n_sim,
      max_episode_steps            = input$ep_max_steps,
      root_max_rollout_depth       = input$ep_root_depth,
      c                            = input$ep_c_ucb,
      min_n_simulations            = input$ep_min_n_sim,
      max_simulations_scale_factor = input$ep_max_sim_scale,
      min_depth_floor              = input$ep_min_depth_floor,
      seed                         = input$seed
    )
  })

  output$ep_steps_table <- renderTable({
    res <- episode_result()
    if (is.null(res)) return(NULL)

    steps <- res$steps
    if (length(steps) == 0) return(NULL)

    df <- bind_rows(steps)

    df %>%
      mutate(
        available_before = vapply(
          available_before,
          function(x) paste(x, collapse = ", "),
          FUN.VALUE = character(1)
        ),
        new_genotypes = vapply(
          new_genotypes,
          function(x) paste(x, collapse = ", "),
          FUN.VALUE = character(1)
        )
      ) %>%
      arrange(step)
  })

  output$ep_waves_table <- renderTable({
    res <- episode_result()
    if (is.null(res)) return(NULL)

    waves <- res$waves
    if (length(waves) == 0) return(NULL)

    df <- bind_rows(waves)

    df %>%
      mutate(
        steps = vapply(
          steps,
          function(s) paste(as.integer(s), collapse = ", "),
          FUN.VALUE = character(1)
        )
      ) %>%
      arrange(wave)
  })

  output$ep_final_state_text <- renderPrint({
    res <- episode_result()
    if (is.null(res)) return(invisible(NULL))

    cat("Final genotypes:\n")
    cat(paste(res$final_state, collapse = ", "), "\n")
  })

  output$ep_summary_text <- renderPrint({
    res <- episode_result()
    if (is.null(res)) return(invisible(NULL))
    str(res$summary)
  })
}

