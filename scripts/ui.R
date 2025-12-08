# scripts/ui.R
library(shiny)
library(shinythemes)
library(shinydashboard)
library(tidyverse)

# Dropdown choices for species (uses global `flowers` prepared elsewhere)
species_choices <- c("All species" = "all",
                     sort(unique(flowers$flower)))

ui <- fluidPage(
  themeSelector(),
  titlePanel("ACNH Flower Breeding"),

  tabsetPanel(
    id = "main_tabs",

    ## ---- Tab 1: existing Punnett square calculator ----
    tabPanel(
      title = "Punnett Squares",
      sidebarLayout(
        sidebarPanel(
          selectInput(
            inputId  = "species",
            label    = "Species",
            choices  = species_choices,
            selected = "all"
          ),
          selectizeInput(
            inputId  = "parents",
            label    = "Select parent flowers",
            choices  = NULL,          # populated on server side
            multiple = TRUE,
            options  = list(
              placeholder = "Choose up to two flowers...",
              maxItems    = 4          # server side will warn/use first two
            )
          )
        ),
        mainPanel(
          uiOutput("warning"),
          plotOutput("punnett_plot", height = "650px"),
          tags$hr(),
          h4("Offspring Genotypes"),
          tableOutput("geno_table"),
          tags$hr(),
          h4("Offspring Phenotypes"),
          tableOutput("pheno_table")
        )
      )
    ),

    ## ---- Tab 2: Episode planner (Python + DAG) ----
    tabPanel(
      title = "Episode Planner",
      sidebarLayout(
        sidebarPanel(
          textInput("ep_species", "Species", value = "rose"),

          tags$label("Target genotype groups (one group per line, comma-separated)"),
          tags$small("Example: 'RRYY, RrYY' on first line; 'rryy' on second line."),
          textAreaInput(
            "ep_targets_raw",
            NULL,
            value = "",
            rows = 3,
            placeholder = "RRYY, RrYY\nrryy"
          ),

          textInput(
            "ep_root_state_raw",
            "Initial genotypes (comma-separated)",
            value = "seed_red, seed_yellow"
          ),

          numericInput("ep_root_n_sim",      "Root n_simulations",            value = 1000, min = 10),
          numericInput("ep_max_steps",       "Max episode steps",             value = 50,   min = 1),
          numericInput("ep_root_depth",      "Root max rollout depth",        value = 20,   min = 1),
          numericInput("ep_n_workers",       "Number of workers",             value = 4,    min = 1),
          numericInput("ep_c_ucb",           "Exploration constant c",        value = sqrt(2), step = 0.1),
          numericInput("ep_min_n_sim",       "Min n_simulations per step",    value = 100,  min = 1),
          numericInput("ep_max_sim_scale",   "Max simulations scale factor",  value = 0,    step = 0.1),
          numericInput("ep_min_depth_floor", "Min depth floor",               value = 10,   min = 1),

          actionButton("ep_run", "Run planner")
        ),
        mainPanel(
          h4("Step-by-step decisions"),
          tableOutput("ep_steps_table"),
          tags$hr(),
          h4("Parallel waves (simultaneous batches)"),
          tableOutput("ep_waves_table"),
          tags$hr(),
          h4("Final genotypes"),
          verbatimTextOutput("ep_final_state_text"),
          tags$hr(),
          h4("Episode summary"),
          verbatimTextOutput("ep_summary_text")
        )
      )
    )
  )
)
