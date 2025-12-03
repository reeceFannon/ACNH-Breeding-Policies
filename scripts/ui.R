library(shiny)
library(shinythemes)
library(shinydashboard)
library(tidyverse)

# Dropdown choices for species
species_choices <- c("All species" = "all",
                     sort(unique(flowers$flower)))

ui <- fluidPage(themeSelector(),
  titlePanel("ACNH Flower Breeding – Punnett Squares"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput(
        inputId  = "species",
        label    = "Filter by flower species:",
        choices  = species_choices,
        selected = "All species"
      ),
      
      selectizeInput(
        inputId  = "parents",
        label    = "Choose up to two flowers to breed:",
        choices  = NULL,        # filled from server based on species
        multiple = TRUE,
        options  = list(maxItems = 2)
      ),
      
      helpText("Selections are of the form: species | phenotype | genotype")
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
)
