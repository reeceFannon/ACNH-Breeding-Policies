suppressPackageStartupMessages({
  library(tidyverse)
  library(readxl)
  library(purrr)
})
source("scripts/utils.R")

# Create Long data from ACNH flowers
inPath = "data/ACNH_ACNL Flower Genes.xlsx"
outPath = "data/ACNH_flower_genetics.csv"
if(!file.exists(outPath))
{
  sheets = c("Roses", "Cosmos", "Lilies", "Pansies", "Hyacinths", "Tulips", "Mums", "Windflowers")
  flowers = map_dfr(sheets, ~read_excel(inPath, sheet = .x) %>% select(Genotype, Color) %>% mutate(flower = .x))
  colnames(flowers) = c("genotype", "phenotype", "flower")
  flowers = flowers %>%
    mutate(flower = str_remove_all(str_to_lower(flower), "s$"),
           flower = if_else(str_equal(flower, "lilie"), "lily", flower),
           flower = if_else(str_equal(flower, "pansie"), "pansy", flower),
           phenotype = str_to_lower(str_trim(str_remove_all(phenotype, "\\(seed\\)")))) %>%
    relocate(3, .before = 1)
  
  write_csv(flowers, outPath)
}else
{
  cat("Genetics path already exists. Skipping creation.")
  flowers = read_csv(outPath)
}


# Precompute all offspring probabilities from breeding pairs of flowers together
outPath = "data/breeding_transitions.csv"
if(!file.exists(outPath))
{
  species_list = unique(flowers$flower)
  all_transitions = map_dfr(species_list, function(sp) {
    df_sp = flowers %>% filter(flower == sp)
    genos = df_sp$genotype %>% unique() %>% sort()
    
    # Unordered parent pairs (A,B) with A <= B (lexicographically)
    parent_pairs = expand.grid(parent1 = genos, parent2 = genos, stringsAsFactors = FALSE) %>%
      as_tibble() %>%
      filter(parent1 <= parent2)  # this enforces symmetry: A×B == B×A, since strings are coded as numbers
    
    # For each (parent1, parent2), compute genotype distribution
    map2_dfr(parent_pairs$parent1, parent_pairs$parent2, ~{
      genes = punnettSquare(.x, .y, df_sp)
      dist = getPunnettDistribution(genes, isGeno = TRUE)
      dist %>%
        transmute(species = sp,
                  parent1 = .x,
                  parent2 = .y,
                  offspring = offspring,
                  prob = prop)
    }
    )
  })
  
  write_csv(all_transitions, outPath)
}else
{
  cat("Transition path already exists. Skipping creation.")
}