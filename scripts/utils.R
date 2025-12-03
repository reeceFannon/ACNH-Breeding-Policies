library(tidyverse)
library(purrr)

genome_split = function(genotype)
{
  alleles = str_split(genotype, "")[[1]]
  matrix(alleles, ncol = 2, byrow = TRUE)
}

passDownAlleles <- function(alleles)
{
  locus_choices = lapply(seq_len(nrow(alleles)), function(i){alleles[i, ]})
  
  grid = expand.grid(locus_choices, stringsAsFactors = FALSE)
  apply(grid, 1, str_c, collapse = "")
}

cross = function(parent1, parent2)
{
  a1 = str_split(parent1, "")[[1]]
  a2 = str_split(parent2, "")[[1]]
  stopifnot(length(a1) == length(a2))
  
  locus_pairs = map2_chr(a1, a2, ~{
    alleles = c(.x, .y)
    idx = order(alleles == str_to_upper(alleles), decreasing = TRUE)
    str_c(alleles[idx], collapse = "")
  })
  
  str_c(locus_pairs, collapse = "")
}

punnettSquare = function(parent1, parent2, df)
{
  g1 = passDownAlleles(genome_split(parent1))
  g2 = passDownAlleles(genome_split(parent2))
  n1 = length(g1)
  n2 = length(g2)
  
  genes = expand.grid(row = seq_len(n1), col = seq_len(n2)) %>%
    as_tibble() %>%
    mutate(gamete1 = g1[row],
           gamete2 = g2[col],
           offspring = mapply(cross, gamete1, gamete2)) %>%
    left_join(df, by = c("offspring" = "genotype"))
  
  return(genes)
}
 
plotPunnettSquare = function(df) 
{
  g1 = df %>% arrange(row) %>% distinct(row, gamete1) %>% pull(gamete1)
  g2 = df %>% arrange(col) %>% distinct(col, gamete2) %>% pull(gamete2)
  
  ggplot(df, aes(x = col, y = row, fill = phenotype)) +
    geom_tile(color = "black") +
    scale_x_continuous(breaks = seq_along(g2), labels = g2, expand = expansion(mult = 0)) +
    scale_y_reverse(breaks = seq_along(g1), labels = g1, expand = expansion(mult = 0)) +
    scale_fill_identity() +
    labs(x = "Parent 2 gamete", y = "Parent 1 gamete", fill = "Phenotype") +
    coord_equal() +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1),
          panel.grid = element_blank())
}

getPunnettDistribution = function(df, isGeno = FALSE)
{
  if(isGeno)
  {
    df %>%
      count(offspring, name = "count") %>%
      mutate(prop = count / sum(count)) %>%
      arrange(desc(prop))
  }
  else
  {
    df %>%
      count(phenotype, name = "count") %>%
      mutate(prop = count / sum(count)) %>%
      arrange(desc(prop))
  }
}
