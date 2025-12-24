library(tidyverse)
library(purrr)

seed_map = list(rose = c("rryyWwss" = "white", "rrYYWWss" = "yellow", "RRyyWWSs" = "red"),
                cosmo = c("rryySs" = "white", "rrYYSs" = "yellow", "RRyyss" = "red"),
                lily = c("rryySS" = "white", "rrYYss" = "yellow", "RRyySs" = "red"),
                pansy = c("rryyWw" = "white", "rrYYWW" = "yellow", "RRyyWW" = "red"),
                hyacinth = c("rryyWw" = "white", "rrYYWW" = "yellow", "RRyyWw" = "red"),
                tulip = c("rryySs" = "white", "rrYYss" = "yellow", "RRyySs" = "red"),
                mum = c("rryyWw" = "white", "rrYYWW" = "yellow", "RRyyWW" = "red"),
                windflower = c("rrooWw" = "white", "rrOOWW" = "orange", "RRooWW" = "red"))

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
      mutate(prop = paste0(100*round(count/sum(count), 3), "%")) %>%
      arrange(desc(count))
  }
  else
  {
    df %>%
      count(phenotype, name = "count") %>%
      mutate(prop = paste0(100*round(count/sum(count), 3), "%")) %>%
      arrange(desc(count))
  }
}

img_filename = function(species, phenotype, genotype)
{
  if (genotype %in% names(seed_map[[species]]))
  {
    seed_color = unname(seed_map[[species]][genotype])
    return(paste0("seed_", seed_color, ".png"))
  }
  paste0(species, "_", phenotype, ".png")
}

build_filtered_df = function(species)
{
  html = "<span class='picker-row'>
    <img src='/imgs/%s' class='picker-img'>
    <span class='picker-text'>%s</span>
    </span>"
  
  flowers %>%
    filter(flower == species) %>%
    distinct(flower, phenotype, genotype) %>%
    mutate(label = paste(flower, phenotype, genotype, sep = " | "),
           img_file = mapply(img_filename, flower, phenotype, genotype),
           img_html = sprintf(html, img_file, label)) %>%
    arrange(phenotype, genotype)
}

get_transition_df = function(sp, p1, p2)
{
  transitions %>% filter(species == sp, (parent1 == p1 & parent2 == p2) | (parent1 == p2 & parent2 == p1))
}

correct_order = function(sp, p1, p2, df) any(df$species == sp & df$parent1 == p1 & df$parent2 == p2)

construct_action = function(sp, p1, p2, df)
{
  get_transition_df(sp, p1, p2) %>% select(-species) %>%
    left_join(df %>% rename(parent1 = genotype,
                            parent1_pheno = phenotype,
                            parent1_label = label,
                            parent1_img_file = img_file,
                            parent1_img_html = img_html), by = "parent1") %>%
    left_join(df %>% rename(parent2 = genotype,
                            parent2_pheno = phenotype,
                            parent2_label = label,
                            parent2_img_file = img_file,
                            parent2_img_html = img_html), by = "parent2") %>%
    left_join(df %>% rename(offspring = genotype,
                            offspring_pheno = phenotype,
                            offspring_label = label,
                            offspring_img_file = img_file,
                            offspring_img_html = img_html), by = "offspring") %>% 
    nest(transitions = c(offspring, offspring_pheno, prob, offspring_label, offspring_img_file, offspring_img_html))
}

build_policy_plan = function(species, ls)
{
  lookup = build_filtered_df(species) %>% select(-flower)
  
  waves_out = lapply(ls$waves, function(w)
  {
    wave_idx = w$wave
    actions_df = tibble(parent1 = vapply(w$actions, function(a) a$parent1, character(1)),
                        parent2 = vapply(w$actions, function(a) a$parent2, character(1))) %>%
      mutate(pair_key = vapply(seq_len(dplyr::n()), function(i) paste(sort(c(parent1[i], parent2[i])), collapse = "|"), character(1))) %>%
      distinct(pair_key, .keep_all = TRUE) %>%
      select(-pair_key)
    
    actions_out = lapply(seq_len(nrow(actions_df)), function(a)
    {
      p1 = actions_df$parent1[a]
      p2 = actions_df$parent2[a]
      
      trans_df = construct_action(species, p1, p2, lookup)
      trans_df
    })
    list(wave = wave_idx, actions = actions_out)
  })
  list(waves = waves_out)
}

wave_parent_genos = function(wave_actions)
{
  df = bind_rows(wave_actions)
  unique(c(df$parent1, df$parent2))
}

add_keep_flags = function(plan, targets)
{
  waves = plan$waves
  n = length(waves)
  if (n <= 1) return(plan)
  
  targets = unique(unlist(targets, use.names = FALSE))
  parents_by_wave = lapply(waves, function(w) wave_parent_genos(w$actions))
  
  future_parents = vector("list", n)
  future_parents[[n]] = character(0)
  if(n >= 2)
  {
    for(i in (n - 1):1) {future_parents[[i]] = unique(c(future_parents[[i + 1]], parents_by_wave[[i + 1]]))}
  }
  
  for (i in seq_len(n)) 
  {
    keeps = unique(c(future_parents[[i]], targets))
    waves[[i]]$actions = lapply(waves[[i]]$actions, function(action_tbl) 
    {
      action_tbl %>%
        mutate(transitions = map(transitions, function(tr) {tr %>% mutate(keep = offspring %in% keeps)}))
    })
  }
  
  plan$waves = waves
  plan
}

as_html = function(x) 
{
  if(is.null(x) || length(x) == 0) return(tags$span("(missing)"))
  HTML(x[[1]])
}

prob_fmt = function(p) sprintf("%.3f", as.numeric(p))

keep_icon = function(keep)
{
  if(isTRUE(keep)) {tags$span(class = "keep-icon", icon("ok", lib = "glyphicon"))} 
  else {tags$span(class = "discard-icon", icon("trash", lib = "glyphicon"))}
}

render_action = function(action_tbl)
{
  p1_html = action_tbl$parent1_img_html
  p2_html = action_tbl$parent2_img_html
  trans = action_tbl$transitions[[1]]
  
  off_rows = lapply(seq_len(nrow(trans)), function(i) 
  {
    tags$div(class = "offspring-row",
             tags$div(class = "offspring-prob", prob_fmt(trans$prob[i])),
             tags$div(class = "offspring-mid", HTML(trans$offspring_img_html[i])),
             keep_icon(trans$keep[i]))
  })
  
  tags$div(class = "action-card",
           tags$div(class = "action-grid",
                    tags$div(class = "parents-stack",
                             tags$div(class = "parent-row", HTML(p1_html[[1]])),
                             tags$div(class = "parent-sep", HTML("&times;")),
                             tags$div(class = "parent-row", HTML(p2_html[[1]]))),
                    tags$div(class = "offspring-stack", off_rows)))
}

render_wave = function(wave_obj)
{
  wave_idx = wave_obj$wave
  actions = wave_obj$actions
  action_cards = lapply(actions, render_action)
  
  tags$div(class = "wave-col",
           tags$div(class = "wave-title", paste0("Step ", wave_idx)),
           action_cards)
}
