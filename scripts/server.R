library(shiny)
library(tidyverse)
library(reticulate)

addResourcePath("imgs", "imgs") # import image path
source_python("scripts/python_wrapper.py") # Import the Python bridge

server = function(input, output, session)
{
  #################################################################################################################
  ####################################### Breeding Tab ############################################################
  #################################################################################################################
  
  ####################################### Input Widgets ###########################################################
  
  seed_map = list(rose = c("rryyWwss" = "white", "rrYYWWss" = "yellow", "RRyyWWSs" = "red"),
                  cosmo = c("rryySs" = "white", "rrYYSs" = "yellow", "RRyyss" = "red"),
                  lily = c("rryySS" = "white", "rrYYss" = "yellow", "RRyySs" = "red"),
                  pansy = c("rryyWw" = "white", "rrYYWW" = "yellow", "RRyyWW" = "red"),
                  hyacinth = c("rryyWw" = "white", "rrYYWW" = "yellow", "RRyyWw" = "red"),
                  tulip = c("rryySs" = "white", "rrYYss" = "yellow", "RRyySs" = "red"),
                  mum = c("rryyWw" = "white", "rrYYWW" = "yellow", "RRyyWW" = "red"),
                  windflower = c("rrooWw" = "white", "rrOOWW" = "orange", "RRooWW" = "red"))
  
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
  
  observe({
    species_choices = sort(unique(flowers$flower))
    updateSelectInput(session, "species",
                      choices  = species_choices,
                      selected = "rose")
    })
  
  observeEvent(input$species, {
    req(input$species)
    df = build_filtered_df(input$species)
    choices = setNames(df$genotype, df$label)

    updatePickerInput(session,
                      inputId = "parent1",
                      choices = choices,
                      choicesOpt = list(content = df$img_html),
                      selected = character(0))
    
    updatePickerInput(session,
                      inputId = "parent2",
                      choices = choices,
                      choicesOpt = list(content = df$img_html),
                      selected = character(0))
  }, ignoreInit = FALSE)
  
  ######################################## Punnett Square Plot ###################################################
  
  punnett_df = reactive(
  {
    req(input$species)
    req(input$parent1, input$parent2)
    
    punnettSquare(input$parent1, input$parent2, flowers %>% filter(flower == input$species))
  })
  
  output$punnettPlot = renderPlot(
  {
    df = punnett_df()
    plotPunnettSquare(df)
  })
  
  ################################## Distribution Tables #########################################################
  
  output$genoTable = renderTable(
  {
    df = punnett_df()
    req(df)
    getPunnettDistribution(df, isGeno = TRUE)
  }, striped = TRUE, bordered = TRUE, spacing = "s")
  
  output$phenoTable = renderTable(
  {
    df = punnett_df()
    req(df)
    getPunnettDistribution(df, isGeno = FALSE)
  }, striped = TRUE, bordered = TRUE, spacing = "s")
  
  #################################################################################################################
  ####################################### Planning Tab ############################################################
  #################################################################################################################
  
  ####################################### Input Widgets ###########################################################
  
  observe({
    species_choices = sort(unique(flowers$flower))
    updateSelectInput(session, "planSpecies",
                      choices  = species_choices,
                      selected = "rose")
  })
  
  rv = reactiveValues(n_groups = 0) # How many target groups do we have?
  observeEvent(input$addTargets, {rv$n_groups = rv$n_groups + 1})
  
  observeEvent(input$planSpecies, {
    req(input$planSpecies)
    
    df = build_filtered_df(input$planSpecies)
    choices = setNames(df$genotype, df$label)
    seed_genos = names(seed_map[[input$planSpecies]])
    
    updatePickerInput(session,
                      inputId = "rootState",
                      choices = choices,
                      choicesOpt = list(content = df$img_html),
                      selected = seed_genos)
  }, ignoreInit = FALSE)
  
  output$targetGroups = renderUI(
  {
    req(input$planSpecies)
    n = rv$n_groups
    if (n == 0) return(NULL)
    
    df = build_filtered_df(input$planSpecies)
    choices = setNames(df$genotype, df$label)
    tagList(lapply(seq_len(n), 
                   function(k) 
                   {
                     id = paste0("targetGroup_", k)
                     pickerInput(inputId = id,
                                 label = paste("Target group", k),
                                 choices = choices,
                                 choicesOpt = list(content = df$img_html),
                                 multiple = TRUE,
                                 selected = isolate(input[[id]]) %||% character(0),
                                 options = pickerOptions(liveSearch = TRUE,
                                                         actionsBox = TRUE,
                                                         multipleSeparator = ""))
                   }))
  })
  
  ######################################### Running Planner ##################################################
  
  get_targets = function()
  {
    if (rv$n_groups <= 0) return(list())
    targets = lapply(seq_len(rv$n_groups), function(k) {input[[paste0("targetGroup_", k)]]})
    Filter(function(x) !is.null(x) && length(x) > 0, targets)
  }
  
  plan_warning = reactiveVal(NULL)
  plan_result  = reactiveVal(NULL)
  plan_running = reactiveVal(FALSE)
  
  observeEvent(input$runPlanner, {
    plan_warning(NULL)
    plan_result(NULL)
    
    # ---- validate ----
    targets = get_targets()
    if(length(targets) == 0) {plan_warning("Please add at least one target group and select at least one genotype in it.")}
    if(is.null(input$planSpecies) || input$planSpecies == "") {plan_warning("Please select a species.")}
    if(is.null(input$rootState) || length(input$rootState) == 0) {plan_warning("Please select at least one starting genotype.")}
    if(is.null(input$episodeSteps)) {plan_warning("Missing Episode Steps.")}
    if(is.null(input$rootSimulations)) {plan_warning("Missing starting number of rollouts.")}
    if(is.null(input$minSimulations)) {plan_warning("Missing minimum number of rollouts.")}
    if(is.null(input$simScaleFactor)) {plan_warning("Missing rollout downscale factor.")}
    if(is.null(input$rootDepth)) {plan_warning("Missing rollout search depth.")}
    if(is.null(input$minDepth)) {plan_warning("Missing minimum rollout search depth.")}
    if(is.null(input$exploreFactor)) {plan_warning("Missing the exploration/exploitation factor (c).")}
    
    if(!is.null(plan_warning()))
    {
      print(plan_warning)
      return()
    }
    
    # ---- run ----
    plan_running(TRUE)
    plan_warning("Running planner…")  # lightweight reassurance
    
    res = run_episode_for_shiny(species = input$planSpecies,
                                targets = targets,
                                root_state = input$rootState,
                                root_n_simulations = input$rootSimulations,
                                max_episode_steps = input$episodeSteps,
                                root_max_rollout_depth = input$rootDepth,
                                c = input$exploreFactor,
                                min_n_simulations = input$minSimulations,
                                max_simulations_scale_factor = input$simScaleFactor,
                                min_depth_floor = input$minDepth,
                                seed = input$seed)
    
    plan_result(res)
    plan_warning("Planner finished.")
    plan_running(FALSE)
  })
  
  output$planWarning = renderUI(
  {
    msg = plan_warning()
    if (is.null(msg)) return(NULL)
    
    # choose warning vs success based on content (simple heuristic)
    status = if (grepl("finished", msg, ignore.case = TRUE)) "success" else "danger"
    box(width = 12, status = status, solidHeader = TRUE, title = "Status", msg)
  })
  
  output$planFinalState = renderPrint(
  {
    res = plan_result()
    req(res)
    paste0(c("Final State: ", res$final_state))
  })
  
  output$planOutcome = renderUI(
  {
    res = plan_result()
    req(res)
    success = isTRUE(res$summary$success)
    
    if(success) 
    {
      box(width = 12,
          status = "success",
          solidHeader = TRUE,
          title = "Planner Result",
          h4("Run was successful"),
          p("Target conditions were reached. DAG / breeding recipe will be shown here."))
    } else 
      {
        box(width = 12,
            status = "danger",
            solidHeader = TRUE,
            title = "Planner Result",
            h4("Run was not successful"),
            p("The planner did not reach the target conditions."))
      }
  })
  
  ########################################## Download Run ###################################################
  
  output$downloadBttn = renderUI(
  {
      res = plan_result()
      req(res)
      success = isTRUE(res$summary$success)
      
      if(!success) return(NULL)
      box(width = 12,
          status = "info",
          solidHeader = TRUE,
          actionBttn(inputId = "download",
                     label = "Download Policy",
                     style = "material-flat",
                     color = "royal",
                     size = "md",
                     block = TRUE,
                     icon = icon("save", lib = "glyphicon")))
  })
  
  get_transition_df = function(sp, p1, p2)
  {
    transitions %>% 
      filter(species == sp, (parent1 == p1 & parent2 == p2) | (parent1 == p2 & parent2 == p1))
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
        
        trans_df = get_transition_df(species, p1, p2) %>% select(-species)
        trans_df = trans_df %>%
          left_join(lookup %>% rename(parent1 = genotype,
                                      parent1_pheno = phenotype,
                                      parent1_label = label,
                                      parent1_img_file = img_file,
                                      parent1_img_html = img_html), by = "parent1") %>%
          left_join(lookup %>% rename(parent2 = genotype,
                                      parent2_pheno = phenotype,
                                      parent2_label = label,
                                      parent2_img_file = img_file,
                                      parent2_img_html = img_html), by = "parent2") %>%
          left_join(lookup %>% rename(offspring = genotype,
                                      offspring_pheno = phenotype,
                                      offspring_label = label,
                                      offspring_img_file = img_file,
                                      offspring_img_html = img_html), by = "offspring")
        
        trans_df %>% nest(transitions = c(offspring, offspring_pheno, prob, offspring_label, offspring_img_file, offspring_img_html))
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
  
  add_keep_flags = function(plan)
  {
    waves = plan$waves
    n = length(waves)
    if (n <= 1) return(plan)
    
    targets = get_targets()
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
  
  output$downloadBttn = downloadHandler(
    filename = function() {paste0("acnh_policy_", input$planSpecies, "_", input$seed, ".rds")},
    content = function(file) 
    {
      res = plan_result()
      req(res)
      
      policy = build_policy_plan(input$planSpecies, res)
      policy = add_keep_flags(policy)
      
      saveRDS(policy, file)
    })
  
  ####################################### View Policy ########################################################3
  
  policy_rv = reactiveVal(NULL)
  
  observeEvent(input$policyFile, {
    req(input$policyFile$datapath)
    pol = readRDS(input$policyFile$datapath)
    policy_rv(pol)
  })
  
  output$loadStatus = renderUI(
  {
    pol = policy_rv()
    if(is.null(pol)) {return(tags$span(style="opacity:0.8;", "No policy loaded yet."))}
    tags$span(style="color:#2e7d32; font-weight:600;", "Policy loaded.")
  })
  
  output$policyView = renderUI(
  {
    pol = policy_rv()
    req(pol)
    req(pol$waves)
    
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
    
    tags$div(class = "policy-grid",
             lapply(pol$waves, render_wave))
  })
}