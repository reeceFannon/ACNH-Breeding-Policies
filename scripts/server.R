library(shiny)
library(tidyverse)
library(reticulate)
source("scripts/utils.R")
source_python("scripts/python_wrapper.py")
addResourcePath("imgs", "imgs")

server = function(input, output, session)
{
  #################################################################################################################
  ####################################### Breeding Tab ############################################################
  #################################################################################################################
  
  ####################################### Input Widgets ###########################################################
  
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
  
  #################################### Edit Policy Object #########################################################
  
  eventLog = reactiveVal(NULL)
  numActions = reactiveVal(0)
  numWaves = reactiveVal(0)
  policyObject = reactiveValues(waves = list())
  Wave = reactiveVal(list(wave = 1, actions = list()))
  
  Action = reactive(
  {
    req(input$species, input$parent1, input$parent2)
    lookup = build_filtered_df(input$species) %>% select(-flower)
    action = if(correct_order(input$species, input$parent1, input$parent2, lookup)) construct_action(input$species, input$parent1, input$parent2, lookup) else construct_action(input$species, input$parent2, input$parent1, lookup) 
    action
  })
  
  output$checkbox = renderUI(
  {
    action = Action()
    req(action)
    
    trans_df = action$transitions[[1]]
    values = trans_df$offspring
    names  = trans_df$offspring_img_html
    
    checkboxGroupButtons(inputId = "keepDiscard",
                         label = "Select Offspring to Keep",
                         direction = "vertical",
                         individual = TRUE,
                         justified = TRUE,
                         size = "xs",
                         choiceValues = values,
                         choiceNames = names,
                         checkIcon = list(yes = icon("check", lib = "glyphicon"),
                                          no = icon("trash", lib = "glyphicon")))
  })
  
  observeEvent(input$addAction, {
    action = Action()
    wave = Wave()
    req(action, wave)
    
    picked = input$keepDiscard %||% character(0)
    action = action %>% mutate(transitions = map(transitions, function(tr) {tr %>% mutate(keep = offspring %in% picked)}))
    
    a = numActions() + 1
    wave$actions[[a]] = action
    numActions(a)
    Wave(wave)
    
    eventLog(paste0("Action ", "(", action$parent1[1], " x ", action$parent2[1], ") ",  "added to Wave ", numWaves() + 1))
  })
  
  observeEvent(input$addWave, {
    wave = Wave()
    req(wave)
    
    w = numWaves() + 1
    wave$wave = w
    
    policyObject$waves[[w]] = wave
    
    numWaves(w)
    numActions(0)
    Wave(list(wave = w + 1, actions = list()))
    
    eventLog(paste0("Wave ", w, " added with ", length(wave$actions), " actions"))
  })
  
  manual_policy = reactive(
  {
    waves = policyObject$waves
    cur = Wave()
    if (!is.null(cur) && length(cur$actions) > 0)
    {
      cur$wave = numWaves() + 1
      waves = c(waves, list(cur))
    }
    list(waves = waves)
  })
  
  output$downloadBttn2 = renderUI(
    {
      pol = manual_policy()
      req(pol)
      
      if(is.null(pol)) return(NULL)
      box(width = 12,
          status = "info",
          solidHeader = TRUE,
          actionBttn(inputId = "download2",
                     label = "Download Policy",
                     style = "material-flat",
                     color = "royal",
                     size = "md",
                     block = TRUE,
                     icon = icon("save", lib = "glyphicon")))
    })
  
  output$downloadBttn2 = downloadHandler(
    filename = function() {paste0("acnh_policy_", input$planSpecies, "_", ".rds")},
    content = function(file) 
    {
      pol = manual_policy()
      req(pol)
      
      saveRDS(pol, file)
    })
  
  output$eventLogText = renderText(
  {
    eventLog()
  })
  
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
  
  plan_warning = reactiveVal(NULL)
  plan_result  = reactiveVal(NULL)
  plan_running = reactiveVal(FALSE)
  
  get_targets = function(rv)
  {
    if (rv$n_groups <= 0) return(list())
    targets = lapply(seq_len(rv$n_groups), function(k) {input[[paste0("targetGroup_", k)]]})
    Filter(function(x) !is.null(x) && length(x) > 0, targets)
  }
  
  observeEvent(input$runPlanner, {
    plan_warning(NULL)
    plan_result(NULL)
    
    # ---- validate ----
    targets = get_targets(rv)
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
  
  output$downloadBttn = downloadHandler(
    filename = function() {paste0("acnh_policy_", input$planSpecies, "_", input$seed, ".rds")},
    content = function(file) 
    {
      res = plan_result()
      req(res)
      
      targets = get_targets(rv)
      policy = build_policy_plan(input$planSpecies, res)
      policy = add_keep_flags(policy, targets)
      
      saveRDS(policy, file)
    })

  #############################################################################################################
  ####################################### Viewing Tab ########################################################
  #############################################################################################################
  
  ####################################### View Policy #########################################################
  
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
    
    tags$div(class = "policy-grid",
             lapply(pol$waves, render_wave))
  })
}