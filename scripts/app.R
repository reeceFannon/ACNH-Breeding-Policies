source("scripts/utils.R")
source("scripts/ui.R")
source("scripts/server.R")

shinyApp(ui, server)

#blue_policy = fromJSON("blue_rose_policy.json", simplifyVector = FALSE)
#blue_policy = build_policy_plan("rose", blue_policy)
#blue_policy = add_keep_flags(blue_policy, "RRYYwwss")
#saveRDS(blue_policy, "blue_rose_policy.rds")
