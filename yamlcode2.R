# needed only for package installation or update
#install.packages("devtools")
library(devtools)
devtools::install_github("lborke/yamldebugger")

# load the package every time you want to use 'yamldebugger'
#install.packages("yamldebugger")
library(yamldebugger)

allKeywords
"plot" %in% allKeywords

  help(yaml.debugger.init)
d_init = yaml.debugger.init("/Users/rdc/GitHub/electricitypriceforecasting", show_keywords = TRUE)

  help(yaml.debugger.get.qnames)
qnames = yaml.debugger.get.qnames(d_init$RootPath)

workdir = "/Users/rdc/GitHub/MMSTAT"

d_init = yaml.debugger.init(workdir, show_keywords = TRUE)

qnames = yaml.debugger.get.qnames(d_init$RootPath)

d_results = yaml.debugger.run(qnames, d_init)

OverView = yaml.debugger.summary(qnames, d_results, summaryType = "mini")

#install.packages("formatR")
library(formatR)
tidy_dir("/Users/rdc/GitHub/electricitypriceforecasting", recursive = T)
