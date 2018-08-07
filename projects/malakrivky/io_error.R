# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/common.R")
source("base/processcurves.R")


load_data <- function(filename) {
  curve_data <- read.csv(filename, header=FALSE, sep = ',', dec='.')
  return(curve_data)
}


analyze_io_error <- function(output_dir) {
  curve_data <- load_data('bigtable.csv')
  sections <- curve_data[, 2]
  unique_sections <- unique(sections)

  for (section_name in unique_sections) {
    print(section_name)
    section <- curve_data[curve_data[, 2]==section_name,]

    groups <- section[, 1]
    section_data <- section[, 3:dim(curve_data)[2]]

    section_output_dir <- file.path(output_dir, section_name)
    dir.create(section_output_dir, recursive = TRUE)
    io_error_analysis(groups, section_data, section_output_dir, 2)
  }
}

analyze_io_error('/home/vajicek/src/shapeexplorer/result')
