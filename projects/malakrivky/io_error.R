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


analyze_io_error <- function() {

  curve_data <- load_data('bigtable.csv')
  sections <- curve_data[,2]
  unique_sections <- unique(sections)

  for (section_name in unique_sections) {
    print(section_name)
    section <- curve_data[curve_data[,2]==section_name,]

    groups <- section[,1]
    section_data <- section[, 3:dim(curve_data)[2]]

    io_error_error <- mean_sl_standard_deviation(section_data, 2)
    io_error_mean_group_error <- curves_group_error('/home/vajicek/src/shapeexplorer', section_data, groups, 2)
    io_error_analysis_report(io_error_mean_group_error, io_error_error)
  }
}

analyze_io_error()
