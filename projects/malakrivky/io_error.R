# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/common.R")
source("base/io_error.R")

use_library("optparse")


load_data <- function(filename) {
  curve_data <- read.csv(filename, header=FALSE, sep = ',', dec='.')
  return(curve_data)
}


analyze_io_error <- function(input_file, output_dir, skip_manova) {
  print(output_dir)
  curve_data <- load_data(input_file)
  sections <- curve_data[,2]
  unique_sections <- unique(sections)

  for (section_name in unique_sections) {
    print(section_name)
    section <- curve_data[curve_data[,2]==section_name,]

    groups <- section[,1]
    section_data <- section[, 3:dim(curve_data)[2]]

    io_error_mean_group_error <- curves_group_error(output_dir, section_data, groups, 2, section_name, FALSE)
    io_error_error <- mean_sl_standard_deviation(section_data, 2, FALSE)
    io_error_analysis_report(io_error_mean_group_error, io_error_error)
    if (!skip_manova) {
      io_error_manova(section_name, output_dir, section_data, groups)
    }
  }
}

# command-line interface
option_list = list(
  make_option(c("--skip_manova"), action="store_true", type="logical", default=FALSE),
  make_option(c("--output"), default="", action="store"),
  make_option(c("--input"), default="", action="store")
);

opt = parse_args(OptionParser(option_list=option_list))
analyze_io_error(opt$input, opt$output, opt$skip_manova)
