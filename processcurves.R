# TODO: Add comment
# 
# Author: vajicek
###############################################################################

LIB_LOCATIONS=file.path(path.expand("~"), 'R/library')

use_library <- function(libname) {
	if (libname %in% rownames(installed.packages(lib.loc=LIB_LOCATIONS)) == FALSE) {
		dir.create(LIB_LOCATIONS, recursive=TRUE, showWarnings=FALSE)
		install.packages(libname, lib=LIB_LOCATIONS, repos="https://cloud.r-project.org")
	}
	library(libname, lib.loc=LIB_LOCATIONS, character.only=TRUE)
}

use_library('ape')
use_library('rgl')
use_library('geomorph')

# p - landmarks
# k - dimenstions
# n - specimens
transform_bigtable_to_pkn <- function(curve_data) {
	n <- dim(curve_data)[1]
	k <- 3
	p <- dim(curve_data)[2] / k
	data <- array(dim=c(p, k, n))
	for (i in 1:n) {
		specimen_row <- as.double(curve_data[i,])
		specimen_matrix <- array(specimen_row, dim=c(1, k, p))
		data[,,i] <- specimen_matrix
	}
	return(data)
}

transform_pkn_to_bigtable <- function(data) {
	p <- dim(data)[1]
	k <- dim(data)[2]
	n <- dim(data)[3]
	bigtable <- array(dim=c(n, k*p))
	for (i in 1:1) {
		specimen_row <- array(t(data[,,i]), dim=c(1, k * p))
		bigtable[i,] <- specimen_row
	}
}

load_curves <- function(category) {
	filepath <- file.path("output", paste(category, ".csv", sep="", dec=''))
	curve_data <- read.csv(filepath, header=FALSE, sep = ';', dec='.')
	curve_data_pkn <- transform_bigtable_to_pkn(curve_data)
	gpa_result <- gpagen(curve_data_pkn)
	names(gpa_result)
	transform_pkn_to_bigtable(gpa_result$coords)
}


load_curves("A_eneolit")

