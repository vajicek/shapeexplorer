# TODO: Add comment
# 
# Author: vajicek
###############################################################################

LIB_LOCATIONS=file.path(path.expand("~"), 'R/library')

use_library <- function(libname) {
	if (libname %in% rownames(installed.packages(lib.loc=LIB_LOCATIONS)) == FALSE) {
		dir.create(LIB_LOCATIONS, recursive=TRUE, showWarnings=FALSE)
		install.packages(libname, lib=LIB_LOCATIONS, dependencies=TRUE, repos="https://cloud.r-project.org")
	}
	library(libname, lib.loc=LIB_LOCATIONS, character.only=TRUE)
}