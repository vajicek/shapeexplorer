# TODO: Add comment
# 
# Author: vajicek
###############################################################################

source("common.R")

use_library('ape')
use_library('rgl')
use_library('geomorph')
use_library('plyr')
use_library('ellipse')
use_library('corpcor')
use_library('Hotelling')

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
		specimen_matrix <- array(specimen_row, dim=c(k, p))
		data[,,i] <- t(specimen_matrix)
	}
	return(data)
}

transform_pkn_to_bigtable <- function(data) {
	p <- dim(data)[1]
	k <- dim(data)[2]
	n <- dim(data)[3]
	bigtable <- array(dim=c(n, k*p))
	for (i in 1:n) {
		specimen_row <- array(t(data[,,i]), dim=c(1, k * p))
		bigtable[i,] <- specimen_row
	}
	return(bigtable)
}

load_curves <- function(sample) {
	filepath <- file.path("output", paste(sample, ".csv", sep="", dec=''))
	curve_data <- read.csv(filepath, header=FALSE, sep = ';', dec='.')
	curve_data_pkn <- transform_bigtable_to_pkn(curve_data)
	return(curve_data_pkn)
}

load_groups <- function(sample_group_file) {
	filepath <- file.path("output", paste(sample_group_file, "_group.csv", sep="", dec=''))
	groups <- read.csv(filepath, header=FALSE, sep = ';', dec='.')
	return(groups)
}

store_gpa <- function(table, sample) {
	filepath <- file.path("output", paste(sample, "_gpa", ".csv", sep="", dec=''))
	write.table(table, filepath, row.names = FALSE, col.names = FALSE, sep=';')
}

# x,y scores
plot_pca_ellipse <- function(x, y, level=0.95, col='r', lty=1) {
	n <- length(x)
	ret <- rep(FALSE, times=n)
	if (n > 2) {			
		cv <- cov(cbind(x,y))
		cr <- cor(cbind(x,y))
		lines(ellipse(cv, centre=c(mean(x), mean(y)), level=level), col=col, lty=lty)
	}
}

plot_pca <- function(filepath, pca, groups, xcomp, ycomp) {
	pdf(filepath)
	colors <- c('red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black')
	unique_groups <- unique(groups)$V1
	groups_count <- length(unique_groups)
	group_cols <- colors[1:groups_count]
	spec_cols <- mapvalues(groups$V1, from=unique_groups, to=group_cols)
	plot(x=pca$x[,xcomp],
			y=pca$x[,ycomp],
			col=spec_cols,
			xlab=paste('PCA', toString(xcomp)),
			ylab=paste('PCA', toString(ycomp)))
	legend("bottomleft", legend=unique_groups, col=group_cols, pch=1)
	
	for (group in 1:groups_count) {
		group_mask <- groups$V1==unique_groups[group]
		x <- pca$x[group_mask, xcomp]
		y <- pca$x[group_mask, ycomp]
		plot_pca_ellipse(x=x, y=y, col=group_cols[group])
	}
	
	dev.off()
}

broken_stick_criterium <- function(variability) {
	n <- length(variability)
	broken <- rep(0, times=n)
	for (i in 1:n) {
		broken[i] <- 1/n * sum(1/i:n)
	}
	return(which.min(variability>broken) - 1)
}

pca <- function(sample, sample_gpa, groups, xcomp, ycomp) {
	filepath <- file.path("output", paste(sample, "_pca.pdf", sep="", dec=''))
	print(dim(sample_gpa))
	pca <- prcomp(sample_gpa, scale = FALSE, retx = TRUE)
	plot_pca(filepath, pca, groups, xcomp, ycomp)
	variability <- as.matrix(pca$sdev)^2
	variability <- variability / sum(variability)
	return(list(score=pca$x, variability=variability))
}

eval_manova <- function(dependent_variable, independent_variable) {
	lmodel <- lm(dependent_variable~., data=independent_variable)
	fit <- manova(lmodel)
	filename <- file.path("output", "manova.txt")
	sink(file=filename)
	for (test in c("Pillai", "Wilks", "Roy", "Hotelling-Lawley")) {
		s1 <- summary(fit, test=test)
		print(s1$stats)
	}
	sink(file=NULL)
}

eval_hotelling <- function(data, groups, nperm) {
	unique_groups <- as.character(unique(groups$V1))
	groups_count <- length(unique_groups)
	pairs <- expand.grid(1:groups_count, 1:groups_count)
	pvals <- matrix(ncol = groups_count, nrow = groups_count, dimnames = list(unique_groups, unique_groups))
	for (i in 1:dim(pairs)[1]) {
		if (pairs[i,]$Var1 <= pairs[i,]$Var2) {
			next
		}
		data1 <- data[groups$V1==unique_groups[pairs[i,]$Var1],]
		data2 <- data[groups$V1==unique_groups[pairs[i,]$Var2],]
		testResult <- hotelling.test(data1, data2, perm=(nperm>0), B=nperm, progBar=FALSE)
		pvals[pairs[i,]$Var1, pairs[i,]$Var2] <- testResult$pval 
	}
	write.csv(pvals, file.path("output", "pvals.csv"))
}

statistics <- function(sample, sample_gpa, sample_groups) {
	# remove dependencies
	pca_results <- pca(sample, sample_gpa, sample_groups, 1, 2)
	sig_components_count <- broken_stick_criterium(pca_results$variability)
	sample_data <- pca_results$score[,1:sig_components_count]
	
	# manova 
	eval_manova(sample_data, sample_groups)
	
	# paired hotelling
	eval_hotelling(sample_data, sample_groups, 10000)
}

mean_curves <- function(sample_gpa, sample_groups) {
	unique_groups <- as.character(unique(sample_groups$V1))
	groups_count <- length(unique_groups)
	means <- matrix(ncol = dim(sample_gpa)[2], nrow = groups_count)
	for (i in 1:groups_count) {
		group_data <- sample_gpa[sample_groups==unique_groups[i],]
		group_mean <- colMeans(group_data)
		means[i,] = group_mean
	}
	write.table(means, file.path("output", "mean.csv"), row.names=FALSE, col.names=FALSE, sep=";")
	write.table(unique_groups, file.path("output", "mean_group.csv"), row.names=FALSE, col.names=FALSE, sep=";")
}

process_curves <- function(sample) {
	sample_data <- load_curves(sample)
	sample_groups <- load_groups(sample)
	
	#
	sample_gpa <- transform_pkn_to_bigtable(gpagen(sample_data)$coords)
	store_gpa(sample_gpa, sample)
	
	#
	mean_curves(sample_gpa, sample_groups)
	
	#
	#statistics(sample, sample_gpa, sample_groups)
}

process_curves("all")

