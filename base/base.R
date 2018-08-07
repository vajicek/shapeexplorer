# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/common.R")

use_library('ellipse')
use_library('plyr')

# p - landmarks
# k - dimenstions
# n - specimens
transform_bigtable_to_pkn <- function(curve_data, coord_dim=3) {
	n <- dim(curve_data)[1]
	k <- coord_dim
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

get_pca_plot_params <- function(filename_prefix) {
	pca95 <- list(xcomp=1, ycomp=2, level=0.95, filename=paste0(filename_prefix, "_95", "_pca.pdf"))
	pca85 <- list(xcomp=1, ycomp=2, level=0.85, filename=paste0(filename_prefix, "_85", "_pca.pdf"))
	pca70 <- list(xcomp=1, ycomp=2, level=0.70, filename=paste0(filename_prefix, "_70", "_pca.pdf"))
	pca55 <- list(xcomp=1, ycomp=2, level=0.55, filename=paste0(filename_prefix, "_55", "_pca.pdf"))
	return(list(pca95, pca85, pca70, pca55))
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

plot_pca <- function(output_dir, pca, groups, params) {
	for (param1 in params) {
		pdf(file.path(output_dir, param1$filename), width=10, height=8)
		colors <- c('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black')
		unique_groups <- unique(groups)
		groups_count <- length(unique_groups)
		group_cols <- colors[1:groups_count]
		spec_cols <- mapvalues(groups, from=unique_groups, to=group_cols)
		palette(group_cols)
		par(xpd = T, mar = par()$mar + c(0,0,0,9))
		plot(x=pca$x[,param1$xcomp],
				y=pca$x[,param1$ycomp],
				col=spec_cols,
				xlab=paste0('PCA ', toString(param1$xcomp)),
				ylab=paste0('PCA ', toString(param1$ycomp)))
		legend("topright", inset=c(-0.30, 0), legend=unique_groups, col=group_cols, pch=1)

		for (group in 1:groups_count) {
			group_mask <- groups==unique_groups[group]
			x <- pca$x[group_mask, param1$xcomp]
			y <- pca$x[group_mask, param1$ycomp]
			plot_pca_ellipse(x=x, y=y, level=param1$level, col=group_cols[group])
		}

		dev.off()
	}
}

analyse_lines_param <- function(lines)  {
	flatten_data <- c()
	names <- c()
	ltys <- c()
	for (line1 in lines) {
		flatten_data <- cbind(flatten_data, line1$data)
		names <- append(names, line1$name)
		ltys <- append(ltys, line1$lty)
	}
	return(list(
		ylim=c(min(flatten_data), max(flatten_data)),
		names=names,
		ltys=ltys))
}

plot_line <- function(output_dir, lines, params) {
	pdf(file.path(output_dir, params$filename), width=10, height=8)
	line_params <- analyse_lines_param(lines)

	plot(x=c(), y=c(), type="n",
		xlim=params$xlim,
		ylim=line_params$ylim,
		xlab=params$xlab,
		ylab=params$ylab)
	for (line1 in lines) {
		lines(x=1:length(unlist(line1$data)), y=line1$data, type="l", lty=line1$lty)
	}

	legend(params$legend_position, legend=line_params$names, lty=line_params$ltys)
	dev.off()
}

get_broken_stick_criterium_sequence <- function(n) {
	broken <- rep(0, times=n)
	for (i in 1:n) {
		broken[i] <- 1/n * sum(1/i:n)
	}
	return (broken)
}

broken_stick_criterium <- function(variability) {
	cat("Broken stick_criterium: \n")
	n <- length(variability)
	broken <- get_broken_stick_criterium_sequence(n)
	index <- which.min(variability>broken) - 1
	cat(paste0("Number of significant components: ", index, "\n"))
	cat(paste0("Variation represented by significant components: ", sum(variability[1:index]), "\n"))
	return(index)
}

compute_pca <- function(output_dir, prefix, sample_gpa, groups, pca_plot_params) {
	pca <- prcomp(sample_gpa, scale=FALSE, retx=TRUE)
	plot_pca(output_dir, pca, groups, pca_plot_params)

	# store loadings
	write.table(t(pca$rotation), file.path(output_dir, paste0(prefix, "_pca_loadings.csv")), row.names=FALSE, col.names=FALSE, sep=";")

	# store pca scores
	write.table(pca$x, file.path(output_dir, paste0(prefix, "_pca_scores.csv")), row.names=FALSE, col.names=FALSE, sep=";")

	variability <- as.matrix(pca$sdev)^2
	variability <- variability / sum(variability)

	# plot screeplot
	n <- length(variability)
	plot_line(output_dir,
		list(list(data=100 * variability, lty=1, name="Variability (%)"),
			list(data=100 * get_broken_stick_criterium_sequence(n), lty=2, name='Broken stick')),
		list(filename="screeplot.pdf",
			legend_position="topright",
			xlab="Component",
			ylab="Variability (%)",
			xlim=c(1, n)))

	# store variability
	write.table(variability, file.path(output_dir, paste0(prefix, "_pca_variability.csv")), row.names=FALSE, col.names=FALSE, sep=";")

	return(list(score=pca$x, variability=variability, loadings=pca$rotation))
}

eval_manova <- function(output_dir, prefix, dependent_variable, independent_variable) {
	fit <- manova(dependent_variable~independent_variable)
	filename <- file.path(output_dir, paste0(prefix, "manova.txt"))
	sink(file=filename)
	for (test in c("Pillai", "Wilks", "Roy", "Hotelling-Lawley")) {
		print(summary(fit, test=test))
	}
	sink(file=NULL)
}
