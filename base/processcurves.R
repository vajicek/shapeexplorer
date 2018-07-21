# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/common.R")

use_library('ape')
use_library('rgl')
use_library('geomorph')
use_library('plyr')
use_library('ellipse')
use_library('corpcor')
use_library('Hotelling')
use_library("optparse")
use_library("stringr")

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

load_curves <- function(sample, input_dir) {
	filepath <- file.path(input_dir, paste0(sample, ".csv"))
	curve_data <- read.csv(filepath, header=FALSE, sep = ';', dec='.')
	curve_data_pkn <- transform_bigtable_to_pkn(curve_data)
	return(curve_data_pkn)
}

load_groups <- function(sample_group_file, input_dir) {
	filepath <- file.path(input_dir, paste0(sample_group_file, "_group.csv"))
	groups <- read.csv(filepath, header=FALSE, sep = ';', dec='.')
	return(groups)
}

store_gpa <- function(table, sample, output_dir) {
	dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)
	filepath <- file.path(output_dir, paste0(sample, "_gpa", ".csv"))
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

plot_pca <- function(output_dir, pca, groups, params) {
	for (param1 in params) {
		pdf(file.path(output_dir, param1$filename), width=10, height=8)
		colors <- c('red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'black')
		unique_groups <- unique(groups)$V1
		groups_count <- length(unique_groups)
		group_cols <- colors[1:groups_count]
		spec_cols <- mapvalues(groups$V1, from=unique_groups, to=group_cols)
		palette(group_cols)
		par(xpd = T, mar = par()$mar + c(0,0,0,9))
		plot(x=pca$x[,param1$xcomp],
				y=pca$x[,param1$ycomp],
				col=spec_cols,
				xlab=paste0('PCA ', toString(param1$xcomp)),
				ylab=paste0('PCA ', toString(param1$ycomp)))
		legend("topright", inset=c(-0.30, 0), legend=unique_groups, col=group_cols, pch=1)

		for (group in 1:groups_count) {
			group_mask <- groups$V1==unique_groups[group]
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
	lmodel <- lm(dependent_variable~., data=independent_variable)
	fit <- manova(lmodel)
	filename <- file.path(output_dir, paste0(prefix, "manova.txt"))
	sink(file=filename)
	for (test in c("Pillai", "Wilks", "Roy", "Hotelling-Lawley")) {
		s1 <- summary(fit, test=test)
		print(s1$stats)
	}
	sink(file=NULL)
}

eval_hotelling <- function(output_dir, data, groups, nperm) {
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
	write.csv(pvals, file.path(output_dir, "hotelling_pvals.csv"))
}

get_pca_plot_params <- function(filename_prefix) {
	pca95 <- list(xcomp=1, ycomp=2, level=0.95, filename=paste0(filename_prefix, "_95", "_pca.pdf"))
	pca85 <- list(xcomp=1, ycomp=2, level=0.85, filename=paste0(filename_prefix, "_85", "_pca.pdf"))
	pca70 <- list(xcomp=1, ycomp=2, level=0.70, filename=paste0(filename_prefix, "_70", "_pca.pdf"))
	pca55 <- list(xcomp=1, ycomp=2, level=0.55, filename=paste0(filename_prefix, "_55", "_pca.pdf"))
	return(list(pca95, pca85, pca70, pca55))
}

statistics <- function(output_dir, sample, sample_gpa, sample_groups) {
	# pca (remove dependencies)
	pca_results <- compute_pca(output_dir, sample, sample_gpa, sample_groups, get_pca_plot_params(sample))
	sig_components_count <- broken_stick_criterium(pca_results$variability)
	sample_data <- pca_results$score[,1:sig_components_count]

	# manova
	eval_manova(output_dir, "", sample_data, sample_groups)

	# paired hotelling
	eval_hotelling(output_dir, sample_data, sample_groups, 10000)
}

mean_curves <- function(sample_gpa, sample_groups) {
	unique_groups <- as.character(unique(sample_groups$V1))
	groups_count <- length(unique_groups)
	means <- matrix(ncol = dim(sample_gpa)[2], nrow = groups_count + 1)

	# group means
	for (i in 1:groups_count) {
		group_data <- sample_gpa[sample_groups==unique_groups[i],]
		group_mean <- colMeans(group_data)
		means[i,] = group_mean
	}

	# all mean
	means[groups_count + 1,] = colMeans(sample_gpa)
	unique_groups = c(unique_groups, "all")

	return(list(means=means, names=unique_groups))
}

store_named_curves <- function(curves, names, prefix, output_dir) {
	write.table(curves, file.path(output_dir, paste0(prefix, ".csv", sep="")), row.names=FALSE, col.names=FALSE, sep=";")
	write.table(names, file.path(output_dir, paste0(prefix, "_group.csv", sep="")), row.names=FALSE, col.names=FALSE, sep=";")
}

get_curve_sliders <- function(sample_data) {
	lmc <- dim(sample_data)[1]
	curve_slider <- data.frame(before=1:(lmc-2), slide=2:(lmc-1), after=3:(lmc))
	return(curve_slider)
}

curves_variability_analysis <- function(input_dir, output_dir, slm_handling, sample) {
	sample_data <- load_curves(sample, input_dir)
	sample_groups <- load_groups(sample, input_dir)

	#
	if (slm_handling == "none") {
		gpa <- gpagen(sample_data, print.progress=FALSE)
	} else if (slm_handling == "procd") {
		curve_sliders <- get_curve_sliders(sample_data)
		gpa <- gpagen(sample_data, print.progress=FALSE, curves=curve_sliders, ProcD=TRUE)
	} else if (slm_handling == "bende") {
		curve_sliders <- get_curve_sliders(sample_data)
		gpa <- gpagen(sample_data, print.progress=FALSE, curves=curve_sliders, ProcD=FALSE)
	}
	cat("Peformed GPA\n")
	print(gpa)

	sample_gpa <- transform_pkn_to_bigtable(gpa$coords)
	store_gpa(sample_gpa, sample, output_dir)

	#
	means <- mean_curves(sample_gpa, sample_groups)
	store_named_curves(means$means, means$names, 'means', output_dir)

	#
	statistics(output_dir, sample, sample_gpa, sample_groups)
}

curves_length_analysis <- function(output_dir) {
	# load data
	filepath <- file.path(output_dir, "curves_lengths.csv.csv")
	lengths <- read.csv(filepath, header=FALSE, sep = ';', dec='.')
	data <- data.frame(length=lengths$V3, group=lengths$V2)

	# plot
	pdf(file.path(output_dir, "curves_length.pdf"), width=14, height=8)
	boxplot(length~group,
			data=data,
			main="Lengths per group",
			xlab="Group",
			ylab="Length")
	dev.off()

	# statistics
	cat("ANOVA\n")
	fit <- aov(length ~ group, data=data)
	print(summary(fit))

	cat("PAIRED t-Test\n")
	pairwise.t.test (data$length, data$group, p.adj="none")
}

# sum_curve (sum (curve - mean curve)^2) / (dim * slc)
curves_variance <- function(curves) {
	curves_dim <- dim(curves)
	curves_count <- curves_dim[1]
	mean_curve <- colMeans(curves)
	err_sum <- 0
	for (c in 1:curves_count) {
		shifted_curve <- curves[c,] - mean_curve
		err <- sum(shifted_curve^2)
		# coordinates variance
		err_sum <- err_sum + err / curves_dim[2]
	}
	# mean error
	return(err_sum / curves_count)
}

# according to von Cramon
sl_standard_deviation <- function(curves, coord_dim=3) {
	# align set
	aligned_curves <- transform_pkn_to_bigtable(gpagen(transform_bigtable_to_pkn(curves), print.progress=FALSE)$coords)

	# lms errors from mean lms
	curves_dim <- dim(aligned_curves)
	curves_count <- curves_dim[1]
	mean_curve <- colMeans(aligned_curves)
	err_sum <- matrix(0L, ncol = curves_dim[2], nrow = 1)
	for (c in 1:curves_count) {
		shifted_curve <- aligned_curves[c,] - mean_curve
		err_sum <- err_sum + shifted_curve^2
	}
	slm_std_dev = sqrt(colSums(array(err_sum, dim=c(coord_dim, curves_dim[2] / coord_dim))) / (coord_dim * curves_count))
	return(slm_std_dev)
}

mean_sl_standard_deviation <- function(curves, curves_dim=3) {
	return(mean(sl_standard_deviation(curves, curves_dim)))
}

error_plot <- function (output_dir, prefix, errors) {
	filepath <- file.path(output_dir, paste0(prefix, "_error.pdf"))
	pdf(filepath)
	print(errors)
	plot(errors, type = "o")
	dev.off()
}

curves_group_error <- function(output_dir, curves, groups, curves_dim=3) {
	unique_groups <- as.character(unique(groups))
	cat("Unique groups: \n")
	print(unique_groups)
	groups_count <- length(unique_groups)
	group_error <- rep(0, groups_count)
	sl_count = dim(curves)[2] / curves_dim
	sl_error <- matrix(0L, ncol = sl_count, nrow = groups_count)
	for (c in 1:groups_count) {
		group_curves <-	curves[groups == unique_groups[c],]
		group_error[c] <- mean_sl_standard_deviation(group_curves, curves_dim)
		sl_error[c,] <- sl_standard_deviation(group_curves, curves_dim)
		# dump
		cat(paste0("Group: size=", sum(groups == unique_groups[c]),
						" name=", str_pad(unique_groups[c], 20, "right"),
						" error=", group_error[c], "\n"))
	}
	error_plot(output_dir, paste0("sl", sl_count), colMeans(sl_error))
	mean_group_error <- mean(group_error)
	cat(paste0("Mean group error: ", mean_group_error, "\n"))
	return(mean_group_error)
}

io_error_analysis_report <- function(io_error_mean_group_error, io_error_error) {
	# dump
	cat("ratio - how large is variance of sample in comparison to variance inside the groups\n")
	cat(paste0(" io_error_mean_group_error = ", io_error_mean_group_error, "\n",
					" io_error_error = ", io_error_error, "\n",
					" io_error_mean_group_error / io_error_error = ", io_error_mean_group_error / io_error_error, "\n"))
}

io_error_analysis <- function(input_dir, output_dir, curves_dim=3) {
	io_error_sample_data <- load_curves("io_error", input_dir)
	cat("Input data dimension (lm x dim x specimens): \n")
	print(dim(io_error_sample_data))

	slm <- dim(io_error_sample_data)[1]
	io_error_sample_groups <- load_groups("io_error")$V1
	io_error_sample_gpa <- transform_pkn_to_bigtable(gpagen(io_error_sample_data, print.progress=FALSE)$coords)
	io_error_mean_group_error <- curves_group_error(output_dir, io_error_sample_gpa, io_error_sample_groups, curves_dim)
	io_error_error <- mean_sl_standard_deviation(io_error_sample_gpa, curves_dim)

	io_error_analysis_report(io_error_mean_group_error, io_error_error)

	# pca and manova on repeated measures
	prefix <- paste0('measurement_error', slm)
	pca_results <- compute_pca(output_dir, prefix, io_error_sample_gpa, io_error_sample_groups, get_pca_plot_params(prefix))
	sig_components_count <- broken_stick_criterium(pca_results$variability)
	sample_data <- pca_results$score[,1:sig_components_count]
	eval_manova(output_dir, prefix, sample_data, io_error_sample_groups)

	result <- list(io_error_mean_group_error=io_error_mean_group_error,
			io_error_error=io_error_error,
			ration=io_error_mean_group_error/io_error_error)
	return(result)
}

# command-line interface
option_list = list(
		make_option(c("--output"), default=""),
		make_option(c("--input"), default=""),
		make_option(c("--io_error"), action="store_true", default=FALSE),
		make_option(c("--variability"), action="store_true", default=FALSE),
		make_option(c("--slm_handling"), action="store", default="none", help="none, procd, bende"),
		make_option(c("--length_analysis"), action="store_true", default=FALSE)
);

opt = parse_args(OptionParser(option_list=option_list))
if (opt$io_error) {
	cat("EVALUATE IO ERROR\n")
	res <- io_error_analysis(opt$input, opt$output)
} else if (opt$variability) {
	cat("VARIABILITY\n")
	curves_variability_analysis(opt$input, opt$output, opt$slm_handling, "all")
} else if (opt$length_analysis) {
	cat("CURVE LENGTH ANALYSIS\n")
	curves_length_analysis(opt$output)
}
