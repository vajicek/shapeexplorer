# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/common.R")
source("base/base.R")
source("base/io_error.R")

use_library('ape')
use_library('rgl')
use_library('geomorph')
use_library('corpcor')
use_library('Hotelling')
use_library("optparse")
use_library("stringr")


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

eval_hotelling <- function(output_dir, data, groups, nperm) {
	unique_groups <- as.character(unique(groups))
	groups_count <- length(unique_groups)
	pairs <- expand.grid(1:groups_count, 1:groups_count)
	pvals <- matrix(ncol = groups_count, nrow = groups_count, dimnames = list(unique_groups, unique_groups))
	for (i in 1:dim(pairs)[1]) {
		if (pairs[i,]$Var1 <= pairs[i,]$Var2) {
			next
		}
		data1 <- data[groups==unique_groups[pairs[i,]$Var1],]
		data2 <- data[groups==unique_groups[pairs[i,]$Var2],]
		testResult <- hotelling.test(data1, data2, perm=(nperm>0), B=nperm, progBar=FALSE)
		pvals[pairs[i,]$Var1, pairs[i,]$Var2] <- testResult$pval
	}
	write.csv(pvals, file.path(output_dir, "hotelling_pvals.csv"))
}

pca_reduction <- function(output_dir, sample_name, sample_gpa, sample_groups) {
	pca_results <- compute_pca(output_dir, sample_name, sample_gpa, sample_groups, get_pca_plot_params(sample_name))
	sig_components_count <- broken_stick_criterium(pca_results$variability)
	if (sig_components_count <= 1) {
		sig_components_count <- 2
		print("WARNING: Number of singnificant components according to broken stick criterion <=1, 2 is used for multivariate tests.")
	}
	sample_data <- pca_results$score[,1:sig_components_count]
	return(sample_data)
}

# output_dir - directory
# sample_name - string, filename prefix
# sample_gpa - landmarks
# sample_groups - groups
statistics <- function(output_dir, sample_name, sample_gpa, sample_groups) {
	cat("Compute PCA\n")
	sample_data <- pca_reduction(output_dir, sample_name, sample_gpa, sample_groups)

	cat("Eval MANOVA\n")
	eval_manova(output_dir, "", sample_data, sample_groups)

	cat("Eval paired Hotelling T2 test\n")
	eval_hotelling(output_dir, sample_data, sample_groups, 10000)
}

mean_curves <- function(sample_gpa, sample_groups) {
	unique_groups <- as.character(unique(sample_groups))
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

eval_allometry <- function(output_dir, sample_gpa, sample_lengths, sample_groups, sample_name) {
	sample_gpa_pca <- pca_reduction(output_dir, sample_name, sample_gpa, sample_groups)
	model <- lm(sample_gpa_pca ~ sample_lengths)
	print(anova(model))
}

point_distance <- function(a, b) {
	return(sqrt(sum((a - b)**2)))
}

compute_lengths <- function(sample_data) {
	dims <- dim(sample_data)
	curve_lengths <- c()
	for (specimen_no in 1:dims[3]) {
		length <- 0
		for (lm_no in 1:dims[1]) {
			if (lm_no > 1) {
				length <- length + point_distance(sample_data[lm_no - 1,,specimen_no],
					sample_data[lm_no,,specimen_no])
			}
		}
		curve_lengths <- c(curve_lengths, length)
	}
	return(curve_lengths)
}

curve_gpa <- function(sample_data, slm_handling) {
	if (slm_handling == "none") {
		gpa <- gpagen(sample_data, print.progress=FALSE)
	} else if (slm_handling == "procd") {
		curve_sliders <- get_curve_sliders(sample_data)
		gpa <- gpagen(sample_data, print.progress=FALSE, curves=curve_sliders, ProcD=TRUE)
	} else if (slm_handling == "bende") {
		curve_sliders <- get_curve_sliders(sample_data)
		gpa <- gpagen(sample_data, print.progress=FALSE, curves=curve_sliders, ProcD=FALSE)
	}
	return(gpa)
}

curves_allometry_analysis <- function(input_dir, output_dir, slm_handling) {
	sample_groups <- load_groups("all", input_dir)$V1
	unique_groups <- as.character(unique(sample_groups))

	for (sample_name in unique_groups) {
		cat("\n--------------------------------------------------\n")
		cat(paste0("Allometry analysis for ", sample_name, "\n"))
		sample_data <- load_curves(sample_name, input_dir)
		gpa <- curve_gpa(sample_data, slm_handling)
		sample_gpa <- transform_pkn_to_bigtable(gpa$coords)
		sample_lengths <- compute_lengths(sample_data)
		eval_allometry(output_dir, sample_gpa, sample_lengths, c(sample_name), sample_name)
	}
}

curves_variability_analysis <- function(input_dir, output_dir, slm_handling, sample_name) {
	sample_data <- load_curves(sample_name, input_dir)
	sample_groups <- load_groups(sample_name, input_dir)$V1

	cat("Peform GPA\n")
	gpa <- curve_gpa(sample_data, slm_handling)

	cat("Store GPA\n")
	sample_gpa <- transform_pkn_to_bigtable(gpa$coords)
	store_gpa(sample_gpa, sample_name, output_dir)

	cat("Store curves\n")
	means <- mean_curves(sample_gpa, sample_groups)
	store_named_curves(means$means, means$names, 'means', output_dir)

	cat("Compute statistics\n")
	statistics(output_dir, sample_name, sample_gpa, sample_groups)
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
	pairwise.t.test(data$length, data$group, p.adj="none")
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

main <- function() {
	# command-line interface
	option_list = list(
			make_option(c("--output"), default=""),
			make_option(c("--input"), default=""),
			make_option(c("--io_error"), action="store_true", default=FALSE),
			make_option(c("--variability"), action="store_true", default=FALSE),
			make_option(c("--allometry"), action="store_true", default=FALSE),
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
	} else if (opt$allometry) {
		cat("ALLOMETRY ANALYSIS\n")
		curves_allometry_analysis(opt$input, opt$output, opt$slm_handling)
	} else if (opt$length_analysis) {
		cat("CURVE LENGTH ANALYSIS\n")
		curves_length_analysis(opt$output)
	}
}

if (sys.nframe() == 0L) {
	main()
}
