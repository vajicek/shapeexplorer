# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/base.R")

use_library("stringr")

aligned_sl_standard_deviation <- function(aligned_curves, coord_dim=3) {
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

# according to von Cramon
sl_standard_deviation <- function(curves, coord_dim=3, align=TRUE) {
	# align set
	if (align) {
		aligned_curves <- transform_pkn_to_bigtable(gpagen(transform_bigtable_to_pkn(curves, coord_dim), print.progress=FALSE)$coords)
	} else {
		aligned_curves <- curves
	}
	return(aligned_sl_standard_deviation(aligned_curves, coord_dim=coord_dim))
}

mean_sl_standard_deviation <- function(curves, curves_dim=3, align=TRUE) {
	return(mean(sl_standard_deviation(curves, curves_dim, align)))
}

error_plot <- function (output_dir, prefix, errors) {
	filepath <- file.path(output_dir, paste0(prefix, "_error.pdf"))
	pdf(filepath)
	plot(errors, type = "o")
	dev.off()
}

odd_elements <- function (line) {
	return(line[1:dim(line)[2] %% 2 == 1])
}

even_elements <- function (line) {
	return(line[1:dim(line)[2] %% 2 == 0])
}

all_measurements_plot <- function (output_dir, prefix, curves, groups, curves_dim=2) {
	filepath <- file.path(output_dir, paste0(prefix, "_all.pdf"))
	pdf(filepath)

	unique_groups <- as.character(unique(groups))
	groups_count <- length(unique_groups)

	min_x <- Inf
	max_x <- -Inf
	min_y <- Inf
	max_y <- -Inf
	for (c in 1:groups_count) {
		group_curves <- curves[groups == unique_groups[c],]
		group_curves_count <- dim(group_curves)[1]
		for (l in 1:group_curves_count) {
			line <- group_curves[l,]
			x <- odd_elements(line)
			y <- even_elements(line)
			min_x <- min(min(x), min_x)
			max_x <- max(max(x), max_x)
			min_y <- min(min(y), min_y)
			max_y <- max(max(y), max_y)
		}
	}

	w <- max_x - min_x
	h <- max_y - min_y
	e <- 0.05
	plot(x=c(), y=c(), type="n",
		xlim=c(min_x - w * e, max_x + w * e),
		ylim=c(min_y - h * e, max_y + h * e))

	for (c in 1:groups_count) {
		group_curves <- curves[groups == unique_groups[c],]
		group_curves_count <- dim(group_curves)[1]
		for (l in 1:group_curves_count) {
			line <- group_curves[l,]
			x <- odd_elements(line)
			y <- even_elements(line)
			lines(x=x, y=y, type="l", lty=1, col=c)
		}
	}
}

curves_group_error <- function(output_dir, curves, groups, curves_dim=3, prefix="", align=TRUE) {
	unique_groups <- as.character(unique(groups))
	cat("Unique groups: \n")
	print(unique_groups)
	groups_count <- length(unique_groups)
	group_error <- rep(0, groups_count)
	sl_count = dim(curves)[2] / curves_dim
	sl_error <- matrix(0L, ncol = sl_count, nrow = groups_count)
	for (c in 1:groups_count) {
		group_curves <- curves[groups == unique_groups[c],]
		group_error[c] <- mean_sl_standard_deviation(group_curves, curves_dim, align)
		sl_error[c,] <- sl_standard_deviation(group_curves, curves_dim, align)
		# dump
		cat(paste0("Group: size=", sum(groups == unique_groups[c]),
						" name=", str_pad(unique_groups[c], 20, "right"),
						" error=", group_error[c], "\n"))
	}
	error_plot(output_dir, paste0(prefix, "sl", sl_count), colMeans(sl_error))
	all_measurements_plot(output_dir, paste0(prefix, "sl", sl_count), curves, groups)
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

io_error_manova <- function(prefix, output_dir, sample_data, sample_groups) {
	pca_results <- compute_pca(output_dir, prefix, sample_data, sample_groups, get_pca_plot_params(prefix))
	sig_components_count <- broken_stick_criterium(pca_results$variability)
	reduced_sample_data <- pca_results$score[,1:sig_components_count]
	eval_manova(output_dir, prefix, reduced_sample_data, sample_groups)
}

io_error_analysis <- function(input_dir, output_dir, curves_dim=3, prefix="") {
	io_error_sample_data <- load_curves("io_error", input_dir)
	cat("Input data dimension (lm x dim x specimens): \n")
	print(dim(io_error_sample_data))

	slm <- dim(io_error_sample_data)[1]
	io_error_sample_groups <- load_groups("io_error")$V1
	io_error_sample_gpa <- transform_pkn_to_bigtable(gpagen(io_error_sample_data, print.progress=FALSE)$coords)

	io_error_mean_group_error <- curves_group_error(output_dir, io_error_sample_gpa, io_error_sample_groups, curves_dim, prefix)
	io_error_error <- mean_sl_standard_deviation(io_error_sample_gpa, curves_dim)
	io_error_analysis_report(io_error_mean_group_error, io_error_error)
	io_error_manova(paste0('measurement_error', slm), output_dir, io_error_sample_gpa, io_error_sample_groups)

	result <- list(io_error_mean_group_error=io_error_mean_group_error,
			io_error_error=io_error_error,
			ration=io_error_mean_group_error/io_error_error)
	return(result)
}
