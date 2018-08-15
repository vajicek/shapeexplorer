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


compute_pca <- function(data) {
  pca <- prcomp(data, retx=TRUE)
  print(sum(pca$sdev))
  variability <- compute_variability(pca)
  significant_count <- broken_stick_criterium(variability)
  return(list(significant_count=significant_count,
    length=length(variability),
    data=data,
    score=pca$x,
    loadings=pca$rotation,
    mean=colMeans(data)))
}

pca_shape_predict <- function(pca, score, scale) {
  coords <- pca$mean + pca$loadings %*% score * scale
  return(list(coords=coords, score=score))
}

compute_mlr <- function(dependent_var, independent_vars) {
  model <- lm(dependent_var~., data.frame(independent_vars))
  return(list(parameter=matrix(model$coefficients), fit=model))
}

plot_all_profiles <- function(data, params) {
  pdf(params$filename, width=params$width, height=params$height)
  plot(c(), c(), type='n',
    bty="n",
    xaxt='n',
    yaxt='n',
    xlab="",
    ylab="",
    xlim=params$xlim, ylim=params$ylim)
  grid(5, 5, lwd = 3)
  for (line_no in  1:dim(data)[1]) {
    x <- data[line_no, (1:dim(data)[2]) %% 2 == 1]
    y <- data[line_no, (1:dim(data)[2] + 1) %%2 == 1]
    lines(x, y, type='l')
  }
  dev.off()
}

plot_vector <- function(input, shift, lwd) {
  xy <- matrix(input, nrow=2)
  lines(xy[2, ] + shift, -xy[1, ], type='l', lwd=lwd)
}

plot_predicted_curves <- function(predictor1, predictor2, predicted1, predicted2, params) {
  pdf(params$filename, width=params$width, height=params$height)
  par(mar=c(1, 1, 1, 1)*0.5)
  plot(c(), c(), type='n',
    bty="n",
    xaxt='n',
    yaxt='n',
    xlab="",
    ylab="",
    xlim=c(params$ylim[1] - 2*params$plotshift, params$ylim[2] + 2*params$plotshift), ylim=params$xlim)
  print(params$ylim)
  grid(5, 5, lwd = 3)
  plot_vector(predictor1, params$plotshift, 1)
  plot_vector(predictor2, params$plotshift, 2)
  plot_vector(predicted1, -params$plotshift, 1)
  plot_vector(predicted2, -params$plotshift, 2)
  dev.off()
}

lmr_score_predict <- function(param, base) {
  # corresponds to predict(model, newdata)
  b0 <- param[1]
  b1 <- matrix(param[2:dim(param)[1],])
  b1[is.na(b1)] <- 0
  predicted_val <- b0 + t(b1) %*% matrix(base)
  return(predicted_val)
}

lmr_score_model_predict <- function(mlr_score_model, score) {
  predicted_pca_score <- rep(0, mlr_score_model$length)
  for(pca_i in 1:length(mlr_score_model$score_model)) {
    pca_i_val <- lmr_score_predict(mlr_score_model$score_model[[pca_i]]$parameter, score)
    rsq <- summary(mlr_score_model$score_model[[pca_i]]$fit)$r.squared
    #print(rsq)
    predicted_pca_score[pca_i] <- pca_i_val
    # * rsq
  }
  return(predicted_pca_score)
}

plot_pca_predict <- function(soft, hard, params) {
  pca_soft <- compute_pca(soft)
  pca_hard <- compute_pca(hard)
  mlr_score_model=list(score_model=list(), length=pca_soft$length)
  for (pca_no in  1:pca_soft$significant_count) {
    mlr_pca_soft_i <- compute_mlr(pca_soft$score[,pca_no], pca_hard$score)
    mlr_score_model$score_model[[pca_no]] <- mlr_pca_soft_i
  }

  print("MODEL CREATED....")

  for (pca_no in 1:pca_hard$significant_count) {
    hard_pca_score <- rep(0, pca_hard$length)

    # plus
    hard_pca_score[pca_no] <- sd(pca_hard$score[,pca_no]) * params$sdtimes
    hard_plus_shape <- pca_shape_predict(pca_hard, hard_pca_score, 1)
    soft_plus_score <- lmr_score_model_predict(mlr_score_model, hard_pca_score)
    soft_plus_shape <- pca_shape_predict(pca_soft, soft_plus_score, 1)

    #minus
    hard_pca_score[pca_no] <- -1 * sd(pca_hard$score[,pca_no]) * params$sdtimes
    hard_minus_shape <- pca_shape_predict(pca_hard, hard_pca_score, 1)
    soft_minus_score <- lmr_score_model_predict(mlr_score_model, hard_pca_score)
    soft_minus_shape <- pca_shape_predict(pca_soft, soft_minus_score, 1)

    # dual plot
    params$filename <- file.path(params$target_dir, paste0('plot_', params$part, '_pc', toString(pca_no), '.pdf'))
    plot_predicted_curves(hard_minus_shape$coords,
      hard_plus_shape$coords,
      soft_minus_shape$coords,
      soft_plus_shape$coords,
      params)
  }
}

get_extents <- function(soft, hard) {
  soft_xx <- soft[, (1:dim(soft)[2]) %% 2 == 1]
  soft_yy <- soft[, (1:dim(soft)[2] + 1) %%2 == 1]
  soft_ext_x <- c(min(soft_xx), max(soft_xx))
  soft_ext_y <- c(min(soft_yy), max(soft_yy))

  hard_xx <- hard[, (1:dim(hard)[2]) %% 2 == 1]
  hard_yy <- hard[, (1:dim(hard)[2] + 1) %%2 == 1]
  hard_ext_x <- c(min(hard_xx), max(hard_xx))
  hard_ext_y <- c(min(hard_yy), max(hard_yy))
  #return(list(ylim=c(soft_ext_y[1], hard_ext_y[2]), xlim=c(-0.5, 0.5)))
  return(list(ylim=c(-0.5, 0.5), xlim=c(-0.5, 0.5)))
}

pca_predict <- function(part, target_dir) {
  soft <- load_data(paste0("data/", part, "_soft.csv"))
  hard <- load_data(paste0("data/", part, "_hard.csv"))
  plot_all_profiles(soft, list(width=8, height=8, filename=file.path(target_dir, paste0('all_', part, '_soft.pdf')), xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)))
  plot_all_profiles(hard, list(width=8, height=8, filename=file.path(target_dir, paste0('all_', part, '_hard.pdf')), xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)))
  exts <- get_extents(hard, soft)
  plot_pca_predict(soft, hard, list(width=8, height=8, part=part, target_dir=target_dir, filename='result.pdf', sdtimes=3.0, plotshift=0.2, xlim=exts$xlim, ylim=exts$ylim))
}

test_plot <- function() {
  predictor = matrix(c(2, 4, 3, 1, 5, 7), ncol=2)
  predicted = matrix(c(2, 3, 0, 1, 7, 3), ncol=2)
  plot_predicted_curves(predictor, predictor, predicted, predicted, list(filename="plot.pdf", plotshift=3))
}

test_predict <- function(){
  pca_predict('koren_nosu', '/home/vajicek/Dropbox/krivky_mala/clanek/GRAFY/predikce/')
}

main <- function() {
  # command-line interface
  option_list = list(
    make_option(c("--output"), default="", action="store"),
    make_option(c("--part"), default="", action="store")
  );

  opt = parse_args(OptionParser(option_list=option_list))
  pca_predict(opt$part, opt$output)
}

main()
