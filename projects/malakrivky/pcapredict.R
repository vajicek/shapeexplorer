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
  pca <- prcomp(data, scale=FALSE, retx=TRUE)
  variability <- compute_variability(pca)
  significant_count <- broken_stick_criterium(variability)
  return(list(significant_count=significant_count,
    data=data,
    score=pca$x,
    loadings=pca$rotation,
    mean=colMeans(data)))
}

pca_shape_predict <- function(pca, score) {
  coords <- pca$mean + pca$loadings * score
  return(list(coords=coords), score=score)
}

compute_mlr <- function(dependent_var, independent_vars) {
  fit <- lm(dependent_var~independent_vars)
  return(list(parameter=matrix(fit$coefficients)))
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

plot_predicted_curves <- function(predictor1, predictor2, predicted1, predicted2, params) {
  pdf(params$filename, width=params$width, height=params$height)
  plot(c(), c(), type='n',
    bty="n",
    xaxt='n',
    yaxt='n',
    xlab="",
    ylab="",
    xlim=c(min(predictor1), max(predictor1)), ylim=c(min(predictor1), max(predictor1)))
  grid(5, 5, lwd = 3)
  lines(predictor1[, 1], predictor1[, 2], type='l')
  lines(predictor2[, 1], predictor2[, 2], type='l')
  lines(predicted1[, 1] + params$plotshift, predicted1[, 2], type='l')
  lines(predicted2[, 1] + params$plotshift, predicted2[, 2], type='l')
  dev.off()
}

lmr_score_predict <- function(param, base, scale) {
  b0 = param[1]
  b1 = matrix(param[2:dim(param)[1],])

  print(dim(b1))
  print(dim(base))
  print(dim(b1 %*% base))
  return(param)
  #return(b0 + b1 * base * scale)
}

plot_pca_predict <- function(soft, hard, params) {
  pca_soft <- compute_pca(soft)
  pca_hard <- compute_pca(hard)
  for (pca_no in  1:pca_soft$significant_count) {
    # KDYZ MI PRIJDE PCA_HARD, JAK Z TOHO UDELAT PCA1_SOFT (t.j. dot(param, hard_score))
    # MODEL PRO SOFT_i SCORE PODLE HARD SCORE
    mlr_pca_soft_i <- compute_mlr(pca_soft$score[,pca_no], pca_hard$score)

    #ZE VSECH HARD SCORE UDELAM VSECHNY SOFT_i_spec predikce
    # score prediction
    pca_hard_i_score_minus <- lmr_score_predict(mlr_pca_soft_i$parameter, pca_hard$score, -params$shift)
    pca_hard_i_score_plus <- lmr_score_predict(mlr_pca_soft_i$parameter, pca_hard$score, params$shift)

    # shape prediction
    hard_plus <- pca_shape_predict(pca_hard, pca_hard_i_score_plus)
    hard_minus <- pca_shape_predict(pca_hard, pca_hard_i_score_minus)
    soft_plus <- pca_shape_predict(pca_soft, soft_score_i_plus)
    soft_minus <- pca_shape_predict(pca_soft, soft_score_i_minus)

    # dual plot
    plot_predicted_curves(soft_plus, soft_minus, hard_plus, hard_minus, params)
  }
}

pca_predict <- function() {
  soft <- load_data("data/g_rhi_soft.csv")
  hard <- load_data("data/g_rhi_hard.csv")
  plot_all_profiles(soft, list(width=8, height=8, filename='soft.pdf', xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)))
  plot_all_profiles(hard, list(width=8, height=8, filename='hard.pdf', xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)))
  print(dim(soft))
  print(dim(hard))
  plot_pca_predict(soft, hard, list(filename='result.pdf', shift=1, plotshift=3))
}

test_plot <- function() {
  predictor = matrix(c(2, 4, 3, 1, 5, 7), ncol=2)
  predicted = matrix(c(2, 3, 0, 1, 7, 3), ncol=2)
  plot_predicted_curves(predictor, predictor, predicted, predicted, list(filename="plot.pdf", plotshift=3))
}

test_predict <- function(){
  pca_predict()
}


test_predict()
