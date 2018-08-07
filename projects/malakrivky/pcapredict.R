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

pca_predict <- function(pca, score) {
  coords <- pca$mean + pca$loadings * score
  return(list(coords=coords), score=score)
}

compute_mlr <- function(dependent_var, independent_vars) {
  fit <- lm(dependent_var~independent_vars)
  return(list(parameter=fit$coefficients))
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

plot_pca_predict <- function(soft, hard, params) {
  pca_soft <- compute_pca(soft)
  pca_hard <- compute_pca(hard)
  for (pca_no in  1:pca_soft$significant_count) {
    mlr_pca_soft_i <- compute_mlr(pca_soft$score[pca_no,], pca_hard$score)

    # score prediction
    pca_hard_i_score_minus <- mlr_pca_soft_i$parameter * pca_hard$score * (-params$shift)
    pca_hard_i_score_plus <- mlr_pca_soft_i$parameter * pca_hard$score * (+params$shift)

    # shape prediction
    hard_plus <- pca_predict(pca_hard, pca_hard_i_score_plus)
    hard_minus <- pca_predict(pca_hard, pca_hard_i_score_minus)
    soft_plus <- pca_predict(pca_soft, soft_score_i_plus)
    soft_minus <- pca_predict(pca_soft, soft_score_i_minus)

    # dual plot
    plot_predicted_curves(soft_plus, soft_minus, hard_plus, hard_minus, params)
  }
}

pca_predict <- function() {
  soft <- load_data("soft.csv")
  hard <- load_data("hard.csv")
  plot_pca_predict(soft, hard, list(filename='result.pdf', shift=1, plotshift=3))
}


predictor = matrix(c(2, 4, 3, 1, 5, 7), ncol=2)
predicted = matrix(c(2, 3, 0, 1, 7, 3), ncol=2)

plot_predicted_curves(predictor, predictor, predicted, predicted, list(filename="plot.pdf", plotshift=3))
