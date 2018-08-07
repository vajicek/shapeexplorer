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
  #TODO!!
  return(list(significant_count=1, data=data, score=(), loadings=(), mean=c()))
}

pca_predict <- function(pca, score) {
  coords <- pca$mean + pca$loadings * score
  return(list(coords=coords), score=score)
}

compute_mlr <- function(dependent_var, independent_vars) {
  #TODO!!
  return(list(parameter=c()))
}

plot_predicted_curves <- function(predictor1, predictor2, predicted1, predicted2, params) {
  pdf(params$filename)
  plot(predictor1)
  plot(predictor2)
  plot(predicted1)
  plot(predicted2)
  dev.off()
}


plot_pca_predict <- function(soft, hard, params) {
  pca_soft <- compute_pca(soft)
  pca_hard <- compute_pca(hard)
  for (pca_no in  1:pca_soft$significant_count) {}
    mlr_pca_soft_i <- compute_mlr(pca_soft$score[pca_no,], pca_hard$score)

    # score prediction
    pca_hard_i_score_minus <- mlr_pca_soft_i$parameter * pca_hard$score * (-1)
    pca_hard_i_score_plus <- mlr_pca_soft_i$parameter * pca_hard$score * (+1)

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
  plot_pca_predict(soft, hard, list(filename='result.pdf'))
}
