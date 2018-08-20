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
  variability <- compute_variability(pca)
  significant_count <- broken_stick_criterium(variability)
  return(list(significant_count=significant_count,
    length=length(variability),
    variability=variability,
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
  return(list(parameter=matrix(model$coefficients), model=model, rsq=summary(model)$r.squared))
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

plot_all_together <- function(data1, data2, params) {
  pdf(params$filename, width=params$width, height=params$height)
  for (line_no in  1:dim(data1)[1]) {
    plot(c(), c(), type='n',
      bty="n",
      xaxt='n',
      yaxt='n',
      xlab="",
      ylab="",
      xlim=params$xlim, ylim=params$ylim)
    grid(5, 5, lwd = 3)
    x1 <- data1[line_no, (1:dim(data1)[2]) %% 2 == 1]
    y1 <- data1[line_no, (1:dim(data1)[2] + 1) %%2 == 1]
    x2 <- data2[line_no, (1:dim(data2)[2]) %% 2 == 1]
    y2 <- data2[line_no, (1:dim(data2)[2] + 1) %%2 == 1]
    lines(x1, y1, type='l', lwd=1)
    lines(x2, y2, type='l', lwd=2)
  }
  dev.off()
}

plot_vector <- function(input, shift, lwd) {
  xy <- matrix(input, nrow=2)
  lines(xy[2, ] + shift, -xy[1, ], type='l', lwd=lwd)
}

plot_predicted_curves <- function(predictor1, predictor2, predicted1, predicted2, params) {
  #pdf(paste0(params$filename, '.pdf'), width=params$width * 0.393701, height=params$height * 0.393701 )
  png(paste0(params$filename, '.png'), width=params$width, height=params$height, units="cm", res=1200)
  par(mar=c(1, 1, 1, 1)*0.5)
  plot(c(), c(), type='n',
    bty="n",
    xaxt='n',
    yaxt='n',
    xlab="",
    ylab="",
    #xlim=c(params$ylim[1] - 2*params$plotshift, params$ylim[2] + 2 * params$plotshift),
    xlim=params$xlim,
    ylim=params$ylim)
  grid(params$grid[1], params$grid[2], lwd = params$grid[3])
  plot_vector(predictor1, params$plotshift, 0.5)
  plot_vector(predictor2, params$plotshift, 1.5)
  plot_vector(predicted1, -params$plotshift, 0.5)
  plot_vector(predicted2, -params$plotshift, 1.5)
  dev.off()
}

plot_pair <- function(minus_shape, plus_shape, params) {
  png(paste0(params$filename, '.png'), width=params$width, height=params$height, units="cm", res=1200)
  par(mar=c(1, 1, 1, 1)*0.5)
  plot(c(), c(), type='n',
    bty="n",
    xaxt='n',
    yaxt='n',
    xlab=params$xlab,
    ylab=params$ylab,
    xlim=params$xlim,
    ylim=params$ylim)
  grid(params$grid[1], params$grid[2], lwd = params$grid[3])
  plot_vector(minus_shape, 0.0, 0.5)
  plot_vector(plus_shape, 0.0, 1.5)
  dev.off()
}

lmr_score_predict <- function(model, newdata) {
  newdata <- data.frame(t(matrix(newdata)))
  names(newdata) <- tail(names(model$coefficients), -1)
  return(predict(model, newdata))
}

lmr_score_predict_manual <- function(model, newdata) {
  # corresponds to predict(model, newdata)
  param <- matrix(model$coefficients)
  b0 <- param[1]
  b1 <- matrix(param[2:dim(param)[1],])
  b1[is.na(b1)] <- 0
  predicted_val <- b0 + t(b1) %*% matrix(newdata)
  return(predicted_val)
}

lmr_score_model_predict <- function(mlr_score_model, score) {
  predicted_pca_score <- rep(0, mlr_score_model$length)
  rsq <- c()
  for(pca_i in 1:length(mlr_score_model$score_model)) {
    pca_i_val <- lmr_score_predict(mlr_score_model$score_model[[pca_i]]$model, score)
    predicted_pca_score[pca_i] <- pca_i_val * mlr_score_model$score_model[[pca_i]]$rsq
  }
  return(list(score=predicted_pca_score))
}

dump_components_weights <- function(mlr_score_model) {
  print("Components rsq: ")
  for (pca_no in  1:mlr_score_model$significant_count) {
    print(mlr_score_model$score_model[[pca_no]]$rsq)
  }
}

plot_shape_change <- function(pca_model, pca_params, name) {
  for (pca_no in 1:pca_model$significant_count) {
    soft_pca_score <- rep(0, pca_model$length)

    soft_pca_score[pca_no] <- sd(pca_model$score[,pca_no]) * pca_params$sdtimes
    soft_plus_shape <- pca_shape_predict(pca_model, soft_pca_score, 1)

    soft_pca_score[pca_no] <- -1 * sd(pca_model$score[,pca_no]) * pca_params$sdtimes
    soft_minus_shape <- pca_shape_predict(pca_model, soft_pca_score, 1)

    pca_params$filename <- file.path(pca_params$target_dir, paste0(
      pca_params$part, '_', name,
      '_pc', toString(pca_no),
      '_', toString(pca_params$sdtimes), 'sd'))
    plot_pair(soft_minus_shape$coords, soft_plus_shape$coords, pca_params)
  }
}

plot_pca_shape_change <- function(data, name, shape_var_params) {
  pca_result <- compute_pca(data)
  write.table(pca_result$score, file = file.path(shape_var_params$target_dir,
    paste0(shape_var_params$part, '_', name, '_pca_score.csv')))
  write.table(pca_result$variability, file = file.path(shape_var_params$target_dir,
      paste0(shape_var_params$part, '_', name, '_pca_variability.csv')))
  plot_shape_change(pca_result, shape_var_params, name)
}

plot_pca_predict <- function(soft, hard, params) {
  pca_soft <- compute_pca(soft)
  pca_hard <- compute_pca(hard)

  mlr_score_model=list(score_model=list(), length=pca_soft$length, significant_count=pca_soft$significant_count)
  for (pca_no in  1:pca_soft$significant_count) {
    mlr_pca_soft_i <- compute_mlr(pca_soft$score[,pca_no], pca_hard$score[,1:pca_hard$significant_count])
    mlr_score_model$score_model[[pca_no]] <- mlr_pca_soft_i
  }

  print("MODEL CREATED....")
  dump_components_weights(mlr_score_model)

  for (pca_no in 1:pca_hard$significant_count) {
    hard_pca_score <- rep(0, pca_hard$length)

    # plus
    hard_pca_score[pca_no] <- sd(pca_hard$score[,pca_no]) * params$sdtimes
    hard_plus_shape <- pca_shape_predict(pca_hard, hard_pca_score, 1)
    soft_plus_prediction <- lmr_score_model_predict(mlr_score_model, hard_pca_score)
    soft_plus_shape <- pca_shape_predict(pca_soft, soft_plus_prediction$score, 1)

    #minus
    hard_pca_score[pca_no] <- -1 * sd(pca_hard$score[,pca_no]) * params$sdtimes
    hard_minus_shape <- pca_shape_predict(pca_hard, hard_pca_score, 1)
    soft_minus_prediction <- lmr_score_model_predict(mlr_score_model, hard_pca_score)
    soft_minus_shape <- pca_shape_predict(pca_soft, soft_minus_prediction$score, 1)

    # dual plot
    params$filename <- file.path(params$target_dir, paste0('plot_', params$part, '_pc', toString(pca_no)))
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
  return(list(ylim=c(-0.3, 0.3), xlim=c(-0.5, 0.5)))
}

pca_predict <- function(part, target_dir) {
  soft <- load_data(paste0("data/", part, "_soft.csv"))
  hard <- load_data(paste0("data/", part, "_hard.csv"))

  # input data plots
  plot_all_profiles(soft, list(width=8, height=8, filename=file.path(target_dir, paste0('all_', part, '_soft.pdf')), xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)))
  plot_all_profiles(hard, list(width=8, height=8, filename=file.path(target_dir, paste0('all_', part, '_hard.pdf')), xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)))
  plot_all_together(soft, hard,
    list(width=8,
      height=8,
      filename=file.path(target_dir, paste0('all_', part, '_both.pdf')),
      xlim=c(-0.55, 0.55), ylim=c(-0.55, 0.55)
    )
  )

  # pca shape variation plots
  shape_var_params = list(
    width=3.214,
    height=3.214,
    part=part,
    target_dir=target_dir,
    filename='result.pdf',
    sdtimes=3.0,
    plotshift=0.25,
    xlim=c(-0.5, 0.5),
    ylim=c(-0.5, 0.5),
    grid=c(5, 5, 1.0))
  plot_pca_shape_change(hard, 'hard', shape_var_params)
  plot_pca_shape_change(soft, 'soft', shape_var_params)

  # pca predict
  plot_pca_predict(soft, hard, list(
    width=4.5,
    height=3.214,
    part=part,
    target_dir=target_dir,
    filename='result.pdf',
    sdtimes=3.0,
    plotshift=0.25,
    xlim=c(-0.7, 0.7),
    ylim=c(-0.5, 0.5),
    grid=c(7, 5, 1.0)))
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
  )

  opt = parse_args(OptionParser(option_list=option_list))
  pca_predict(opt$part, opt$output)
}

main()
