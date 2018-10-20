# TODO: Add comment
#
# Author: vajicek
###############################################################################

source("base/processcurves.R")
source("base/base.R")

use_library('unittest')

ok(1 == point_distance(c(0, 0), c(1, 0)),
  "distance (0, 0) and (1, 0)")

ok(sqrt(3) == point_distance(c(0, 0, 0), c(1, 1, 1)),
  "distance (0, 0, 0) and (1, 1, 1)")


ok(all(c(2,1) == compute_lengths(array(c(
  # first
  0, 0,
  2, 0,
  # second curve
  0, 0,
  0, 1),
  c(2, 2, 2)))),
  "two 2d curves distance")
