/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_H
#define OPENKALMAN_EIGEN3_H

#include <Eigen/Dense>

#include "variables/support/Traits.h"
#include "variables/support/ForwardDeclarations.h"
#include "variables/support/MatrixBase.h"
#include "variables/support/OpenKalman-coefficients.h"

#include "variables/interfaces/Eigen3/EigenForwardDeclarations.h"

#include "variables/interfaces/Eigen3/EigenMatrixBase.h"
#include "variables/interfaces/Eigen3/EigenCovarianceBase.h"

#include "variables/interfaces/Eigen3/EigenMatrixTraits.h"
#include "variables/interfaces/Eigen3/EigenMatrixOverloads.h"
#include "variables/interfaces/Eigen3/EigenZero.h"

#include "variables/interfaces/Eigen3/EigenSelfAdjointMatrix.h"
#include "variables/interfaces/Eigen3/EigenTriangularMatrix.h"
#include "variables/interfaces/Eigen3/EigenDiagonal.h"
#include "variables/interfaces/Eigen3/EigenSpecialMatrixOverloads.h"

#include "variables/interfaces/Eigen3/ToEuclideanExpr.h"
#include "variables/interfaces/Eigen3/FromEuclideanExpr.h"
#include "variables/interfaces/Eigen3/EuclideanExprOverloads.h"

#include "variables/interfaces/Eigen3/EigenTraits.h"
#include "variables/interfaces/Eigen3/EigenEvaluators.h"


#endif //OPENKALMAN_EIGEN3_H
