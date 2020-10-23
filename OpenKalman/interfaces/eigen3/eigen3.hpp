/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_HPP
#define OPENKALMAN_EIGEN3_HPP


// Note: Requires >= Eigen 3.3.9 if compiling in c++20 mode! See Eigen Commit 7a0a2a500, which fixes issue #2012.
#include <Eigen/Dense>


#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#ifdef __clang__
#pragma clang diagnostic ignored "-Wunknown-attributes"
#endif
#endif


#include "typed-matrices/details/Traits.hpp"
#include "typed-matrices/details/ForwardDeclarations.hpp"
#include "coefficient-types/coefficient-types.hpp"
#include "details/EigenForwardDeclarations.hpp"

#include "details/EigenMatrixOverloads.hpp"

#include "details/EigenMatrixBase.hpp"
#include "details/EigenCovarianceBase.hpp"
#include "details/EigenMatrixTraits.hpp"

#include "details/EigenCholesky.hpp"
#include "typed-matrices/details/MatrixBase.hpp"

#include "EigenZero.hpp"
#include "EigenDiagonal.hpp"
#include "EigenSelfAdjointMatrix.hpp"
#include "EigenTriangularMatrix.hpp"
#include "details/EigenSpecialMatrixOverloads.hpp"

#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"
#include "details/EuclideanExprOverloads.hpp"

#include "details/EigenTraits.hpp"
#include "details/EigenEvaluators.hpp"

#include "typed-matrices/details/ElementSetter.hpp"


#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif //OPENKALMAN_EIGEN3_HPP
