/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * @file typed-matrices.hpp
 * A meta-header file including all the headers relating to OpenKalman typed matrices.
 */

#ifndef OPENKALMAN_MATRICES_HPP
#define OPENKALMAN_MATRICES_HPP

#include "details/ElementSetter.hpp"
//
#include "details/MatrixBase.hpp"
#include "details/TypedMatrixBase.hpp"
#include "Mean.hpp"
#include "EuclideanMean.hpp"
#include "Matrix.hpp"
#include "details/typed-matrix-overloads.hpp"
#include "details/typed-matrix-arithmetic.hpp"
//
#include "details/ConvertBaseMatrix.hpp"
#include "details/CovarianceBaseBase.hpp"
#include "details/CovarianceBase.hpp"
#include "Covariance.hpp"
#include "SquareRootCovariance.hpp"
#include "details/covariance-overloads.hpp"
#include "details/covariance-arithmetic.hpp"

#endif //OPENKALMAN_MATRICES_HPP
