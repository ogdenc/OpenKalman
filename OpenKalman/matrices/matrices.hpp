/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Files relating to matrices.
 *
 * \dir matrices/details
 * \brief Support files for matrices.
 *
 * \file
 * A meta-header file including all the headers relating to OpenKalman typed matrices.
 */

#ifndef OPENKALMAN_MATRICES_HPP
#define OPENKALMAN_MATRICES_HPP

#include "adapters/Mean.hpp"
#include "adapters/EuclideanMean.hpp"
#include "adapters/Matrix.hpp"
#include "adapters/details/typed-matrix-overloads.hpp"
#include "adapters/details/typed-matrix-arithmetic.hpp"

#include "adapters/details/to_covariance_nestable.hpp"
#include "adapters/details/CovarianceBase1.hpp"
#include "adapters/details/CovarianceBase2.hpp"
#include "adapters/details/CovarianceBase3Impl.hpp"
#include "adapters/details/CovarianceBase3-1.hpp"
#include "adapters/details/CovarianceBase3-2.hpp"
#include "adapters/details/CovarianceBase4.hpp"
#include "adapters/details/CovarianceImpl.hpp"
#include "adapters/Covariance.hpp"
#include "adapters/SquareRootCovariance.hpp"
#include "adapters/details/covariance-overloads.hpp"
#include "adapters/details/covariance-arithmetic.hpp"

#endif //OPENKALMAN_MATRICES_HPP
