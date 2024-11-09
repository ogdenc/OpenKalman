/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Native Eigen3 traits for Eigen3 tensor extensions
 */

#ifndef OPENKALMAN_EIGEN_NATIVE_TENSOR_TRAITS_HPP
#define OPENKALMAN_EIGEN_NATIVE_TENSOR_TRAITS_HPP

#include "interfaces/eigen/native-traits/eigen-native-traits.hpp"

#include "LibraryWrapper.hpp"
#include "FixedSizeAdapter.hpp"
#include "VectorSpaceAdapter.hpp"
#include "ConstantAdapter.hpp"
#include "HermitianAdapter.hpp"
#include "TriangularAdapter.hpp"
#include "DiagonalAdapter.hpp"
#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"
#include "Mean.hpp"
#include "Covariance.hpp"
#include "SquareRootCovariance.hpp"


#endif //OPENKALMAN_EIGEN_NATIVE_TENSOR_TRAITS_HPP
