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

#ifndef OPENKALMAN_SPECIAL_MATRICES_HPP
#define OPENKALMAN_SPECIAL_MATRICES_HPP

#include "ConstantAdapter.hpp"
#include "DiagonalMatrix.hpp"
#include "SelfAdjointMatrix.hpp"
#include "TriangularMatrix.hpp"
#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"

#include "details/special-matrix-traits.hpp"
#include "details/special-matrix-overloads.hpp"
#include "details/euclidean-overloads.hpp"
#include "details/special-matrix-arithmetic.hpp"

#endif //OPENKALMAN_SPECIAL_MATRICES_HPP
