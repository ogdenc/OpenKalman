/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Files relating to adapters.
 *
 * \dir adapters/details
 * \brief Support files for adapters.
 *
 * \dir adapters/interfaces
 * \brief Interfaces for adapters.
 *
 * \dir adapters/tests
 * \brief Tests for adapters.
 *
 * \file
 * Includes header files for all adapter types.
 */

#ifndef OPENKALMAN_ADAPTERS_HPP
#define OPENKALMAN_ADAPTERS_HPP

#include "ConstantAdapter.hpp"
#include "DiagonalAdapter.hpp"
#include "HermitianAdapter.hpp"
#include "TriangularAdapter.hpp"
#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"
#include "VectorSpaceAdapter.hpp"

#include "linear-algebra/adapters/interfaces/adapters-interface.hpp"

#include "linear-algebra/adapters/details/adapters-arithmetic.hpp"

#endif OPENKALMAN_ADAPTERS_HPP
