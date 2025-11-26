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
 * \dir details
 * \internal
 * \brief Support files for adapters.
*
 * \dir internal
 * \internal
 * \brief Support files for adapters.
 *
 * \dir interfaces
 * \brief Interfaces for adapters.
 *
 * \dir tests
 * \brief Tests for adapters.
 *
 * \file
 * Includes header files for all adapter types.
 */

#ifndef OPENKALMAN_ADAPTERS_HPP
#define OPENKALMAN_ADAPTERS_HPP

#include "constant_adapter.hpp"

#include "internal/ElementAccessor.hpp"
#include "internal/AdapterBase.hpp"

#include "pattern_adapter.hpp"
#include "interfaces/VectorSpaceAdapter.hpp"

#include "diagonal_adapter.hpp"
#include "HermitianAdapter.hpp"
#include "TriangularAdapter.hpp"
#include "interfaces/adapters-interface.hpp"

#include "ToEuclideanExpr.hpp"
#include "FromEuclideanExpr.hpp"

#include "internal/LibraryWrapper.hpp"
#include "interfaces/LibraryWrapper.hpp"

#include "internal/FixedSizeAdapter.hpp"
#include "interfaces/FixedSizeAdapter.hpp"


#endif
