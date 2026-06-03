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

#include "constant_adapter.obsolete.hpp"

#include "internal/ElementAccessor.hpp"
#include "internal/adapter_base.hpp"

#include "pattern_adapter.hpp"
#include "interfaces/VectorSpaceAdapter.hpp"

#include "to_diagonal_adapter.obsolete.hpp"
#include "hermitian_adapter.hpp"
#include "triangular_adapter.hpp"
#include "interfaces/adapters-interface.hpp"

#include "to_stat_space_adapter.hpp"
#include "from_stat_space_adapter.hpp"

#include "internal/LibraryWrapper.hpp"
#include "interfaces/LibraryWrapper.hpp"

#include "internal/FixedSizeAdapter.hpp"
#include "interfaces/FixedSizeAdapter.hpp"


#endif
