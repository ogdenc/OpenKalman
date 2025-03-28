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
 * \brief Definitions and interface for basic linear algebra.
 *
 * \file
 * \brief Basic forward definitions for OpenKalman as a whole.
 * \details This should be included by any OpenKalman file, including interface files.
 */

#ifndef OPENKALMAN_LINEAR_ALGEBRA_HPP
#define OPENKALMAN_LINEAR_ALGEBRA_HPP

#include "basics/basics.hpp"
#include "values/values.hpp"
#include "coordinates/coordinates.hpp"

// objects, properties, and interfaces

#include "interfaces/default/indexible_object_traits.hpp"
#include "interfaces/object-traits-defined.hpp"
#include "interfaces/default/library_interface.hpp"
#include "interfaces/library-interfaces-defined.hpp"

#include "traits/internal/library_base.hpp"

#include "property-functions/property-functions.hpp"

#include "concepts/concepts.hpp"
#include "traits/traits.hpp"
#include "adapters/internal/forward-class-declarations.hpp"

// object functions

#include "functions/functions.hpp"

// internal objects

#include "adapters/internal/ElementAccessor.hpp"

#include "adapters/internal/AdapterBase.hpp"

#include "adapters/internal/LibraryWrapper.hpp"
#include "adapters/interfaces/LibraryWrapper.hpp"

#include "adapters/internal/FixedSizeAdapter.hpp"
#include "adapters/interfaces/FixedSizeAdapter.hpp"

#include "adapters/VectorSpaceAdapter.hpp"
#include "adapters/interfaces/VectorSpaceAdapter.hpp"


#endif //OPENKALMAN_LINEAR_ALGEBRA_HPP
