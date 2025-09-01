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

#include "coordinates/coordinates.hpp"

#include "enumerations.hpp"

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

#include "functions/functions.hpp"

#include "adapters/adapters.hpp"


#endif
