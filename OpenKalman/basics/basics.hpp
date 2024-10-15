/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Basic definitions for OpenKalman as a whole.
 *
 * \file
 * \brief Basic forward definitions for OpenKalman as a whole.
 * \details This should be included by any OpenKalman file, including interface files.
 */

#ifndef OPENKALMAN_BASICS_HPP
#define OPENKALMAN_BASICS_HPP


// namespaces

/**
 * \brief The root namespace for OpenKalman.
 */
namespace OpenKalman {}


/**
 * \internal
 * \brief The root namespace for OpenKalman interface types.
 */
namespace OpenKalman::interface {}


/**
 * \internal
 * \brief Namespace for internal definitions, not intended for use outside of OpenKalman development.
 */
namespace OpenKalman::internal {}


/**
 * \internal
 * \brief The root namespace for OpenKalman values (e.g., \ref constant_coefficient, \ref constant_diagonal_coefficient).
 */
namespace OpenKalman::values {}


/**
 * \brief The root namespace for OpenKalman \ref vector_space_descriptor objects.
 */
namespace OpenKalman::vector_space_descriptors {}
namespace OpenKalman { using namespace vector_space_descriptors; }


// global

#include "language-features.hpp"
#include "global-definitions.hpp"
#include "utils.hpp"


// values

#include "values/values.hpp"


// vector space descriptors

#include "vector-space-descriptors/vector-space-descriptors.hpp"


// objects, properties, and interfaces

#include "interfaces/object-traits-defined.hpp"
#include "interfaces/library-interfaces-defined.hpp"
#include "internal/library_base.hpp"

#include "property-functions/property-functions.hpp"

#include "traits/traits.hpp"
#include "forward-class-declarations.hpp"

#include "interfaces/default/indexible_object_traits.hpp"
#include "interfaces/default/library_interface.hpp"


// object functions

#include "functions/functions.hpp"


// function-dependent traits

#include "basics/traits/other-traits.hpp"


// internal objects

#include "internal/ElementAccessor.hpp"

#include "adapters/internal/AdapterBase.hpp"

#include "adapters/internal/LibraryWrapper.hpp"
#include "adapters/interfaces/LibraryWrapper.hpp"

#include "adapters/internal/FixedSizeAdapter.hpp"
#include "adapters/interfaces/FixedSizeAdapter.hpp"

#include "adapters/VectorSpaceAdapter.hpp"
#include "adapters/interfaces/VectorSpaceAdapter.hpp"


#endif //OPENKALMAN_BASICS_HPP
