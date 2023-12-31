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
 * \brief Namespace for internal definitions, not intended for use outside of OpenKalman development.
 */
namespace OpenKalman::internal {}


/**
 * \internal
 * \brief The root namespace for OpenKalman interface types.
 */
namespace OpenKalman::interface {}


// global

#include "language-features.hpp"
#include "global-definitions.hpp"
#include "utils.hpp"


// values

#include "values/values.hpp"


// vector space descriptors

#include "vector-space-descriptors/vector-space-descriptors.hpp"


// forward definitions for objects, properties, and interfaces

#include "interfaces/interfaces-defined.hpp"
#include "internal/library_base.hpp"

#include "property-functions/property-functions.hpp"

#include "traits/forward-traits.hpp"
#include "forward-class-declarations.hpp"

#include "interfaces/default/indexible_object_traits.hpp"
#include "interfaces/default/library_interface.hpp"


// object functions

#include "functions/functions.hpp"


// properties and interfaces

#include "traits/traits.hpp"


// internal objects

#include "internal/ElementAccessor.hpp"

#include "internal/LibraryWrapper.hpp"
#include "interfaces/internal/LibraryWrapper.hpp"

#include "internal/SelfContainedWrapper.hpp"
#include "interfaces/internal/SelfContainedWrapper.hpp"

#include "internal/FixedSizeAdapter.hpp"
#include "interfaces/internal/FixedSizeAdapter.hpp"

#include "internal/MatrixBase.hpp"

#include "internal/TypedMatrixBase.hpp"


#endif //OPENKALMAN_BASICS_HPP
