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
 * \brief Basic definitions for OpenKalman as a whole.
 *
 * \file
 * \brief Basic forward definitions for OpenKalman as a whole.
 * \details This should be included by any OpenKalman file, including interface files.
 */


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


#ifndef OPENKALMAN_BASICS_HPP
#define OPENKALMAN_BASICS_HPP

#include "language-features.hpp"
#include "global-definitions.hpp"
#include "utils.hpp"

#include "internal/scalar-traits.hpp"
#include "scalar-types.hpp"
#include "functions/scalar_functions.hpp"
#include "internal/math_constexpr.hpp"

#include "index-descriptors/index-descriptors.hpp"

#include "IndexibleObjectTraits.hpp"
#include "LibraryRoutines.hpp"

#include "forward-traits.hpp"
#include "forward-class-declarations.hpp"

#include "functions/functions.hpp"

#include "traits.hpp"

#include "internal/MatrixBase.hpp"
#include "internal/TypedMatrixBase.hpp"
#include "internal/ElementAccessor.hpp"

#include "internal/FixedSizeAdapter.hpp"

#endif //OPENKALMAN_BASICS_HPP
