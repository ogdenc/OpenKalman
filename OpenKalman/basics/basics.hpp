/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir
 * \brief Basic definitions for OpenKalman as a whole.
 *
 * \dir compatibility
 * \brief Definitions for compatibility with c++17 or other legacy versions of c++.
*
 * \dir classes
 * \brief Classes for general use in the library.
 *
 * \file
 * \brief Basic definitions for OpenKalman as a whole.
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
 * \brief Namespace for recreating or updating elements of the Standard Template Library.
 */
namespace OpenKalman::stdcompat {}

#include "compatibility/language-features.hpp"
#include "compatibility/core-concepts.hpp"
#include "compatibility/internal/exposition.hpp"
#include "compatibility/common.hpp"
#include "compatibility/comparison.hpp"
#include "compatibility/object-concepts.hpp"
#include "compatibility/invoke.hpp"
#include "compatibility/callable-concepts.hpp"

#include "compatibility/internal/movable_box.hpp"
#include "compatibility/iterator.hpp"
#include "compatibility/ranges.hpp"

#include "global-definitions.hpp"

#include "compatibility/internal/generalized_std_get.hpp"

#endif
