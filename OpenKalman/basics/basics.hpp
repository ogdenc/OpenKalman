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


#include "compatibility/language-features.hpp"

#if __cplusplus < 202002L
#include "compatibility/common_reference.hpp"
#endif

#ifndef __cpp_lib_ranges
#include "compatibility/iterator.hpp"
#include "compatibility/ranges.hpp"
#endif

#if __cpp_lib_ranges < 202202L
#include "compatibility/views/range_adaptor_closure.hpp"
#endif

#ifndef __cpp_lib_ranges
#include "compatibility/views.hpp"
#endif


#include "global-definitions.hpp"

#include "classes/equal_to.hpp"
#include "classes/not_equal_to.hpp"
#include "classes/less.hpp"
#include "classes/less_equal.hpp"
#include "classes/greater.hpp"
#include "classes/greater_equal.hpp"


#endif //OPENKALMAN_BASICS_HPP
