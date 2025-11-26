/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Header file for compatibility definitions equivalent to those in header &lt;linalg&gt;
 */

#ifndef OPENKALMAN_COMPATIBILITY_LINALG_HPP
#define OPENKALMAN_COMPATIBILITY_LINALG_HPP

//#include "mdspan.hpp"

#ifdef __cpp_lib_linalg
#include <linalg>
namespace OpenKalman::stdex
{
  using namespace std::linalg;
}
#else

#define MDSPAN_IMPL_STANDARD_NAMESPACE OpenKalman
#define MDSPAN_IMPL_PROPOSED_NAMESPACE stdex
#define MDSPAN_IMPL_TRAIT(TRAIT, ...) TRAIT<__VA_ARGS__>::value
#include "std-lib-reference/linalg-reference-implementation/include/experimental/linalg"

#endif


#endif
