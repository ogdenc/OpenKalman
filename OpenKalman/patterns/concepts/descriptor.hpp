/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref patterns::descriptor.
 */

#ifndef OPENKALMAN_PATTERNS_DESCRIPTOR_HPP
#define OPENKALMAN_PATTERNS_DESCRIPTOR_HPP

#include "values/values.hpp"
#include "patterns/interfaces/pattern_descriptor_traits.hpp"

namespace OpenKalman::patterns
{
  /**
   * \brief T is an atomic (non-separable or non-composite) grouping of \ref patterns::pattern objects.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept descriptor =
#else
  constexpr bool descriptor =
#endif
    interface::pattern_descriptor_traits<std::decay_t<stdex::unwrap_ref_decay_t<T>>>::is_specialized or
    values::index<T>;

}

#endif
