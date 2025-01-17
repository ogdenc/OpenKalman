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
 * \brief Definition for \ref maybe_has_suffix.
 */

#ifndef OPENKALMAN_MAYBE_HAS_SUFFIX_HPP
#define OPENKALMAN_MAYBE_HAS_SUFFIX_HPP

#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \brief T may have the suffix U, where T and U are sets of \ref vector_space_descriptor types.
   * \details
   * StaticDescriptor<Cs...> is a suffix of StaticDescriptor<C, Cs...>.
   * U is a suffix of T if U == T.
   * StaticDescriptor<> is a suffix of any \ref vector_space_descriptor.
   * The concept is satisfied if either T or U is a \ref dynamic_vector_space_descriptor,
   * because it cannot be known at compile time whether U is a suffix of T.
   * \par Example:
   * <code>prefix_of&lt;StaticDescriptor&lt;Axis&gt;, StaticDescriptor&lt;Axis, angle::Radians&gt;&gt;</code>
   * \sa internal::suffix_of, internal::maybe_prefix_of
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept maybe_has_suffix =
#else
  constexpr bool maybe_has_suffix =
#endif
    dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U> or (T{} <= U{});


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_MAYBE_HAS_SUFFIX_HPP
