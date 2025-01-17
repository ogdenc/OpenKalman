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
 * \brief Definition for \ref maybe_prefix_of.
 */

#ifndef OPENKALMAN_MAYBE_PREFIX_OF_HPP
#define OPENKALMAN_MAYBE_PREFIX_OF_HPP

#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/comparison-operators.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \brief T may be a prefix of U, where T and U are sets of \ref vector_space_descriptor types.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of StaticDescriptor<C, Cs...> for any \ref static_vector_space_descriptor Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * StaticDescriptor<> is a prefix of any \ref static_vector_space_descriptor.
   * The concept is satisfied if either T or U is a \ref dynamic_vector_space_descriptor,
   * because it cannot be known at compile time whether T is a prefix of U.
   * \par Example:
   * <code>prefix_of&lt;StaticDescriptor&lt;Axis&gt;, StaticDescriptor&lt;Axis, angle::Radians&gt;&gt;</code>
   * \sa internal::prefix_of
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept maybe_prefix_of =
#else
  constexpr bool maybe_prefix_of =
#endif
    dynamic_vector_space_descriptor<T> or dynamic_vector_space_descriptor<U> or (T{} <= U{});


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_MAYBE_PREFIX_OF_HPP
