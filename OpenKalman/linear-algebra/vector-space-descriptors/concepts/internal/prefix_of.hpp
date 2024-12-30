/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref prefix_of.
 */

#ifndef OPENKALMAN_PREFIX_OF_HPP
#define OPENKALMAN_PREFIX_OF_HPP

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/is_prefix.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \brief T is a prefix of U, where T and U are sets of \ref static_vector_space_descriptor types.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of StaticDescriptor<C, Cs...> for any \ref static_vector_space_descriptor Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * StaticDescriptor<> is a prefix of any \ref static_vector_space_descriptor.
   * \par Example:
   * <code>prefix_of&lt;StaticDescriptor&lt;Axis&gt;, StaticDescriptor&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of =
#else
  constexpr bool prefix_of =
#endif
    static_vector_space_descriptor<T> and static_vector_space_descriptor<U> and
      static_cast<bool>(internal::is_prefix(T{}, U{}));


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_PREFIX_OF_HPP
