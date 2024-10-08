/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
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

#include <type_traits>


namespace OpenKalman::internal
{
  /**
   * \brief T is a prefix of U, where T and U are sets of \ref fixed_vector_space_descriptor types.
   * \details If T is a prefix of U, then U is equivalent_to concatenating T with the remaining part of U.
   * C is a prefix of FixedDescriptor<C, Cs...> for any \ref fixed_vector_space_descriptor Cs.
   * T is a prefix of U if equivalent_to<T, U>.
   * FixedDescriptor<> is a prefix of any \ref fixed_vector_space_descriptor.
   * \par Example:
   * <code>prefix_of&lt;FixedDescriptor&lt;Axis&gt;, FixedDescriptor&lt;Axis, angle::Radians&gt;&gt;</code>
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of =
#else
  constexpr bool prefix_of =
#endif
    fixed_vector_space_descriptor<T> and fixed_vector_space_descriptor<U> and internal::is_prefix<T, U>::value;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_PREFIX_OF_HPP
