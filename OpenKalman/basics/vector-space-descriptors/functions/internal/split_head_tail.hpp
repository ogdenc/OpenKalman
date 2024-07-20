/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref split_head_tail function.
 */

#ifndef OPENKALMAN_SPLIT_HEAD_TAIL_HPP
#define OPENKALMAN_SPLIT_HEAD_TAIL_HPP

#include <type_traits>


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Split a \ref vector_space_descriptor object into head and tail parts.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor D>
#else
  template<typename D, std::enable_if_t<fixed_vector_space_descriptor<D>, int> = 0>
#endif
  constexpr auto
  split_head_tail(const D& d)
  {
    static_assert(dimension_size_of_v<head_of_t<D>> == 1);
    return std::tuple {head_of_t<D>{}, tail_of_t<D>{}};
  }


  /**
   * \internal
   * \overload
   */
#ifdef __cpp_concepts
  template<dynamic_vector_space_descriptor D> requires euclidean_vector_space_descriptor<D>
#else
  template<typename D, std::enable_if_t<dynamic_vector_space_descriptor<D> and euclidean_vector_space_descriptor<D>, int> = 0>
#endif
  constexpr auto
  split_head_tail(D&& d)
  {
    return std::tuple {Dimensions<1>{}, Dimensions{get_dimension_size_of(d) - 1}};
  }


  /**
   * \internal
   * \overload
   */
  template<typename...S>
  constexpr auto
  split_head_tail(const DynamicDescriptor<S...>& d)
  {
    return d.split_head_tail();
  }


} // namespace OpenKalman::internal


#endif //OPENKALMAN_SPLIT_HEAD_TAIL_HPP
