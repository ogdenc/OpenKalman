/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref is_vector function.
 */

#ifndef OPENKALMAN_IS_VECTOR_HPP
#define OPENKALMAN_IS_VECTOR_HPP

#include "linear-algebra/traits/index_count.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t N, std::size_t i = 0, typename T>
    constexpr bool get_is_vector_impl(const T& t)
    {
      if constexpr (i < index_count_v<T>)
      {
        return values::operation(
          std::logical_and{},
          values::operation(
            std::logical_or{},
            std::bool_constant<N == i>{},
            values::operation(std::equal_to{}, get_index_extent<i>(t), std::integral_constant<std::size_t, 1>{})
            ),
          get_is_vector_impl<N, i + 1>(t));
      }
      else
      {
        return std::true_type {};
      }
    }
  }


  /**
   * \brief Return true if T is a \ref vector at runtime.
   * \details In this context, a vector is an object in which every index but one is 1D.
   * \tparam N An index designating the "large" index (e.g., 0 for a column vector, 1 for a row vector)
   * \sa vector
   */
#ifdef __cpp_concepts
  template<std::size_t N = 0, indexible T>
  constexpr internal::boolean_testable auto
#else
  template<std::size_t N = 0, typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  is_vector(const T& t)
  {
    if constexpr (index_count_v<T> == stdex::dynamic_extent)
    {
      for (std::size_t i = 0; i < count_indices(t); ++i)
        if (N != i and get_index_extent(t, i) != 1) return false;
      return true;
    }
    else
    {
      return detail::get_is_vector_impl<N>(t);
    }
  }


}

#endif
