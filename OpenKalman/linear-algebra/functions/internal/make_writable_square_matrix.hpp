/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref make_writable_square_matrix function.
 */

#ifndef OPENKALMAN_MAKE_WRITABLE_SQUARE_MATRIX_HPP
#define OPENKALMAN_MAKE_WRITABLE_SQUARE_MATRIX_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Given inputs to a rank update function, return a writable square matrix
   */
  template<typename U, typename A>
  constexpr decltype(auto)
  make_writable_square_matrix(A&& a)
  {
    constexpr auto dim = not dynamic_dimension<A, 0> ? index_dimension_of_v<A, 0> :
                         not dynamic_dimension<A, 1> ? index_dimension_of_v<A, 1> : index_dimension_of_v<U, 0>;
    if constexpr (writable<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (not has_dynamic_dimensions<A> or dim == dynamic_size)
    {
      return to_dense_object(std::forward<A>(a));
    }
    else
    {
      constexpr auto d = std::integral_constant<std::size_t, dim>{};
      auto ret {make_dense_object<A>(d, d)};
      ret = std::forward<A>(a);
      return ret;
    }
  }

}

#endif
