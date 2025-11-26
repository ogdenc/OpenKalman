/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref max_tensor_order.
 */

#ifndef OPENKALMAN_MAX_TENSOR_ORDER_HPP
#define OPENKALMAN_MAX_TENSOR_ORDER_HPP

#include "linear-algebra/concepts/dimension_size_of_index_is.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t i, typename T>
    constexpr std::size_t max_tensor_order_impl(std::size_t result = 0)
    {
      if constexpr (i == 0) return result;
      else if constexpr (dimension_size_of_index_is<T, i - 1, 1>) return max_tensor_order_impl<i - 1, T>(result);
      else if constexpr (dimension_size_of_index_is<T, i - 1, 0>) return 0;
      else return max_tensor_order_impl<i - 1, T>(result + 1);
    }
  }


  /**
   * \brief The maximum number of indices of structure T of size other than 1 (including any dynamic indices).
   * \details If T has any zero-dimensional indices, the tensor order is considered to be 0, based on the theory that
   * a zero-dimensional vector space has 0 as its only element, and 0 is a scalar value.
   * (This may be subject to change.)
   * \tparam T A tensor (vector, matrix, etc.)
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct max_tensor_order
    : std::integral_constant<std::size_t, indexible<T> ? stdex::dynamic_extent : 0> {};


#ifdef __cpp_concepts
  template<typename T> requires (index_count_v<T> != std::dynamic_extent)
  struct max_tensor_order<T>
#else
  template<typename T>
  struct max_tensor_order<T, std::enable_if_t<index_count<T>::value != stdex::dynamic_extent>>
#endif
    : std::integral_constant<std::size_t, detail::max_tensor_order_impl<index_count_v<T>, T>()> {};


  /**
   * \brief helper template for \ref index_count.
   */
  template<typename T>
  static constexpr std::size_t max_tensor_order_v = max_tensor_order<T>::value;


}

#endif
