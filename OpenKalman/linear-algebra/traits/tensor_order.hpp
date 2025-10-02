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
 * \brief Definition of \ref tensor_order function.
 */

#ifndef OPENKALMAN_TENSOR_ORDER_HPP
#define OPENKALMAN_TENSOR_ORDER_HPP

#include "coordinates/coordinates.hpp"
#include "linear-algebra/traits/count_indices.hpp"
#include "linear-algebra/traits/get_index_extent.hpp"

namespace OpenKalman
{

  namespace detail
  {
    template<typename T>
    constexpr auto get_tensor_order_of_impl(std::index_sequence<>, const T& t) { return 0; }

    template<std::size_t I, std::size_t...Is, typename T>
    constexpr auto get_tensor_order_of_impl(std::index_sequence<I, Is...>, const T& t)
    {
      auto dim = get_index_extent<I>(t);
      if constexpr (values::fixed<decltype(dim)>)
      {
        constexpr std::size_t stat_dim = std::decay_t<decltype(dim)>::value;
        if constexpr (stat_dim == 0) return dim;
        else
        {
          auto next = get_tensor_order_of_impl(std::index_sequence<Is...> {}, t);
          if constexpr (stat_dim == 1) return next;
          else if constexpr (values::fixed<decltype(next)>)
            return std::integral_constant<std::size_t, 1_uz + std::decay_t<decltype(next)>::value> {};
          else
            return 1_uz + next;
        }
      }
      else
      {
        if (dim == 0) return dim;
        else
        {
          std::size_t next = get_tensor_order_of_impl(std::index_sequence<Is...> {}, t);
          if (dim == 1) return next;
          else return 1_uz + next;
        }
      }
    }
  }


  /**
   * \brief Return the tensor order of T (i.e., the number of indices of dimension greater than 1).
   * \details If T has any zero-dimensional indices, the tensor order is considered to be 0, based on the theory that
   * a zero-dimensional vector space has 0 as its only element, and 0 is a scalar value.
   * This may be subject to change.
   * \tparam T A matrix or array
   */
#ifdef __cpp_concepts
  template<indexible T>
  constexpr values::index auto
#else
  template<typename T, std::enable_if_t<indexible<T>, int> = 0>
  constexpr auto
#endif
  tensor_order(const T& t)
  {
    if constexpr (values::fixed<decltype(count_indices(t))>)
    {
      constexpr std::make_index_sequence<std::decay_t<decltype(count_indices(t))>::value> seq;
      return detail::get_tensor_order_of_impl(seq, t);
    }
    else
    {
      std::size_t count = 0;
      for (std::size_t i = 0; i < count_indices(t); ++i)
      {
        auto dim = get_index_extent(t, i);
        if (dim > 1) ++count;
        else if (dim == 0) return 0;
      }
      return count;
    }
  }


}

#endif
