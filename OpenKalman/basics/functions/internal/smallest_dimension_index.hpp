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
 * \file
 * \brief Definition for \ref internal::smallest_dimension_index function.
 */

#ifndef OPENKALMAN_SMALLEST_DIMENSION_INDEX_HPP
#define OPENKALMAN_SMALLEST_DIMENSION_INDEX_HPP

namespace OpenKalman::detail
{
  template<typename Arg, std::size_t I, std::size_t...Is>
  constexpr auto
  smallest_dimension_index_impl(const Arg& arg, std::index_sequence<I, Is...>)
  {
    if constexpr (sizeof...(Is) == 0)
    {
      return std::integral_constant<std::size_t, I>{};
    }
    else
    {
      auto tail_min_ix = smallest_dimension_index_impl(arg, std::index_sequence<Is...>{});
      if constexpr (static_index_value<decltype(tail_min_ix)>)
      {
        constexpr auto tail_min_ix_value = std::decay_t<decltype(tail_min_ix)>::value;
        if constexpr (not dynamic_dimension<Arg, I> and not dynamic_dimension<Arg, tail_min_ix_value>)
        {
          if constexpr (index_dimension_of_v<Arg, I> <= index_dimension_of_v<Arg, tail_min_ix_value>)
            return std::integral_constant<std::size_t, I>{};
          else
            return tail_min_ix;
        }
        else
        {
          if (get_index_dimension_of<I>(arg) <= get_index_dimension_of(arg, tail_min_ix)) return I;
          else return static_cast<std::size_t>(tail_min_ix);
        }
      }
      else
      {
        if (get_index_dimension_of<I>(arg) <= get_index_dimension_of(arg, tail_min_ix)) return I;
        else return static_cast<std::size_t>(tail_min_ix);
      }
    }
  }


  template<typename Arg, typename N>
  constexpr auto
  smallest_dimension_index_impl(const Arg& arg, const N& n)
  {
    std::size_t min_i = 0;
    for (std::size_t i = 0, min_size = std::numeric_limits<std::size_t>::max(); i < n; ++i)
    {
      auto d = get_index_dimension_of(arg, i);
      if (d < min_size) { min_i = i; min_size = d; }
    }
    return min_i;
  }

} // namespace OpenKalman::detail


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Return the \ref index_value of the index having the smallest dimension among the first N indices
   * \details If the dimensions are the same, this will return index 0.
   * \tparam N The number of indices to compare
   * \tparam Arg A matrix
   * \return An \ref index_value
   */
#ifdef __cpp_concepts
  template<indexible Arg, index_value N = std::integral_constant<std::size_t, 2>>
  constexpr index_value auto
#else
  template<typename Arg, typename N = std::integral_constant<std::size_t, 2>>
  constexpr auto
#endif
  smallest_dimension_index(const Arg& arg, const N& n = std::integral_constant<std::size_t, 2>{})
  {
    if constexpr (static_index_value<N>) static_assert(N::value > 0);

    if constexpr (dynamic_index_value<N>)
      return OpenKalman::detail::smallest_dimension_index_impl(arg, n);
    else if constexpr (N::value == 0)
      return std::integral_constant<std::size_t, 0>{};
    else
      return OpenKalman::detail::smallest_dimension_index_impl(arg, std::make_index_sequence<N::value>{});
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_SMALLEST_DIMENSION_INDEX_HPP
