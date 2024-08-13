/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref diagonal_of function.
 */

#ifndef OPENKALMAN_DIAGONAL_OF_HPP
#define OPENKALMAN_DIAGONAL_OF_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, typename C, typename D0, std::size_t...Is>
    constexpr auto
    make_constant_column_vector(C&& c, D0&& d0, std::index_sequence<Is...>)
    {
      return make_constant<T>(std::forward<C>(c), std::forward<D0>(d0), Dimensions<Is==Is?1:1>{}...);
    }
  }


  /**
   * \brief Extract the main diagonal from a matrix.
   * \tparam Arg A 2D matrix, which may or may not be square
   * \returns Arg A column vector whose \ref vector_space_descriptor corresponds to the smallest-dimension index.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (index_count_v<Arg> == dynamic_size) or (index_count_v<Arg> <= 2)
  constexpr vector decltype(auto)
#else
  template<typename Arg, std::enable_if_t<(index_count_v<Arg> == dynamic_size or index_count_v<Arg> <= 2), int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    if constexpr (diagonal_adapter<Arg>)
    {
      return nested_object(std::forward<Arg>(arg));
    }
    else if constexpr (diagonal_adapter<Arg, 1>)
    {
      return transpose(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (one_dimensional<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (constant_matrix<Arg>)
      {
        return detail::make_constant_column_vector<Arg>(
          constant_coefficient{std::forward<Arg>(arg)},
          get_vector_space_descriptor(arg, internal::smallest_dimension_index(arg)),
          std::make_index_sequence<index_count_v<Arg> - 1>{});
      }
      else if constexpr (constant_diagonal_matrix<Arg>)
      {
        return detail::make_constant_column_vector<Arg>(
          constant_diagonal_coefficient {std::forward<Arg>(arg)},
          get_vector_space_descriptor(arg, internal::smallest_dimension_index(arg)),
          std::make_index_sequence<index_count_v<Arg> - 1>{});
      }
      else
      {
        return interface::library_interface<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
      }
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
