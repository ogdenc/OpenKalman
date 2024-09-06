/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
    template<typename T, typename C, typename V0, typename V1, typename...Vs>
    static constexpr decltype(auto)
    constant_diagonal_of_impl(C&& c, V0&& v0, V1&& v1, const Vs&...vs)
    {
      auto d0 = internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(std::forward<V0>(v0), std::forward<V1>(v1));
      return make_constant<T>(std::forward<C>(c), d0, vs...);
    }
  } // namespace detail


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
    else if constexpr (constant_matrix<Arg>)
    {
      return detail::constant_diagonal_of_impl<Arg>(
        constant_coefficient {std::forward<Arg>(arg)},
        std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{Dimensions<1>{}, Dimensions<1>{}}));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return detail::constant_diagonal_of_impl<Arg>(
        constant_diagonal_coefficient {std::forward<Arg>(arg)},
        std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{Dimensions<1>{}, Dimensions<1>{}}));
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
