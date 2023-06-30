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
    template<typename T, std::size_t...Is, typename C, typename D0>
    constexpr auto make_constant_column_vector(std::index_sequence<Is...>, C&& c, D0&& d0)
    {
      return make_constant_matrix_like<T>(std::forward<C>(c), std::forward<D0>(d0), Dimensions<static_cast<decltype(Is)>(1)>{}...);
    }
  }


  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A square matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg>
  constexpr dimension_size_of_index_is<1, 1> decltype(auto)
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe>, int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    constexpr std::make_index_sequence<max_indices_of_v<Arg> - 1> seq;

    if constexpr (diagonal_adapter<Arg>)
    {
      return nested_matrix(std::forward<Arg>(arg));
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (max_tensor_order_of_v<Arg> < max_indices_of_v<Arg>) // arg is a one-by-one matrix with at least one dynamic dimension
    {
      if (not get_is_square(arg)) throw std::invalid_argument{"Argument of diagonal_of is not a square matrix."};
      return detail::make_constant_column_vector<Arg>(seq, constant_coefficient {std::forward<Arg>(arg)}, Dimensions<1>{});
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto d = get_is_square(arg);
      if (not d) throw std::invalid_argument{"Argument of diagonal_of is not a square matrix."};
      return detail::make_constant_column_vector<Arg>(seq, constant_coefficient{std::forward<Arg>(arg)}, *d);
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      auto d = get_is_square(arg);
      if (not d) throw std::invalid_argument {"Argument of diagonal_of is not a square matrix."};
      return detail::make_constant_column_vector<Arg>(seq, constant_diagonal_coefficient {std::forward<Arg>(arg)}, *d);
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
