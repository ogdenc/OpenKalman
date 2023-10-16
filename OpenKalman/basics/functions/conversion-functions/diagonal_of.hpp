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
      return make_constant_matrix_like<T>(std::forward<C>(c), std::forward<D0>(d0), Dimensions<Is==Is?1:1>{}...);
    }
  }


  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A square matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg>
  constexpr vector<0, Likelihood::maybe> decltype(auto)
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe>, int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    constexpr std::make_index_sequence<index_count_v<Arg> - 1> seq;

    if constexpr (diagonal_adapter<Arg>)
    {
      return nested_matrix(std::forward<Arg>(arg));
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    // arg is a one-by-one matrix with at least one dynamic dimension, or an empty matrix
    else if constexpr (max_tensor_order_of_v<Arg> < index_count_v<Arg>)
    {
      if (not get_is_square(arg)) throw std::invalid_argument{"Argument of diagonal_of is not a square matrix."};
      if constexpr (one_by_one_matrix<Arg, Likelihood::maybe>)
      {
        constant_coefficient c {std::forward<Arg>(arg)};
        return detail::make_constant_column_vector<Arg>(seq, c, Dimensions<1>{});
      }
      else
      {
        internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0> z;
        return detail::make_constant_column_vector<Arg>(seq, z, Dimensions<0>{});
      }
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
      return interface::library_interface<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
