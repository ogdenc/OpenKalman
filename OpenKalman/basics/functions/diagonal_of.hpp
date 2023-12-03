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
      return make_constant_matrix_like<T>(std::forward<C>(c), std::forward<D0>(d0), Dimensions<Is==Is?1:1>{}...);
    }
  }


  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A 2D square matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg> requires (index_count_v<Arg> == dynamic_size) or (index_count_v<Arg> <= 2)
  constexpr vector decltype(auto)
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and
    (index_count_v<Arg> == dynamic_size or index_count_v<Arg> <= 2), int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    if constexpr (diagonal_adapter<Arg>)
    {
      return nested_matrix(std::forward<Arg>(arg));
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    // arg is maybe a one-by-one matrix and has at least one dynamic dimension, or is an empty matrix
    else if constexpr (one_by_one_matrix<Arg, Likelihood::maybe> and
      (dynamic_index_count_v<Arg> < index_count_v<Arg> or (dynamic_dimension<Arg, 0> and index_count_v<Arg> == 1)))
    {
      if (not get_is_square(arg)) throw std::invalid_argument{"Argument of diagonal_of is not a square matrix."};
      constexpr std::make_index_sequence<index_count_v<Arg> - 1> seq;
      if constexpr (one_by_one_matrix<Arg, Likelihood::maybe>)
      {
        constant_coefficient c {std::forward<Arg>(arg)};
        return detail::make_constant_column_vector<Arg>(c, Dimensions<1>{}, seq);
      }
      else
      {
        internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0> z;
        return detail::make_constant_column_vector<Arg>(z, Dimensions<0>{}, seq);
      }
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto d = get_is_square(arg);
      if (not d) throw std::invalid_argument {"Argument of diagonal_of is not a square matrix."};
      constexpr std::make_index_sequence<index_count_v<Arg> - 1> seq;
      return detail::make_constant_column_vector<Arg>(constant_coefficient{std::forward<Arg>(arg)}, *d, seq);
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      auto d = get_is_square(arg);
      if (not d) throw std::invalid_argument {"Argument of diagonal_of is not a square matrix."};
      constexpr std::make_index_sequence<index_count_v<Arg> - 1> seq;
      return detail::make_constant_column_vector<Arg>(constant_diagonal_coefficient {std::forward<Arg>(arg)}, *d, seq);
    }
    else
    {
      using D = std::conditional_t<dynamic_dimension<Arg, 0>, vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Arg, 0>>;
      auto ret {internal::make_fixed_size_adapter<D>(interface::library_interface<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg)))};
      return ret;
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
