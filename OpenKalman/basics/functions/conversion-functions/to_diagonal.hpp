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
 * \brief Definition for \ref to_diagonal function.
 */

#ifndef OPENKALMAN_TO_DIAGONAL_HPP
#define OPENKALMAN_TO_DIAGONAL_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg>
    constexpr void check_if_column_vector_at_runtime(const Arg& arg)
    {
      if constexpr (dynamic_dimension<Arg, 1>) if (get_index_dimension_of<1>(arg) != 1)
        throw std::domain_error {"Argument of to_diagonal must have 1 column; instead it has " +
          std::to_string(get_index_dimension_of<1>(arg))};
    };


    template<typename T, std::size_t...Is, typename C, typename D0>
    constexpr auto make_constant_square(std::index_sequence<Is...>, C&& c, D0&& d0)
    {
      return make_constant_matrix_like<T>(std::forward<C>(c), (Is>=0?d0:d0)...);
    }


#ifndef __cpp_concepts
    template<typename Arg, typename = void>
    struct to_diagonal_exists : std::false_type {};

    template<typename Arg>
    struct to_diagonal_exists<Arg, std::void_t<decltype(
      interface::Conversions<std::decay_t<Arg>>::template to_diagonal(std::declval<Arg&&>()))>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Convert a column vector into a diagonal matrix.
   * \tparam Arg A column vector matrix
   * \returns A diagonal matrix
   */
#ifdef __cpp_concepts
  template<vector<0, Likelihood::maybe> Arg>
  constexpr diagonal_matrix decltype(auto)
#else
  template<typename Arg, std::enable_if_t<vector<Arg, 0, Likelihood::maybe>, int> = 0>
  constexpr decltype(auto)
#endif
  to_diagonal(Arg&& arg)
  {
    constexpr std::make_index_sequence<max_indices_of_v<Arg>> seq;

    if constexpr (diagonal_matrix<Arg>)
    {
      detail::check_if_column_vector_at_runtime(arg);
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg>)
    {
      detail::check_if_column_vector_at_runtime(arg);
      internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0> c;
      return detail::make_constant_square<Arg>(seq, c, get_index_descriptor<0>(arg));
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 1> and element_gettable<Arg, 0>)
    {
      detail::check_if_column_vector_at_runtime(arg);
      return detail::make_constant_square<Arg>(seq, get_element(std::forward<Arg>(arg)), get_index_descriptor<0>(arg));
    }
#ifdef __cpp_concepts
    else if constexpr (requires { interface::Conversions<std::decay_t<Arg>>::template to_diagonal(std::forward<Arg>(arg)); })
#else
    else if constexpr (detail::to_diagonal_exists<Arg>::value)
#endif
    {
      return interface::Conversions<std::decay_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
    else
    {
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_DIAGONAL_HPP
