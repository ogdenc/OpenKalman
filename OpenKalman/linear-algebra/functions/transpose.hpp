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
 * \brief Definition of transpose function.
 */

#ifndef OPENKALMAN_TRANSPOSE_HPP
#define OPENKALMAN_TRANSPOSE_HPP

#include<complex>


namespace OpenKalman
{
  namespace internal
  {
    template<typename C, typename Arg, std::size_t...Is>
    constexpr decltype(auto) transpose_constant(C&& c, Arg&& arg, std::index_sequence<Is...>)
    {
      return make_constant<Arg>(std::forward<C>(c),
        get_vector_space_descriptor<1>(arg), get_vector_space_descriptor<0>(arg), get_vector_space_descriptor<Is + 2>(arg)...);
    }
  } // namespace internal


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Arg, typename = void>
    struct constant_transpose_defined_for: std::false_type {};

    template<typename T, typename Arg>
    struct constant_transpose_defined_for<T, Arg, std::enable_if_t<
      constant_matrix<decltype(interface::library_interface<T>::transpose(std::declval<Arg>()))>>>
      : std::true_type {};
  } // namespace detail
#endif


  /**
   * \brief Take the transpose of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (max_tensor_order_v<Arg> <= 2)
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and (max_tensor_order_v<Arg> <= 2), int> = 0>
#endif
  constexpr decltype(auto) transpose(Arg&& arg)
  {
    if constexpr (((diagonal_matrix<Arg> or constant_matrix<Arg>) and square_shaped<Arg>) or
      (hermitian_matrix<Arg> and not value::complex<scalar_type_of_t<Arg>>))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (hermitian_matrix<Arg>)
    {
      return conjugate(std::forward<Arg>(arg));
    }
#ifdef __cpp_concepts
    else if constexpr (constant_matrix<Arg> and
      not requires { {interface::library_interface<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg))} -> constant_matrix; })
#else
    else if constexpr (constant_matrix<Arg> and not detail::constant_transpose_defined_for<Arg, Arg>::value)
#endif
    {
      constexpr std::make_index_sequence<std::max({index_count_v<Arg>, 2_uz}) - 2_uz> seq;
      return internal::transpose_constant(constant_coefficient{arg}, std::forward<Arg>(arg), seq);
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TRANSPOSE_HPP
