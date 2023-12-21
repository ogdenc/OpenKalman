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
 * \brief the contract_in_place function.
 */

#ifndef OPENKALMAN_CONTRACT_IN_PLACE_HPP
#define OPENKALMAN_CONTRACT_IN_PLACE_HPP


namespace OpenKalman
{
  /**
   * \brief In-place matrix multiplication of A * B, storing the result in A
   * \tparam on_the_right Whether the application is on the right (true) or on the left (false)
   * \result Either either A * B (if on_the_right == true) or B * A (if on_the_right == false)
   */
#ifdef __cpp_concepts
  template<bool on_the_right = true, square_shaped<Likelihood::maybe> A, square_shaped<Likelihood::maybe> B> requires
    maybe_same_shape_as<A, B> and (writable<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>) and
    (index_count_v<A> == dynamic_size or index_count_v<A> <= 2) and (index_count_v<B> == dynamic_size or index_count_v<B> <= 2)
#else
  template<bool on_the_right = true, typename A, typename B, std::enable_if_t<
    square_shaped<A, Likelihood::maybe> and square_shaped<B, Likelihood::maybe> and maybe_same_shape_as<A, B> and
    (writable<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>) and
    (index_count<A>::value == dynamic_size or index_count<A>::value <= 2) and (index_count<B>::value == dynamic_size or index_count<B>::value <= 2), int> = 0>
#endif
  constexpr A&
  contract_in_place(A& a, B&& b)
  {
    if constexpr (not square_shaped<A> or not square_shaped<B> or not same_shape_as<A, B>) if (not same_shape(a, b))
      throw std::invalid_argument {"Arguments to contract_in_place must match in size and be square matrices"};

    if constexpr (zero<A> or identity_matrix<B>)
    {
      return a;
    }
    else if constexpr (zero<B>)
    {
      return a = std::forward<B>(b);
    }
    else if constexpr (diagonal_adapter<A> and diagonal_matrix<B>)
    {
      internal::set_triangle<TriangleType::diagonal>(a, n_ary_operation(std::multiplies<>{}, diagonal_of(a), diagonal_of(std::forward<B>(b))));
      return a;
    }
    else if constexpr (triangular_adapter<A> and triangle_type_of_v<A> == triangle_type_of_v<A, B>)
    {
      internal::set_triangle<triangle_type_of_v<A>>(a, contract(a, std::forward<B>(b)));
      return a;
    }
    else if constexpr (interface::contract_in_place_defined_for<std::decay_t<A>, on_the_right, A&, B&&>)
    {
      return interface::library_interface<std::decay_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b));
    }
    else
    {
      a = contract(a, std::forward<B>(b));
    }
  }


} // namespace OpenKalman


#endif //OPENKALMAN_CONTRACT_IN_PLACE_HPP
