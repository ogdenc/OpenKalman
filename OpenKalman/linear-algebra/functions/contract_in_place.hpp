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
   * \result Either A * B (if on_the_right == true) or B * A (if on_the_right == false)
   */
#ifdef __cpp_concepts
  template<bool on_the_right = true, square_shaped<2, applicability::permitted> A, square_shaped<2, applicability::permitted> B> requires
    vector_space_descriptors_may_match_with<A, B> and (not triangular_matrix<A> or triangle_type_of_v<A> == triangle_type_of_v<A, B>) and
    (index_count_v<A> == stdex::dynamic_extent or index_count_v<A> <= 2) and (index_count_v<B> == stdex::dynamic_extent or index_count_v<B> <= 2)
#else
  template<bool on_the_right = true, typename A, typename B, std::enable_if_t<
    square_shaped<A, 2, applicability::permitted> and square_shaped<B, 2, applicability::permitted> and
    vector_space_descriptors_may_match_with<A, B> and (not triangular_matrix<A> or triangle_type_of<A>::value == triangle_type_of<A, B>::value) and
    (index_count<A>::value == stdex::dynamic_extent or index_count<A>::value <= 2) and (index_count<B>::value == stdex::dynamic_extent or index_count<B>::value <= 2), int> = 0>
#endif
  constexpr A&&
  contract_in_place(A&& a, B&& b)
  {
    if constexpr (not square_shaped<A, 2> or not square_shaped<B, 2> or not vector_space_descriptors_match_with<A, B>) if (not vector_space_descriptors_match(a, b))
      throw std::invalid_argument {"Arguments to contract_in_place must match in size and be square matrices"};

    if constexpr (zero<A> or identity_matrix<B>)
    {
      ;
    }
    else if constexpr (zero<B> and writable<A>)
    {
      copy(a, std::forward<B>(b));
    }
    else if constexpr (interface::contract_in_place_defined_for<A, on_the_right, A&, B&&>)
    {
      return interface::library_interface<stdex::remove_cvref_t<A>>::template contract_in_place<on_the_right>(a, std::forward<B>(b));
    }
    else
    {
      copy(a, to_dense_object(contract(a, std::forward<B>(b))));
    }
    return std::forward<A>(a);
  }


}


#endif
