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
 * \internal
 * \brief Definition for \ref make_constant_matrix_reduction function.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_MATRIX_REDUCTION_HPP
#define OPENKALMAN_MAKE_CONSTANT_MATRIX_REDUCTION_HPP

namespace OpenKalman::internal
{

  namespace detail
  {
    template<std::size_t I, std::size_t...index, typename Arg>
    constexpr auto get_reduced_index(Arg&& arg)
    {
      if constexpr (((I == index) or ...))
      {
        using T = vector_space_descriptor_of_t<Arg, I>;
        if constexpr (has_uniform_dimension_type<T>) return uniform_dimension_type_of_t<T>{};
        else return Dimensions<1>{}; // \todo Can we extract the dynamic vector space descriptor?
      }
      else return get_vector_space_descriptor<I>(std::forward<Arg>(arg));
    }
  } // namespace detail


  template<std::size_t...index, typename C, typename T, std::size_t...I>
  constexpr auto make_constant_matrix_reduction(C&& c, T&& t, std::index_sequence<I...>)
  {
    return make_constant<T>(std::forward<C>(c), detail::get_reduced_index<I, index...>(std::forward<T>(t))...);
  }

} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAKE_CONSTANT_MATRIX_REDUCTION_HPP
