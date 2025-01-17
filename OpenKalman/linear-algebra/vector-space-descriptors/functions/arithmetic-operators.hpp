/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and b recursive filters.
 *
 * Copyright (c) 2020-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Arithmetic operators for \ref vector_space_descriptor objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/DynamicDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/Concatenate.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/internal/Replicate.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief Add two sets of \ref vector_space_descriptor objects, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor T, vector_space_descriptor U>
#else
  template<typename T, typename U, std::enable_if_t<vector_space_descriptor<T> and vector_space_descriptor<U>, int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U>)
    {
      if constexpr (static_vector_space_descriptor<T> and static_vector_space_descriptor<U>)
        return Dimensions<dimension_size_of_v<T> + dimension_size_of_v<U>>{};
      else
        return Dimensions {get_dimension_size_of(t) + get_dimension_size_of(u)};
    }
    else return descriptor::internal::Concatenate {std::forward<T>(t), std::forward<U>(u)};
  }


  /**
   * \brief Replicate a \ref vector_space_descriptor some number of times.
   */
#ifdef __cpp_concepts
  template<vector_space_descriptor Arg, value::index N>
#else
  template<typename Arg, typename N, std::enable_if_t<vector_space_descriptor<Arg> and value::index<N>, int> = 0>
#endif
  constexpr auto operator*(Arg&& arg, const N& n)
  {
    if constexpr (euclidean_vector_space_descriptor<Arg>)
    {
      return descriptor::Dimensions {value::operation {std::multiplies<>{}, get_dimension_size_of(arg), n}};
    }
    else
    {
      return internal::Replicate {std::forward<Arg>(arg), n};
    }
  }


  /**
   * \overload
   * \brief Replicate a \ref vector_space_descriptor some number of times.
   */
#ifdef __cpp_concepts
  template<value::index N, vector_space_descriptor Arg>
#else
  template<typename N, typename Arg, std::enable_if_t<vector_space_descriptor<Arg> and value::index<N>, int> = 0>
#endif
  constexpr auto operator*(const N& n, Arg&& arg)
  {
    return operator*(std::forward<Arg>(arg), n);
  }


} // namespace OpenKalman::descriptor


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP