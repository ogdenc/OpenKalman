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
 * \brief Arithmetic operators for \ref coordinates::pattern objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/DynamicDescriptor.hpp"
#include "linear-algebra/coordinates/descriptors/internal/Concatenate.hpp"
#include "linear-algebra/coordinates/descriptors/internal/Replicate.hpp"

namespace OpenKalman::coordinates
{
  /**
   * \brief Add two sets of \ref coordinates::pattern objects, whether fixed or dynamic.
   */
#ifdef __cpp_concepts
  template<pattern T, pattern U>
#else
  template<typename T, typename U, std::enable_if_t<pattern<T> and pattern<U>, int> = 0>
#endif
  constexpr auto operator+(T&& t, U&& u)
  {
    if constexpr (euclidean_pattern<T> and euclidean_pattern<U>)
    {
      if constexpr (fixed_pattern<T> and fixed_pattern<U>)
        return Dimensions<dimension_of_v<T> + dimension_of_v<U>>{};
      else
        return Dimensions {get_dimension(t) + get_dimension(u)};
    }
    else return coordinates::internal::Concatenate {std::forward<T>(t), std::forward<U>(u)};
  }


  /**
   * \brief Replicate a \ref coordinates::pattern some number of times.
   */
#ifdef __cpp_concepts
  template<pattern Arg, values::index N>
#else
  template<typename Arg, typename N, std::enable_if_t<pattern<Arg> and values::index<N>, int> = 0>
#endif
  constexpr auto operator*(Arg&& arg, const N& n)
  {
    if constexpr (euclidean_pattern<Arg>)
    {
      return coordinates::Dimensions {values::operation {std::multiplies{}, get_dimension(arg), n}};
    }
    else
    {
      return internal::Replicate {std::forward<Arg>(arg), n};
    }
  }


  /**
   * \overload
   * \brief Replicate a \ref coordinates::pattern some number of times.
   */
#ifdef __cpp_concepts
  template<values::index N, pattern Arg>
#else
  template<typename N, typename Arg, std::enable_if_t<pattern<Arg> and values::index<N>, int> = 0>
#endif
  constexpr auto operator*(const N& n, Arg&& arg)
  {
    return operator*(std::forward<Arg>(arg), n);
  }


} // namespace OpenKalman::coordinates


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP