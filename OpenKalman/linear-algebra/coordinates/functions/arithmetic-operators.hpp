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
 * \brief Arithmetic operators for \ref coordinate::pattern objects.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP

#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/traits/size_of.hpp"
#include "linear-algebra/coordinates/descriptors/Dimensions.hpp"
#include "linear-algebra/coordinates/descriptors/DynamicDescriptor.hpp"
#include "linear-algebra/coordinates/descriptors/internal/Concatenate.hpp"
#include "linear-algebra/coordinates/descriptors/internal/Replicate.hpp"

namespace OpenKalman::coordinate
{
  /**
   * \brief Add two sets of \ref coordinate::pattern objects, whether fixed or dynamic.
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
        return Dimensions<size_of_v<T> + size_of_v<U>>{};
      else
        return Dimensions {get_size(t) + get_size(u)};
    }
    else return coordinate::internal::Concatenate {std::forward<T>(t), std::forward<U>(u)};
  }


  /**
   * \brief Replicate a \ref coordinate::pattern some number of times.
   */
#ifdef __cpp_concepts
  template<pattern Arg, value::index N>
#else
  template<typename Arg, typename N, std::enable_if_t<pattern<Arg> and value::index<N>, int> = 0>
#endif
  constexpr auto operator*(Arg&& arg, const N& n)
  {
    if constexpr (euclidean_pattern<Arg>)
    {
      return coordinate::Dimensions {value::operation {std::multiplies{}, get_size(arg), n}};
    }
    else
    {
      return internal::Replicate {std::forward<Arg>(arg), n};
    }
  }


  /**
   * \overload
   * \brief Replicate a \ref coordinate::pattern some number of times.
   */
#ifdef __cpp_concepts
  template<value::index N, pattern Arg>
#else
  template<typename N, typename Arg, std::enable_if_t<pattern<Arg> and value::index<N>, int> = 0>
#endif
  constexpr auto operator*(const N& n, Arg&& arg)
  {
    return operator*(std::forward<Arg>(arg), n);
  }


} // namespace OpenKalman::coordinate


#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_ARITHMETIC_OPERATORS_HPP