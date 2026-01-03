/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref uniform_pattern.
 */

#ifndef OPENKALMAN_UNIFORM_PATTERN_HPP
#define OPENKALMAN_UNIFORM_PATTERN_HPP

#include "collections/collections.hpp"
#include "patterns/concepts/compares_with.hpp"
#include "patterns/traits/uniform_pattern_type.hpp"

namespace OpenKalman::patterns
{
  namespace detail
  {
    template<typename T, typename C = void, std::size_t i = 0>
    constexpr auto
    heterogeneous_pattern_impl()
    {
      if constexpr (i < collections::size_of_v<T>)
      {
        using A = common_descriptor_type_t<collections::collection_element_t<i, T>>;
        if constexpr (i == 0)
        {
          return heterogeneous_pattern_impl<T, A, i + 1>();
        }
        else
        {
          constexpr auto dA = dimension_of_v<A>;
          if constexpr ((dA != stdex::dynamic_extent and dA != 1) or not compares_with<C, A, &stdex::is_eq, applicability::permitted>)
            return std::true_type {};
          else if constexpr (dA != stdex::dynamic_extent and dimension_of_v<C> == stdex::dynamic_extent)
            return heterogeneous_pattern_impl<T, A, i + 1>();
          else
            return heterogeneous_pattern_impl<T, C, i + 1>();
        }
      }
      else
      {
        return std::false_type{};
      }
    }


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct heterogeneous_pattern : std::false_type {};


#ifdef __cpp_concepts
    template<descriptor T>
    struct heterogeneous_pattern<T>
#else
    template<typename T>
    struct heterogeneous_pattern<T, std::enable_if_t<descriptor<T>>>
#endif
      : std::bool_constant<
          (not euclidean_pattern<T>) and
          (dimension_of_v<T> != stdex::dynamic_extent) and
          (dimension_of_v<T> != 1)
        > {};


#ifdef __cpp_concepts
    template<descriptor_collection T> requires
      collections::sized<T> and
      (collections::size_of_v<T> != stdex::dynamic_extent) and
      (collections::size_of_v<T> > 1) and
      collections::uniformly_gettable<T>
    struct heterogeneous_pattern<T>
#else
    template<typename T>
    struct heterogeneous_pattern<T, std::enable_if_t<
      descriptor_collection<T> and
      values::fixed_value_compares_with<collections::size_of<T>, stdex::dynamic_extent, &stdex::is_neq> and
      values::fixed_value_compares_with<collections::size_of<T>, 1, &stdex::is_gt> and
      collections::uniformly_gettable<T>
    >>
#endif
      : std::bool_constant<heterogeneous_pattern_impl<T>()> {};


#ifdef __cpp_concepts
    template<descriptor_collection T> requires
      (not collections::uniformly_gettable<T>) and
      (dimension_of_v<common_descriptor_type_t<T>> != stdex::dynamic_extent) and
      (dimension_of_v<common_descriptor_type_t<T>> != 1)
    struct heterogeneous_pattern<T>
#else
    template<typename T>
    struct heterogeneous_pattern<T, std::enable_if_t<
      descriptor_collection<T> and
      (not collections::uniformly_gettable<T>) and
      values::fixed_value_compares_with<dimension_of<common_descriptor_type_t<T>>, stdex::dynamic_extent, &stdex::is_neq> and
      values::fixed_value_compares_with<dimension_of<common_descriptor_type_t<T>>, 1, &stdex::is_neq>
    >>
#endif
      : std::true_type {};


#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_uniform_pattern_impl : std::false_type {};

    template<typename T>
    struct is_uniform_pattern_impl<T, std::void_t<typename uniform_pattern_type<T>::type>>
      : std::true_type {};
#endif

  }


  /**
   * \brief T is a \ref patterns::pattern that is either empty or can be decomposed into a uniform set of 1D \ref patterns::pattern.
   * \details If T is a uniform pattern, \ref uniform_pattern_type<T>::type will exist and will be one-dimensional.
   */
  template<typename T, applicability a = applicability::guaranteed>
#ifdef __cpp_concepts
  concept uniform_pattern =
    (a == applicability::guaranteed and requires { typename uniform_pattern_type<T>::type; }) or
    (a == applicability::permitted and not detail::heterogeneous_pattern<T>::value);
#else
  constexpr bool uniform_pattern =
    (a == applicability::guaranteed and detail::is_uniform_pattern_impl<T>::value) or
    (a == applicability::permitted and not detail::heterogeneous_pattern<T>::value);
#endif


}

#endif
