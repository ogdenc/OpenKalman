/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref uniform_static_vector_space_descriptor_query.
 */

#ifndef OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP
#define OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP

#include <type_traits>
#include "linear-algebra/coordinates/concepts/fixed_pattern.hpp"
#include "linear-algebra/coordinates/concepts/dynamic_pattern.hpp"
#include "linear-algebra/coordinates/concepts/euclidean_pattern.hpp"
#include "linear-algebra/coordinates/concepts/descriptor.hpp"
#include "linear-algebra/coordinates/concepts/compares_with.hpp"
#include "linear-algebra/coordinates/traits/dimension_of.hpp"


namespace OpenKalman::coordinates::internal
{
#ifdef __cpp_concepts
    template<typename C>
#else
    template<typename C, typename = void>
#endif
    struct uniform_static_vector_space_descriptor_query : std::false_type {};


#ifdef __cpp_concepts
    template<descriptor C> requires
      (dimension_of_v<C> == 1) and (not euclidean_pattern<C>)
    struct uniform_static_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_static_vector_space_descriptor_query<C, std::enable_if_t<descriptor<C> and
      (dimension_of_v<C> == 1) and (not euclidean_pattern<C>)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<euclidean_pattern C> requires
      (dynamic_pattern<C> or descriptor<C>) and (dimension_of_v<C> != 0)
    struct uniform_static_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_static_vector_space_descriptor_query<C, std::enable_if_t<euclidean_pattern<C> and
      (dynamic_pattern<C> or descriptor<C>) and (dimension_of_v<C> != 0)>>
#endif
      : std::true_type { using uniform_type = coordinates::Dimensions<1>; };


#ifdef __cpp_concepts
    template<typename C> requires (dimension_of_v<C> == 1)
    struct uniform_static_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_static_vector_space_descriptor_query<C, std::enable_if_t<dimension_of_v<C> == 1>>
#endif
      : uniform_static_vector_space_descriptor_query<C> {};


#ifdef __cpp_concepts
    template<descriptor C, fixed_pattern...Cs> requires (dimension_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and compares_with<C, typename uniform_static_vector_space_descriptor_query<std::tuple<Cs...>>::uniform_type>
    struct uniform_static_vector_space_descriptor_query<std::tuple<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_static_vector_space_descriptor_query<std::tuple<C, Cs...>, std::enable_if_t<
      descriptor<C> and (... and fixed_pattern<Cs>) and (dimension_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and compares_with<C, typename uniform_static_vector_space_descriptor_query<std::tuple<Cs...>>::uniform_type>>>
#endif
      : std::true_type { using uniform_type = C; };

} // namespace OpenKalman::coordinates::internal

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP
