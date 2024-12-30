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
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/dynamic_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/equivalent_to.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"


namespace OpenKalman::descriptor::internal
{
#ifdef __cpp_concepts
    template<typename C>
#else
    template<typename C, typename = void>
#endif
    struct uniform_static_vector_space_descriptor_query : std::false_type {};


#ifdef __cpp_concepts
    template<atomic_static_vector_space_descriptor C> requires
      (dimension_size_of_v<C> == 1) and (not euclidean_vector_space_descriptor<C>)
    struct uniform_static_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_static_vector_space_descriptor_query<C, std::enable_if_t<atomic_static_vector_space_descriptor<C> and
      (dimension_size_of_v<C> == 1) and (not euclidean_vector_space_descriptor<C>)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<euclidean_vector_space_descriptor C> requires
      (dynamic_vector_space_descriptor<C> or atomic_static_vector_space_descriptor<C>) and (dimension_size_of_v<C> != 0)
    struct uniform_static_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_static_vector_space_descriptor_query<C, std::enable_if_t<euclidean_vector_space_descriptor<C> and
      (dynamic_vector_space_descriptor<C> or atomic_static_vector_space_descriptor<C>) and (dimension_size_of_v<C> != 0)>>
#endif
      : std::true_type { using uniform_type = descriptor::Dimensions<1>; };


#ifdef __cpp_concepts
    template<typename C> requires (dimension_size_of_v<C> == 1)
    struct uniform_static_vector_space_descriptor_query<descriptor::StaticDescriptor<C>>
#else
    template<typename C>
    struct uniform_static_vector_space_descriptor_query<descriptor::StaticDescriptor<C>, std::enable_if_t<dimension_size_of_v<C> == 1>>
#endif
      : uniform_static_vector_space_descriptor_query<C> {};


#ifdef __cpp_concepts
    template<atomic_static_vector_space_descriptor C, static_vector_space_descriptor...Cs> requires (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and equivalent_to<C, typename uniform_static_vector_space_descriptor_query<descriptor::StaticDescriptor<Cs...>>::uniform_type>
    struct uniform_static_vector_space_descriptor_query<descriptor::StaticDescriptor<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_static_vector_space_descriptor_query<descriptor::StaticDescriptor<C, Cs...>, std::enable_if_t<
      atomic_static_vector_space_descriptor<C> and (... and static_vector_space_descriptor<Cs>) and (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and equivalent_to<C, typename uniform_static_vector_space_descriptor_query<descriptor::StaticDescriptor<Cs...>>::uniform_type>>>
#endif
      : std::true_type { using uniform_type = C; };

} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_UNIFORM_FIXED_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP
