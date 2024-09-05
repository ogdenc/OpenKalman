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
 * \brief Definition for \ref uniform_vector_space_descriptor_query.
 */

#ifndef OPENKALMAN_UNIFORM_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP
#define OPENKALMAN_UNIFORM_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP

#include <type_traits>


namespace OpenKalman::internal
{
#ifdef __cpp_concepts
    template<typename C>
#else
    template<typename C, typename = void>
#endif
    struct uniform_vector_space_descriptor_query : std::false_type {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C> requires
      (dimension_size_of_v<C> == 1) and (not euclidean_vector_space_descriptor<C>)
    struct uniform_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_vector_space_descriptor_query<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C> and
      (dimension_size_of_v<C> == 1) and (not euclidean_vector_space_descriptor<C>)>>
#endif
      : std::true_type { using uniform_type = C; };


#ifdef __cpp_concepts
    template<euclidean_vector_space_descriptor C> requires
      (dynamic_vector_space_descriptor<C> or atomic_fixed_vector_space_descriptor<C>) and (dimension_size_of_v<C> != 0)
    struct uniform_vector_space_descriptor_query<C>
#else
    template<typename C>
    struct uniform_vector_space_descriptor_query<C, std::enable_if_t<euclidean_vector_space_descriptor<C> and
      (dynamic_vector_space_descriptor<C> or atomic_fixed_vector_space_descriptor<C>) and (dimension_size_of_v<C> != 0)>>
#endif
      : std::true_type { using uniform_type = Dimensions<1>; };


#ifdef __cpp_concepts
    template<typename C> requires (dimension_size_of_v<C> == 1)
    struct uniform_vector_space_descriptor_query<FixedDescriptor<C>>
#else
    template<typename C>
    struct uniform_vector_space_descriptor_query<FixedDescriptor<C>, std::enable_if_t<dimension_size_of_v<C> == 1>>
#endif
      : uniform_vector_space_descriptor_query<C> {};


#ifdef __cpp_concepts
    template<atomic_fixed_vector_space_descriptor C, fixed_vector_space_descriptor...Cs> requires (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and equivalent_to<C, typename uniform_vector_space_descriptor_query<FixedDescriptor<Cs...>>::uniform_type>
    struct uniform_vector_space_descriptor_query<FixedDescriptor<C, Cs...>>
#else
    template<typename C, typename...Cs>
    struct uniform_vector_space_descriptor_query<FixedDescriptor<C, Cs...>, std::enable_if_t<
      atomic_fixed_vector_space_descriptor<C> and (... and fixed_vector_space_descriptor<Cs>) and (dimension_size_of_v<C> == 1) and
      (sizeof...(Cs) > 0) and equivalent_to<C, typename uniform_vector_space_descriptor_query<FixedDescriptor<Cs...>>::uniform_type>>>
#endif
      : std::true_type { using uniform_type = C; };

} // namespace OpenKalman::internal

#endif //OPENKALMAN_UNIFORM_VECTOR_SPACE_DESCRIPTOR_QUERY_HPP
