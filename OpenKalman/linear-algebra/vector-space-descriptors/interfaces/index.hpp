/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief \ref vector_space_traits for index values.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_INTERFACES_INDEX_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_INTERFACES_INDEX_HPP

#include <type_traits>
#include "linear-algebra/values/concepts/index.hpp"
#include "vector_space_traits.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/StaticDescriptor.hpp"
#include "linear-algebra/vector-space-descriptors/descriptors/Dimensions.hpp"

namespace OpenKalman::interface
{
/**
 * \internal
 * \brief \ref vector_space_traits for \ref value::index values.
 */
#ifdef __cpp_concepts
  template<value::index T>
  struct vector_space_traits<T>
#else
  template<typename T>
  struct vector_space_traits<T, std::enable_if_t<value::index<T>>>
#endif
  {
    static constexpr auto
    size(const T& t) { return t; };


    static constexpr auto
    euclidean_size(const T& t) { return t; };


    static constexpr auto
    component_count(const T& t) { return t; }


    static constexpr auto
    is_euclidean(const T&) { return std::true_type{}; }


    static constexpr auto
    canonical_equivalent(const T& t)
    {
      if constexpr (descriptor::dimension_size_of_v<T> == 0)
        return descriptor::StaticDescriptor<>{};
      else
        return descriptor::Dimensions{t};
    };

  };


} // namespace OpenKalman::interface



#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_INTERFACES_INDEX_HPP
