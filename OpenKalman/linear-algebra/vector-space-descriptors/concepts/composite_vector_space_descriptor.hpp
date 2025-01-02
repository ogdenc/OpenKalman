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
 * \brief Definition for \ref composite_vector_space_descriptor.
 */

#ifndef OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP

#include "vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/vector_space_component_count.hpp"

namespace OpenKalman::descriptor
{
  /**
   * \brief T is a \ref composite_vector_space_descriptor.
   * \details A composite \ref vector_space_descriptor object is a container for other \ref vector_space_descriptor, and can either be
   * StaticDescriptor or DynamicDescriptor.
   * \sa StaticDescriptor, DynamicDescriptor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept composite_vector_space_descriptor =
#else
  constexpr bool composite_vector_space_descriptor =
#endif
    vector_space_descriptor<T> and (vector_space_component_count_v<T> != 1);


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_COMPOSITE_VECTOR_SPACE_DESCRIPTOR_HPP
