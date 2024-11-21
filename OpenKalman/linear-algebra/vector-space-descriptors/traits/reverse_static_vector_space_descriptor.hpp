/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref reverse_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_REVERSE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_REVERSE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"


namespace OpenKalman
{
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
#else
  template<typename T>
#endif
  struct reverse_static_vector_space_descriptor
  {
#ifndef __cpp_concepts
    static_assert(static_vector_space_descriptor<T>);
#endif
    using type = T;
  };


  template<>
  struct reverse_static_vector_space_descriptor<descriptors::StaticDescriptor<>>
  {
    using type = descriptors::StaticDescriptor<>;
  };


  template<typename C, typename...Cs>
  struct reverse_static_vector_space_descriptor<descriptors::StaticDescriptor<C, Cs...>>
  {
    using type = concatenate_static_vector_space_descriptor_t<typename reverse_static_vector_space_descriptor<
      descriptors::StaticDescriptor<Cs...>>::type, descriptors::StaticDescriptor<C>>;
  };


} // namespace OpenKalman

#endif //OPENKALMAN_REVERSE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
