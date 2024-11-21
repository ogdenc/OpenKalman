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
 * \brief Definition for \ref concatenate_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_CONCATENATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_CONCATENATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"


namespace OpenKalman
{
  template<>
  struct concatenate_static_vector_space_descriptor<>
  {
    using type = descriptors::StaticDescriptor<>;
  };


#ifdef __cpp_concepts
  template<static_vector_space_descriptor C, static_vector_space_descriptor...Cs>
#else
  template<typename C, typename...Cs>
#endif
  struct concatenate_static_vector_space_descriptor<C, Cs...>
  {
    using type = concatenate_static_vector_space_descriptor_t<Cs...>::template Prepend<C>;
  };


#ifdef __cpp_concepts
  template<static_vector_space_descriptor...Cs, static_vector_space_descriptor...Ds>
#else
  template<typename...Cs, typename...Ds>
#endif
  struct concatenate_static_vector_space_descriptor<descriptors::StaticDescriptor<Cs...>, Ds...>
  {
    using type = concatenate_static_vector_space_descriptor_t<Ds...>::template Prepend<Cs...>;
  };


} // namespace OpenKalman

#endif //OPENKALMAN_CONCATENATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
