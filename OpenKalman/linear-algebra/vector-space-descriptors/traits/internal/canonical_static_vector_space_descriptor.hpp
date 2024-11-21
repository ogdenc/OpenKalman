/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref canonical_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_CANONICAL_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_CANONICAL_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/euclidean_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/atomic_static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/traits/dimension_size_of.hpp"
#include "linear-algebra/vector-space-descriptors/traits/replicate_static_vector_space_descriptor.hpp"


namespace OpenKalman::descriptors::internal
{
#ifdef __cpp_concepts
  template<atomic_static_vector_space_descriptor C>
  struct canonical_static_vector_space_descriptor<C>
#else
  template<typename C>
  struct canonical_static_vector_space_descriptor<C, std::enable_if_t<atomic_static_vector_space_descriptor<C>>>
#endif
  {
    using type = std::conditional_t<
      euclidean_vector_space_descriptor<C>,
      replicate_static_vector_space_descriptor_t<descriptors::Dimensions<1>, dimension_size_of_v<C>>,
      descriptors::StaticDescriptor<C>>;
  };


  template<typename...Cs>
  struct canonical_static_vector_space_descriptor<descriptors::StaticDescriptor<descriptors::StaticDescriptor<Cs...>>>
  {
    using type = typename canonical_static_vector_space_descriptor<descriptors::StaticDescriptor<Cs...>>::type;
  };


  template<typename...Cs>
  struct canonical_static_vector_space_descriptor<descriptors::StaticDescriptor<Cs...>>
  {
    using type = concatenate_static_vector_space_descriptor_t<typename canonical_static_vector_space_descriptor<Cs>::type...>;
  };


} // namespace OpenKalman::descriptors::internal

#endif //OPENKALMAN_CANONICAL_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
