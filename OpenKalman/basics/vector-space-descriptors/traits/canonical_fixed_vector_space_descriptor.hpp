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
 * \brief Definition for \ref canonical_fixed_vector_space_descriptor.
 */

#ifndef OPENKALMAN_CANONICAL_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_CANONICAL_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>


namespace OpenKalman
{
#ifdef __cpp_concepts
  template<atomic_fixed_vector_space_descriptor C>
  struct canonical_fixed_vector_space_descriptor<C>
#else
  template<typename C>
  struct canonical_fixed_vector_space_descriptor<C, std::enable_if_t<atomic_fixed_vector_space_descriptor<C>>>
#endif
  {
    using type = std::conditional_t<
      euclidean_vector_space_descriptor<C>,
      std::conditional_t<
        dimension_size_of_v<C> == 1,
        FixedDescriptor<Dimensions<1>>,
        replicate_fixed_vector_space_descriptor_t<Dimensions<1>, dimension_size_of_v<C>>>,
      FixedDescriptor<C>>;
  };


  template<typename...Cs>
  struct canonical_fixed_vector_space_descriptor<FixedDescriptor<FixedDescriptor<Cs...>>>
  {
    using type = typename canonical_fixed_vector_space_descriptor<FixedDescriptor<Cs...>>::type;
  };


  template<typename...Cs>
  struct canonical_fixed_vector_space_descriptor<FixedDescriptor<Cs...>>
  {
    using type = concatenate_fixed_vector_space_descriptor_t<typename canonical_fixed_vector_space_descriptor<Cs>::type...>;
  };


} // namespace OpenKalman

#endif //OPENKALMAN_CANONICAL_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
