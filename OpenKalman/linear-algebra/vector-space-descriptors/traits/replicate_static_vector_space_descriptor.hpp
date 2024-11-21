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
 * \brief Definition for \ref replicate_static_vector_space_descriptor.
 */

#ifndef OPENKALMAN_REPLICATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
#define OPENKALMAN_REPLICATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP

#include <type_traits>
#include "basics/global-definitions.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"


namespace OpenKalman
{
  /**
   * \brief Replicate a set of \ref vector_space_descriptor a given number of times.
   * \tparam C A \ref vector_space_descriptor object to be repeated.
   * \tparam N The number of times to repeat coefficient C.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor C, std::size_t N> requires (N != dynamic_size)
#else
  template<typename C, std::size_t N>
#endif
  struct replicate_static_vector_space_descriptor
  {
  private:

#ifndef __cpp_concepts
    static_assert(static_vector_space_descriptor<C>);
    static_assert(N != dynamic_size);
#endif

    template<typename T, std::size_t...I>
    static constexpr auto replicate_inds(std::index_sequence<I...>)
    {
      return descriptors::StaticDescriptor<std::conditional_t<(I>=0), T, T>...> {};
    };

  public:

    using type = std::conditional_t<N == 1, C, decltype(replicate_inds<std::decay_t<C>>(std::make_index_sequence<N> {}))>;
  };


  /**
   * \brief Helper template for \ref replicate_static_vector_space_descriptor.
   */
  template<typename C, std::size_t N>
  using replicate_static_vector_space_descriptor_t = typename replicate_static_vector_space_descriptor<C, N>::type;


} // namespace OpenKalman

#endif //OPENKALMAN_REPLICATE_FIXED_VECTOR_SPACE_DESCRIPTOR_HPP
