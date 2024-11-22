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
 * \brief Definition for \ref static_reverse.
 */

#ifndef OPENKALMAN_DESCRIPTORS_STATIC_REVERSE_HPP
#define OPENKALMAN_DESCRIPTORS_STATIC_REVERSE_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "static_concatenate.hpp"


namespace OpenKalman::descriptor
{
  /**
   * \brief Reverse the order of a \ref vector_space_descriptor.
   */
#ifdef __cpp_concepts
  template<static_vector_space_descriptor T>
#else
  template<typename T>
#endif
  struct static_reverse
  {
    using type = T;
  };


  /**
   * \brief Helper template for \ref static_reverse.
   */
  template<typename T>
  using static_reverse_t = typename static_reverse<T>::type;


  template<typename C, typename...Cs>
  struct static_reverse<StaticDescriptor<C, Cs...>>
  {
    using type = static_concatenate_t<static_reverse_t<StaticDescriptor<Cs...>>, StaticDescriptor<C>>;
  };


} // namespace OpenKalman::descriptor

#endif //OPENKALMAN_DESCRIPTORS_STATIC_REVERSE_HPP
