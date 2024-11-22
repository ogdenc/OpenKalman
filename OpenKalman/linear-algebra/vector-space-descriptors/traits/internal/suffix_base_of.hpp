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
 * \internal
 * \brief Definition for \ref descriptor::internal::suffix_base_of.
 */

#ifndef OPENKALMAN_DESCRIPTORS_SUFFIX_BASE_OF_HPP
#define OPENKALMAN_DESCRIPTORS_SUFFIX_BASE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "prefix_base_of.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/internal/suffix_of.hpp"


namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief If T is a \ref internal::suffix_of "suffix of" U, return the non-overlapping base part.
   * \details If T is a suffix of U, <code>type</code> will be an alias for the base.
   */
#ifdef __cpp_concepts
  template<typename T, typename U>
#else
  template<typename T, typename U, typename = void>
#endif
  struct suffix_base_of {};


#ifdef __cpp_concepts
  template<static_vector_space_descriptor T, static_vector_space_descriptor U> requires suffix_of<T, U>
  struct suffix_base_of<T, U>
#else
  template<typename T, typename U>
  struct is_prefix<T, U, std::enable_if_t<suffix_of<T, U>>>
#endif
  {
    using type = static_reverse_t<
      prefix_base_of_t<static_reverse_t<T>, static_reverse_t<U>>>;
  };


  /**
   * \brief Helper template for \ref suffix_base_of.
   */
  template<typename T, typename U>
  using suffix_base_of_t = typename suffix_base_of<T, U>::type;


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_DESCRIPTORS_SUFFIX_BASE_OF_HPP
