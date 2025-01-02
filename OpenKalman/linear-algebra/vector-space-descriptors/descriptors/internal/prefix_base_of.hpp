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
 * \brief Definition for \ref descriptor::internal::prefix_base_of.
 */

#ifndef OPENKALMAN_DESCRIPTORS_PREFIX_BASE_OF_HPP
#define OPENKALMAN_DESCRIPTORS_PREFIX_BASE_OF_HPP

#include <type_traits>
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"

namespace OpenKalman::descriptor::internal
{
  /**
   * \internal
   * \brief If T is a prefix of U, return the non-overlapping base part.
   * \details Assumes that the inputs are in their \ref internal::canonical_equivalent "canonical equivalent" form.
   */
#ifdef __cpp_concepts
  template<typename T, typename U>
#else
  template<typename T, typename U, typename = void>
#endif
  struct prefix_base_of {};


#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T,  typename U> requires (dimension_size_of_v<T> == 0)
  struct prefix_base_of<T, U>
#else
  template<typename T, typename U>
  struct prefix_base_of<T, U, enable_if_t<euclidean_vector_space_descriptor<T> and (dimension_size_of_v<T> == 0)>>
#endif
  {
    using type = U;
  };


#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T,  typename U> requires (dimension_size_of_v<T> == 0)
  struct prefix_base_of<T, StaticDescriptor<U>>
#else
  template<typename T, typename U>
  struct prefix_base_of<T, StaticDescriptor<U>, enable_if_t<euclidean_vector_space_descriptor<T> and (dimension_size_of_v<T> == 0)>>
#endif
  {
    using type = U;
  };


  template<typename T, typename U>
  struct prefix_base_of<T, StaticDescriptor<T, U>>
  {
    using type = U;
  };


  template<typename T, typename U, typename...Us>
  struct prefix_base_of<T, StaticDescriptor<T, U, Us...>>
  {
    using type = StaticDescriptor<U, Us...>;
  };


#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T, euclidean_vector_space_descriptor U> requires
    (dimension_size_of_v<T> > 0) and (dimension_size_of_v<T> < dimension_size_of_v<U>)
  struct prefix_base_of<T, U>
#else
  template<typename T, typename U>
  struct prefix_base_of<T, U, std::enable_if_t<euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U> and
    (dimension_size_of_v<T> > 0) and (dimension_size_of_v<T> < dimension_size_of_v<U>)>>
#endif
  {
    using type = Dimensions<dimension_size_of_v<U> - dimension_size_of_v<T>>;
  };


#ifdef __cpp_concepts
  template<euclidean_vector_space_descriptor T, euclidean_vector_space_descriptor U, typename...Us> requires
    (dimension_size_of_v<T> > 0) and (dimension_size_of_v<T> < dimension_size_of_v<U>) and (sizeof...(Us) > 0)
  struct prefix_base_of<T, StaticDescriptor<U, Us...>>
#else
  template<typename T, typename U>
  struct prefix_base_of<T, StaticDescriptor<U, Us...>, std::enable_if_t<
    euclidean_vector_space_descriptor<T> and euclidean_vector_space_descriptor<U> and
    (dimension_size_of_v<T> > 0) and (dimension_size_of_v<T> < dimension_size_of_v<U>) and (sizeof...(Us) > 0)>>
#endif
  {
    using type = StaticDescriptor<Dimensions<dimension_size_of_v<U> - dimension_size_of_v<T>>, Us...>;
  };


  template<typename T, typename...Cs, typename...Ds>
  struct prefix_base_of<StaticDescriptor<T, Cs...>, StaticDescriptor<T, Ds...>>
    : prefix_base_of<StaticDescriptor<Cs...>, StaticDescriptor<Ds...>> {};


  /**
   * \internal
   * \brief Helper template for \ref prefix_base_of.
   */
  template<typename T, typename U>
  using prefix_base_of_t = typename prefix_base_of<T, U>::type;


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_DESCRIPTORS_PREFIX_BASE_OF_HPP
