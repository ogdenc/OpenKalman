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
 * \brief Definition for \ref vector_space_descriptor_collection_common_type.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_COMMON_TYPE_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_COMMON_TYPE_HPP

#ifdef __cpp_lib_ranges
#include <ranges>
#endif
#include <type_traits>
#include <tuple>
#include <utility>
#include "linear-algebra/values/concepts/number.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_tuple.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/vector_space_descriptor_collection.hpp"
#include "linear-algebra/vector-space-descriptors/concepts/static_vector_space_descriptor_tuple.hpp"
#include "linear-algebra/vector-space-descriptors/internal/forward-declarations.hpp"
#include "linear-algebra/vector-space-descriptors/functions/internal/canonical_equivalent.hpp"


namespace OpenKalman::descriptor::internal
{
  namespace detail
  {
    template<typename T, std::size_t...Ix>
    constexpr bool same_descriptors_in_tuple(std::index_sequence<Ix...>)
    {
      return equivalent_to<std::tuple_element_t<Ix, T>...>;
    }
  } // namespace detail

  /**
   * \brief Indices is a std::ranges::sized_range of indices that are compatible with \ref indexible object T.
   * \tparam Scalar The scalar type associated with the \ref vector_space_descriptor if it is DynamicDescriptor
   */
#ifdef __cpp_concepts
  template<typename T, typename Scalar>
#else
  template<typename T, typename Scalar, typename = void>
#endif
  struct vector_space_descriptor_collection_common_type;


#ifdef __cpp_concepts
  template<static_vector_space_descriptor_tuple T, value::number Scalar> requires
    (std::tuple_size_v<T> > 0) and (detail::same_descriptors_in_tuple<T>(std::make_index_sequence<std::tuple_size_v<T>>{}))
  struct vector_space_descriptor_collection_common_type<T, Scalar>
#else
  template<typename T, typename Scalar>
  struct vector_space_descriptor_collection_common_type<T, Scalar, std::enable_if_t<
    static_vector_space_descriptor_tuple<T> and (std::tuple_size<T>::value > 0) and
    detail::same_descriptors_in_tuple(std::make_index_sequence<std::tuple_size<T>::value>{})>>
#endif
  {
    using type = std::decay_t<decltype(internal::canonical_equivalent(std::declval<std::tuple_element_t<0, T>>()))>;
  };


#ifdef __cpp_concepts
  template<vector_space_descriptor_tuple T, value::number Scalar> requires (not static_vector_space_descriptor_tuple<T>)
  struct vector_space_descriptor_collection_common_type<T, Scalar>
#else
  template<typename T, typename Scalar>
  struct vector_space_descriptor_collection_common_type<T, Scalar, std::enable_if_t<vector_space_descriptor_tuple<T> and
    (not static_vector_space_descriptor_tuple<T>)>>
#endif
  {
    using type = descriptor::DynamicDescriptor<Scalar>;
  };


#ifdef __cpp_lib_ranges
  template<vector_space_descriptor_collection T, value::number Scalar> requires (not vector_space_descriptor_tuple<T>)
  struct vector_space_descriptor_collection_common_type<T, Scalar>
  {
    using type = std::ranges::range_value_t<T>;
  };
#else
  template<typename T, typename Scalar>
  struct vector_space_descriptor_collection_common_type<T, Scalar, std::enable_if_t<
   vector_space_descriptor_collection<T> and (not vector_space_descriptor_tuple<T>)>>
  {
    using std::begin;
    using type = std::decay_t<decltype(*begin(std::declval<T>()))>;
  };
#endif


  /**
   * \brief Helper for \ref vector_space_descriptor_collection_common_type.
   */
  template<typename T, typename Scalar>
  constexpr std::size_t vector_space_descriptor_collection_common_type_t =
    vector_space_descriptor_collection_common_type<T, Scalar>::type;


} // namespace OpenKalman::descriptor::internal

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTOR_COLLECTION_COMMON_TYPE_HPP
