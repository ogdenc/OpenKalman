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
 * \brief Definition for \ref compatible_with_vector_space_descriptor_collection.
 */

#ifndef OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
#define OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP


namespace OpenKalman
{
  namespace detail
  {

    template<typename T, typename D, std::size_t...Ix>
    constexpr bool is_compatible_descriptor_tuple(std::index_sequence<Ix...>)
    {
      return compatible_with_vector_space_descriptors<T, std::tuple_element_t<Ix, D>...>; 
    }
    
    
#ifdef __cpp_concepts
    template<typename T, typename D>
#else
    template<typename T, typename D, typename = void>
#endif
    struct compatible_with_vector_space_descriptor_collection_impl : std::true_type {};
 
 
#ifdef __cpp_concepts
    template<typename T, vector_space_descriptor_tuple D>
    struct compatible_with_vector_space_descriptor_collection_impl<T, D> 
#else
    template<typename T, typename D>
    struct compatible_with_vector_space_descriptor_collection_impl<T, D, std::enable_if_t<
      vector_space_descriptor_tuple<D>>> 
#endif
      : std::bool_constant<is_compatible_descriptor_tuple<T, D>(std::make_index_sequence<std::tuple_size_v<D>>{})> {}; 

  } // namespace detail


  /**
   * \brief \ref indexible T is compatible with \ref vector_space_descriptor_collection D.
   * \tparam T An \ref indexible object
   * \tparam D A \ref vector_space_descriptor_collection
   * \sa compatible_with_vector_space_descriptors.
   */
  template<typename T, typename D>
#ifdef __cpp_concepts
  concept compatible_with_vector_space_descriptor_collection =
#else
  constexpr bool compatible_with_vector_space_descriptor_collection =
#endif
    indexible<T> and vector_space_descriptor_collection<D> and
      detail::compatible_with_vector_space_descriptor_collection_impl<T, D>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
