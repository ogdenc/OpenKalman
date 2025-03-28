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
    template<typename T, std::size_t N, std::size_t...Ix>
    constexpr bool compatible_extension(std::index_sequence<Ix...>)
    {
      return (... and (compares_with<vector_space_descriptor_of_t<T, N + Ix>, coordinate::Dimensions<1>, equal_to<>, Applicability::permitted>));
    }


    template<typename T, typename D, std::size_t...Ix>
    constexpr bool is_compatible_descriptor_tuple(std::index_sequence<Ix...>)
    {
      constexpr std::size_t N = sizeof...(Ix);
      constexpr bool Dsmatch = (... and (compares_with<vector_space_descriptor_of_t<T, Ix>, std::tuple_element_t<Ix, D>, equal_to<>, Applicability::permitted>));

      if constexpr (index_count_v<T> != dynamic_size and N < index_count_v<T>)
        return Dsmatch and compatible_extension<T, N>(std::make_index_sequence<index_count_v<T> - N>{});
      else
        return Dsmatch;
    }
    
    
#ifdef __cpp_concepts
    template<typename T, typename D>
#else
    template<typename T, typename D, typename = void>
#endif
    struct compatible_with_vector_space_descriptor_collection_impl : std::true_type {};
 
 
#ifdef __cpp_concepts
    template<typename T, pattern_tuple D>
    struct compatible_with_vector_space_descriptor_collection_impl<T, D> 
#else
    template<typename T, typename D>
    struct compatible_with_vector_space_descriptor_collection_impl<T, D, std::enable_if_t<
      pattern_tuple<D>>>
#endif
      : std::bool_constant<is_compatible_descriptor_tuple<T, D>(std::make_index_sequence<std::tuple_size_v<D>>{})> {}; 

  } // namespace detail


  /**
   * \brief \ref indexible T is compatible with \ref pattern_collection D.
   * \tparam T An \ref indexible object
   * \tparam D A \ref pattern_collection
   */
  template<typename T, typename D>
#ifdef __cpp_concepts
  concept compatible_with_vector_space_descriptor_collection =
#else
  constexpr bool compatible_with_vector_space_descriptor_collection =
#endif
    indexible<T> and pattern_collection<D> and
      detail::compatible_with_vector_space_descriptor_collection_impl<T, D>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTOR_COLLECTION_HPP
