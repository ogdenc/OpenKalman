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
 * \brief Definition for \ref maybe_same_shape_as_vector_space_descriptors.
 */

#ifndef OPENKALMAN_MAYBE_SAME_SHAPE_AS_VECTOR_SPACE_DESCRIPTORS_HPP
#define OPENKALMAN_MAYBE_SAME_SHAPE_AS_VECTOR_SPACE_DESCRIPTORS_HPP


namespace OpenKalman::internal
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, typename Descriptors, std::size_t...Ix>
    constexpr bool maybe_same_shape_impl(std::index_sequence<Ix...>)
    {
      return (... and (dynamic_dimension<T, Ix> or
                        dynamic_pattern<std::tuple_element_t<Ix, Descriptors>> or
                        index_dimension_of_v<T, Ix> == coordinates::dimension_of_v<std::tuple_element_t<Ix, Descriptors>>));
    }


    template<typename T, typename Descriptors, std::size_t...Ix>
    constexpr bool maybe_same_shape_ext(std::index_sequence<Ix...>)
    {
      return (... and (index_dimension_of_v<T, std::tuple_size_v<Descriptors> + Ix> == 1));
    }
  } // namespace detail
#endif

  /**
   * \brief Specifies that it is not ruled out, at compile time, that T has dimensions corresponding to a \ref pattern_collection.
   * \details Two dimensions are considered the same if their \ref coordinates::pattern are equivalent.
   * \tparam T an \ref indexible object
   * \tparam Ds a set of vector space descriptors
   * \sa vector_space_descriptors_may_match_with
   */
  template<typename T, typename Descriptors>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept maybe_same_shape_as_vector_space_descriptors =
    indexible<T> and pattern_collection<Descriptors> and
    (not pattern_tuple<Descriptors> or (
      []<std::size_t...Ix>(std::index_sequence<Ix...>){
        return (... and (dynamic_dimension<T, Ix> or
                          dynamic_pattern<std::tuple_element_t<Ix, Descriptors>> or
                          index_dimension_of_v<T, Ix> == coordinates::dimension_of_v<std::tuple_element_t<Ix, Descriptors>>));
        }(std::make_index_sequence<std::tuple_size_v<Descriptors>>{}) and
      (index_count_v<T> == dynamic_size or index_count_v<T> <= std::tuple_size_v<Descriptors> or
        []<std::size_t...Ix>(std::index_sequence<Ix...>){
          return (... and (index_dimension_of_v<T, std::tuple_size_v<Descriptors> + Ix> == 1));
          }(std::make_index_sequence<index_count_v<T> - std::tuple_size_v<Descriptors>>{})
        )));
#else
  constexpr bool maybe_same_shape_as_vector_space_descriptors =
    indexible<T> and pattern_collection<Descriptors> and
    (not pattern_tuple<Descriptors> or
      (detail::maybe_same_shape_impl<T, Descriptors>(std::make_index_sequence<std::tuple_size_v<Descriptors>>{}) and
        (index_count_v<T> == dynamic_size or index_count_v<T> <= std::tuple_size_v<Descriptors> or
          detail::maybe_same_shape_ext<T, Descriptors>(std::make_index_sequence<index_count_v<T> - std::tuple_size_v<Descriptors>>{}))));
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAYBE_SAME_SHAPE_AS_VECTOR_SPACE_DESCRIPTORS_HPP
