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
  namespace detail
  {
    template<typename T, typename...Ds, std::size_t...Ix, std::size_t...IxE>
    constexpr bool maybe_same_shape_as_vector_space_descriptors_impl(std::index_sequence<Ix...>, std::index_sequence<IxE...>)
    {
      return (... and (dynamic_dimension<T, Ix> or dynamic_vector_space_descriptor<Ds> or index_dimension_of_v<T, Ix> == dimension_size_of_v<Ds>)) and
        (... and (index_dimension_of_v<T, sizeof...(Ix) + IxE> == 1));
    }
  } // namespace detail

  /**
   * \brief Specifies that it is not ruled out, at compile time, that T has the same dimensions and vector-space types as Ts.
   * \details Two dimensions are considered the same if their \ref vector_space_descriptor are \ref equivalent_to "equivalent".
   * \tparam T an \ref indexible object
   * \tparam Ds a set of vector space descriptors
   * \sa vector_space_descriptors_may_match_with
   */
  template<typename T, typename...Ds>
#ifdef __cpp_concepts
  concept maybe_same_shape_as_vector_space_descriptors =
#else
  constexpr bool maybe_same_shape_as_vector_space_descriptors =
#endif
    indexible<T> and (... and vector_space_descriptor<Ds>) and
    detail::maybe_same_shape_as_vector_space_descriptors_impl<T, Ds...>(
      std::index_sequence_for<Ds...>{},
      std::make_index_sequence<(index_count_v<T> > sizeof...(Ds)) ? index_count_v<T> - sizeof...(Ds) : 0>{});


} // namespace OpenKalman::internal

#endif //OPENKALMAN_MAYBE_SAME_SHAPE_AS_VECTOR_SPACE_DESCRIPTORS_HPP
