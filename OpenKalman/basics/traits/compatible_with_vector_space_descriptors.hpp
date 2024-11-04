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
 * \brief Definition for \ref compatible_with_vector_space_descriptors.
 */

#ifndef OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTORS_HPP
#define OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTORS_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t N, std::size_t...IxT>
    constexpr bool compatible_ext(std::index_sequence<IxT...>)
    {
      return (... and (maybe_equivalent_to<vector_space_descriptor_of_t<T, N + IxT>, Dimensions<1>>));
    }


    template<typename T, typename...Ds, std::size_t...IxD>
    constexpr bool compatible_impl(std::index_sequence<IxD...>)
    {
      constexpr std::size_t N = sizeof...(Ds);
      constexpr bool Dsmatch = (... and (maybe_equivalent_to<vector_space_descriptor_of_t<T, IxD>, Ds>));

      if constexpr (index_count_v<T> != dynamic_size and N < index_count_v<T>)
        return Dsmatch and compatible_ext<T, N>(std::make_index_sequence<index_count_v<T> - N>{});
      else
        return Dsmatch;
    }
  } // namespace detail


  /**
   * \brief \ref indexible T is compatible with \ref vector_space_descriptor set Ds.
   * \details If T has a fixed number of indices, then any trailing indices beyond the set of Ds must be compatible with Dimensions<1>.
   */
  template<typename T, typename...Ds>
#ifdef __cpp_concepts
  concept compatible_with_vector_space_descriptors =
#else
  constexpr bool compatible_with_vector_space_descriptors =
#endif
    indexible<T> and (vector_space_descriptor<Ds> and ...) and
      detail::compatible_impl<T, Ds...>(std::index_sequence_for<Ds...>{});


} // namespace OpenKalman

#endif //OPENKALMAN_COMPATIBLE_WITH_VECTOR_SPACE_DESCRIPTORS_HPP
