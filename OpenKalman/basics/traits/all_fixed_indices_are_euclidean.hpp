/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref all_fixed_indices_are_euclidean.
 */

#ifndef OPENKALMAN_ALL_FIXED_INDICES_ARE_EUCLIDEAN_HPP
#define OPENKALMAN_ALL_FIXED_INDICES_ARE_EUCLIDEAN_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool all_fixed_indices_are_euclidean_impl(std::index_sequence<I...>) {
      return ((dynamic_dimension<T, I> or has_untyped_index<T, I>) and ...); }
  }


  /**
   * \brief Specifies that every fixed-size index of T is euclidean.
   * \details No fixed_size index of T is modular (e.g., Angle, Polar, Spherical, etc.).
   */
#ifdef __cpp_concepts
  template<typename T>
  concept all_fixed_indices_are_euclidean =
#else
  template<typename T>
  constexpr bool all_fixed_indices_are_euclidean =
#endif
    indexible<T> and (detail::all_fixed_indices_are_euclidean_impl<T>(std::make_index_sequence<index_count_v<T>> {}));


} // namespace OpenKalman

#endif //OPENKALMAN_ALL_FIXED_INDICES_ARE_EUCLIDEAN_HPP
