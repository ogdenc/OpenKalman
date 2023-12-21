/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref is_one_dimensional function.
 */

#ifndef OPENKALMAN_IS_ONE_DIMENSIONAL_HPP
#define OPENKALMAN_IS_ONE_DIMENSIONAL_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t I, std::size_t...Is, typename T>
    constexpr bool is_one_dimensional_impl(std::index_sequence<I, Is...>, const T& t)
    {
      return get_index_dimension_of<I>(t) == 1_uz and
        (... and (get_vector_space_descriptor<I>(t) == get_vector_space_descriptor<Is>(t)));
    }
  }


  /**
   * \brief Return true if T is a \ref one_dimensional at runtime.
   * \details Each index also must have an equivalent \ref vector_space_descriptor object.
   * \tparam T A tensor or matrix
   */
#ifdef __cpp_concepts
  template<interface::count_indices_defined_for T>
#else
  template<typename T, std::enable_if_t<interface::count_indices_defined_for<T>, int> = 0>
#endif
  constexpr bool is_one_dimensional(const T& t)
  {
    if constexpr (static_index_value<decltype(count_indices(t))>)
    {
      constexpr std::size_t count = std::decay_t<decltype(count_indices(t))>::value;
      if constexpr (count == 0) return true;
      else return detail::is_one_dimensional_impl(std::make_index_sequence<count>{}, t);
    }
    else
    {
      auto d0 = get_vector_space_descriptor<0>(t);
      if (get_dimension_size_of(d0) != 1_uz) return false;
      else for (std::size_t i = 1; i < count_indices(t); ++i) if (d0 != get_vector_space_descriptor(t, i)) return false;
      return true;
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_INDEXIBLE_PROPERTY_FUNCTIONS_HPP
