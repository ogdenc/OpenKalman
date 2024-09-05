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
 * \internal
 * \brief Definition for \ref indices_are_uniform function.
 */

#ifndef OPENKALMAN_INDICES_ARE_UNIFORM_HPP
#define OPENKALMAN_INDICES_ARE_UNIFORM_HPP

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename Arg, std::size_t...Is>
    constexpr bool indices_are_uniform_impl(std::index_sequence<Is...>)
    {
      return (... and uniform_vector_space_descriptor<vector_space_descriptor_of_t<Arg, Is>>);
    }
  } // namespace detail


  /**
   * \internal
   * \brief Whether an object's specified indices are uniform such that the object can be reduced along those indices.
   * \details If no indices are specified, it will check all indices.
   */
  template<typename Arg, std::size_t...indices>
#ifdef __cpp_concepts
  concept indices_are_uniform =
#else
  constexpr bool indices_are_uniform =
#endif
    indexible<Arg> and
    (sizeof...(indices) == 0 or detail::indices_are_uniform_impl<Arg>(std::index_sequence<indices...>{})) and
    (sizeof...(indices) > 0 or detail::indices_are_uniform_impl<Arg>(std::make_index_sequence<index_count_v<Arg>>{}));


} // namespace OpenKalman::internal

#endif //OPENKALMAN_INDICES_ARE_UNIFORM_HPP
