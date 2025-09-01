/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref indexible_object_traits for C++ arrays.
 */

#ifndef OPENKALMAN_INTERFACES_ARRAYS_INDEXIBLE_OBJECT_TRAITS_HPP
#define OPENKALMAN_INTERFACES_ARRAYS_INDEXIBLE_OBJECT_TRAITS_HPP

#include "linear-algebra/interfaces/default/indexible_object_traits.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to standard c++ arrays of any rank.
   */
#ifdef __cpp_concepts
  template<typename T> requires std::is_array_v<T> and (std::extent_v<T> > 0)
  struct indexible_object_traits<T>
#else
  template<typename T>
  struct indexible_object_traits<T, std::enable_if_t<std::is_array_v<T> and (std::extent_v<T> > 0)>>
#endif
  {
    /**
     * \sa scalar_type_of
     */
    using scalar_type = std::remove_all_extents_t<T>;


    /**
     * \sa OpenKalman::index_count
     * \sa OpenKalman::count_indices
     */
    static constexpr auto
    count_indices = std::rank<T>{};

  private:

    template<std::size_t...i>
    static constexpr auto
    get_pattern_collection_impl(std::index_sequence<i...>) { return std::tuple {std::extent<T, i>{}...}; }

  public:

    /**
     * \returns A tuple of extents.
     */
    static constexpr auto
    get_pattern_collection = [](const T&) -> coordinates::pattern_collection auto
    {
      return get_pattern_collection_impl(std::make_index_sequence<std::rank_v<T>>{});
    };


    // nested_object is not defined.
    // get_constant is not defined.
    // get_constant_diagonal is not defined.
    // one_dimensional is not defined.
    // is_square is not defined.
    // is_triangular is not defined.
    // is_triangular_adapter is not defined.
    // is_hermitian is not defined.
    // is_hermitian_adapter_type is not defined.


    /**
     * \brief Whether T is a writable, self-contained matrix or array.
     */
    static constexpr bool
    is_writable = false;


    /**
     * \brief Pointer to the first element of the array.
     */
    static constexpr auto
    raw_data = [](auto&& t) { return std::addressof(t[0]); };


    /**
     * \brief The standard C++ layout is right (row-major).
     */
    static constexpr data_layout
    layout = data_layout::right;


    // strides is not defined.

  };

}


#endif
