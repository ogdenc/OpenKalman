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
 * \brief Definition for has_static_strides function.
 */

#ifndef OPENKALMAN_HAS_STATIC_STRIDES_HPP
#define OPENKALMAN_HAS_STATIC_STRIDES_HPP

namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename Strides, std::size_t...ix>
    constexpr bool has_static_strides_i(std::index_sequence<ix...>)
    {
      return (values::fixed<collections::collection_element_t<ix, Strides>> and ...);
    };

    template<typename Strides>
    constexpr bool has_static_strides_impl()
    {
      return has_static_strides_i<Strides>(std::make_index_sequence<collections::size_of_v<Strides>>{});
    };
  }


  /**
   * \brief Specifies that T has strides that are known at compile time.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_static_strides =
#else
  constexpr bool has_static_strides =
#endif
    detail::has_static_strides_impl<decltype(internal::strides(std::declval<T>()))>();


}

#endif
