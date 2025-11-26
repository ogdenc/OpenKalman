/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref collections::get.
 */

#ifndef OPENKALMAN_COLLECTIONS_GET_HPP
#define OPENKALMAN_COLLECTIONS_GET_HPP

#include "collections/functions/get_element.hpp"

namespace OpenKalman::collections
{
  namespace detail_get
  {
    template<std::size_t i>
    struct get_impl
    {
#ifdef __cpp_concepts
      template<typename Arg> requires
        requires { get_element(std::declval<Arg>(), std::integral_constant<std::size_t, i>{}); }
#else
      template<typename Arg, typename = std::void_t<decltype(get_element(std::declval<Arg>(), std::integral_constant<std::size_t, i>{}))>>
#endif
      constexpr decltype(auto)
      operator() [[nodiscard]] (Arg&& arg) const
      {
        return get_element(std::forward<Arg>(arg), std::integral_constant<std::size_t, i>{});
      }
    };

  }


  /**
   * \brief A generalization of std::get, where the index is known at compile time
   * \note This function performs no runtime bounds checking.
   */
  template<std::size_t i>
  inline constexpr detail_get::get_impl<i>
  get;


}

#endif
