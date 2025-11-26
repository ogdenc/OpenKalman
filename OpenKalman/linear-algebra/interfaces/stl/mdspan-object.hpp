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
 * \brief Definition of \ref object_traits for std::mdspan.
 */

#ifndef OPENKALMAN_INTERFACES_MDSPAN_OBJECT_TRAITS_HPP
#define OPENKALMAN_INTERFACES_MDSPAN_OBJECT_TRAITS_HPP

#include "basics/basics.hpp"
#include "linear-algebra/interfaces/object_traits.hpp"
#include "linear-algebra/functions/internal/constant_mdspan_policies.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to an std::mdspan.
   */
  template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
  struct object_traits<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
  {
    static const bool is_specialized = true;

    // An identity function for get_mdspan is not necessary because it will never be called.


    /**
     * \brief If AccessorPolicy indicates that the mdspan is constant, return the constant value.
     */
    static constexpr auto
#ifdef __cpp_concepts
    get_constant = [](const auto& m) requires std::same_as<AccessorPolicy, internal::accessor_constant<T>>
#else
    get_constant = [](const auto& m,
      std::enable_if_t<stdex::same_as<typename std::decay_t<decltype(m)>::accessor_type, internal::accessor_constant<T>>, int> = 0)
#endif
    {
      return *(m.accessor().data_handle());
    };

  };

}


#endif
