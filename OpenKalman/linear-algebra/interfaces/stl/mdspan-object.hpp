/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025-2026 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "linear-algebra/interfaces/interfaces-defined.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to a generic std::mdspan.
   */
  template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
  struct object_traits<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
  {
    static const bool is_specialized = true;

    template<typename M>
    static constexpr decltype(auto)
    get_mdspan(M&& m)
    {
      return std::forward<M>(m);
    };
  };


  namespace internal
  {
    template<typename N, typename E, typename L, typename A>
    struct mdspan_base_object_traits
    {
      template<typename M>
      static constexpr decltype(auto)
      get_mdspan(M&& m) { return std::forward<M>(m); }

#ifdef __cpp_concepts
      template<typename M> requires get_constant_defined_for<stdex::mdspan<N, E, L, A>>
#else
      template<typename M, bool Enable = true, std::enable_if_t<
        Enable and get_constant_defined_for<stdex::mdspan<N, E, L, A>>, int> = 0>
#endif
      static constexpr auto
      get_constant(M&& m)
      {
        return m.accessor().access(m.data_handle(), 0);
      }
    };

  }

}


#endif
