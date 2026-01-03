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
#include "linear-algebra/enumerations.hpp"
#include "linear-algebra/interfaces/object_traits.hpp"
#include "to_diagonal_mdspan_policies.hpp"
#include "constant_mdspan_policies.hpp"

namespace OpenKalman::interface
{
  /**
   * \internal
   * \brief An interface to an std::mdspan.
   */
  template<typename T, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
  struct object_traits<stdex::mdspan<T, Extents, LayoutPolicy, AccessorPolicy>>
  {
  private:

    template<typename>
    struct is_diagonal_layout : std::false_type {};

    template<typename N>
    struct is_diagonal_layout<layout_to_diagonal<N>> : std::true_type {};

    template<typename>
    struct is_constant_accessor : std::false_type {};

    template<typename E>
    struct is_constant_accessor<constant_accessor<E>> : std::true_type {};

    template<typename>
    struct is_constant_diagonal_accessor : std::false_type {};

    template<typename E>
    struct is_constant_diagonal_accessor<to_diagonal_accessor<constant_accessor<E>>> : std::true_type {};

  public:

    static const bool is_specialized = true;


    template<typename M>
    static constexpr auto
    get_mdspan(M&& m)
    {
      return std::forward<M>(m);
    };


    static constexpr triangle_type
    triangle_type_value = is_diagonal_layout<LayoutPolicy>::value ? triangle_type::diagonal : triangle_type::none;


    /**
     * \brief If AccessorPolicy indicates that the mdspan is constant, return the constant value.
     */
#ifdef __cpp_concepts
    template<typename M> requires
      is_constant_accessor<typename M::accessor_type>::value or
      is_constant_diagonal_accessor<typename M::accessor_type>::value
#else
    template<typename M, std::enable_if_t<
      is_constant_accessor<typename M::accessor_type>::value or
      is_constant_diagonal_accessor<typename M::accessor_type>::value, int> = 0>
#endif
    static constexpr auto
    get_constant(const M& m)
    {
      if constexpr (is_constant_accessor<typename M::accessor_type>::value)
        return *(m.accessor().data_handle());
      else
        return *(m.accessor().nested_accessor().data_handle());
    }

  };

}


#endif
