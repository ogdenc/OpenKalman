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
 * \brief Definition for \ref fill_components function.
 */

#ifndef OPENKALMAN_FILL_COMPONENTS_HPP
#define OPENKALMAN_FILL_COMPONENTS_HPP

#include "linear-algebra/concepts/internal/layout_mapping_policy.hpp"

namespace OpenKalman
{
  namespace detail
  {
    // \todo this is nonsensical right now, but can be converted to a generic function for copying elements
    template<std::size_t rank = 0, typename U, typename Indices, typename S>
    static constexpr void
    fill_components_impl(U& u, const Indices& indices, const S& s)
    {
      if constexpr (rank < std::rank_v<T>)
      {
        auto i = values::to_value_type(collections::get_element(indices, std::integral_constant<std::size_t, rank>{}));
        set_component_impl<rank + 1>(u[i], indices, s);
      }
      else
      {
        u = s;
      }
    }
  }


  /**
   * \overload
   * \brief Fill the components of an object from a list of scalar values.
   * \details The scalar components are listed in the specified layout order, as follows:
   * - \ref std::layout_left: column-major;
   * - \ref std::layout_right: row-major (the default).
   * \tparam layout The \ref layout_mapping_policy of Args and the resulting object (std::layout_right if unspecified).
   * \param arg The object to be modified.
   * \param s Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<internal::layout_mapping_policy layout = stdex::layout_right, indexible Arg, values::number ... S> requires
    (std::same_as<layout, stdex::layout_right> or std::same_as<layout, stdex::layout_left>) and
    internal::may_hold_components<Arg, S...> and
    (sizeof...(S) == 0 or interface::fill_components_defined_for<Arg, layout, std::add_lvalue_reference_t<Arg>, S...>)
  inline Arg&&
#else
  template<typename layout layout = stdex::layout_right, typename Arg, typename...S, std::enable_if_t<
    indexible<Arg> and (values::number<S> and ...) and
    (std::same_as<layout, stdex::layout_right> or std::same_as<layout, stdex::layout_left>) and
    internal::may_hold_components<Arg, S...> and
    (sizeof...(S) == 0 or interface::fill_components_defined_for<Arg, layout, std::add_lvalue_reference_t<Arg>, S...>), int> = 0>
  inline Arg&&
#endif
  fill_components(Arg&& arg, S...s)
  {
    if constexpr (sizeof...(S) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = element_type_of_t<Arg>;
      using Trait = interface::library_interface<stdex::remove_cvref_t<Arg>>;
      Trait::template fill_components<layout>(arg, static_cast<const Scalar>(s)...);
      return std::forward<Arg>(arg);
    }
    // \todo add a facility for when the interface is not defined
  }


}

#endif
