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


namespace OpenKalman
{
  /**
   * \overload
   * \brief Fill the components of an object from a list of scalar values.
   * \details The scalar components are listed in the specified layout order, as follows:
   * - \ref data_layout::left: column-major;
   * - \ref data_layout::right: row-major (the default).
   * \tparam layout The \ref data_layout of Args and the resulting object (\ref data_layout::right if unspecified).
   * \param arg The object to be modified.
   * \param s Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<data_layout layout = data_layout::right, indexible Arg, values::number ... S> requires
    (layout == data_layout::right or layout == data_layout::left) and internal::may_hold_components<Arg, S...> and
    (sizeof...(S) == 0 or interface::fill_components_defined_for<Arg, layout, std::add_lvalue_reference_t<Arg>, S...>)
  inline Arg&&
#else
  template<data_layout layout = data_layout::right, typename Arg, typename...S, std::enable_if_t<
    indexible<Arg> and (values::number<S> and ...) and (layout == data_layout::right or layout == data_layout::left) and
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
      using Scalar = scalar_type_of_t<Arg>;
      using Trait = interface::library_interface<std::decay_t<Arg>>;
      Trait::template fill_components<layout>(arg, static_cast<const Scalar>(s)...);
      return std::forward<Arg>(arg);
    }
  }


}

#endif
