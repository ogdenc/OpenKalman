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
   * - \ref Layout::left: column-major;
   * - \ref Layout::right: row-major (the default).
   * \tparam layout The \ref Layout of Args and the resulting object (\ref Layout::right if unspecified).
   * \param arg The object to be modified.
   * \param s Scalar values to fill the new matrix.
   */
#ifdef __cpp_concepts
  template<Layout layout = Layout::right, writable Arg, scalar_type ... S>
    requires (layout == Layout::right or layout == Layout::left) and internal::may_hold_components<Arg, S...>
  inline Arg&&
#else
  template<Layout layout = Layout::right, typename Arg, typename...S, std::enable_if_t<
    writable<Arg> and (scalar_type<S> and ...) and
    (layout == Layout::right or layout == Layout::left) and internal::may_hold_components<Arg, S...>, int> = 0>
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


} // namespace OpenKalman

#endif //OPENKALMAN_FILL_COMPONENTS_HPP
