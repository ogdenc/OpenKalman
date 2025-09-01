/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref layout_of.
 */

#ifndef OPENKALMAN_LAYOUT_OF_HPP
#define OPENKALMAN_LAYOUT_OF_HPP


namespace OpenKalman
{
#ifndef __cpp_lib_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_layout : std::false_type {};

    template<typename T>
      struct has_layout<T, std::void_t<decltype(interface::indexible_object_traits<T>::layout)>> : std::true_type {};
  }
#endif


  /**
   * \brief The row dimension of a matrix, expression, or array.
   * \note If the row dimension is dynamic, then <code>value</code> is \ref dynamic_size.
   * \tparam T The matrix, expression, or array.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct layout_of : std::integral_constant<data_layout, data_layout::none> {};


#ifdef __cpp_concepts
  template<typename T> requires requires { interface::indexible_object_traits<stdcompat::remove_cvref_t<T>>::layout; }
  struct layout_of<T>
#else
  template<typename T>
  struct layout_of<T, std::enable_if_t<detail::has_layout<std::decay_t<T>>::value>>
#endif
    : std::integral_constant<data_layout, interface::indexible_object_traits<stdcompat::remove_cvref_t<T>>::layout> {};


  /**
   * \brief helper template for \ref layout_of.
   */
  template<typename T>
  static constexpr auto layout_of_v = layout_of<T>::value;


}

#endif
