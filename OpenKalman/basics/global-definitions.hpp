/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Global definitions for OpenKalman.
 */

#ifndef OPENKALMAN_GLOBAL_DEFINITIONS_HPP
#define OPENKALMAN_GLOBAL_DEFINITIONS_HPP

#include <type_traits>
#include <functional>

namespace OpenKalman::internal
{
  // ----------------------- //
  //  is_plus, is_multiplies //
  // ----------------------- //

  template<typename T>
  struct is_plus : std::false_type {};

  template<typename T>
  struct is_plus<std::plus<T>> : std::true_type {};

  template<typename T>
  struct is_multiplies : std::false_type {};

  template<typename T>
  struct is_multiplies<std::multiplies<T>> : std::true_type {};


  // ------------------------ //
  //  remove_rvalue_reference //
  // ------------------------ //

  /**
   * \brief If T is an rvalue reference, remove the reference. Otherwise, the result is T.
   */
  template<typename T>
  struct remove_rvalue_reference { using type = T; };


  /// \overload
  template<typename T>
  struct remove_rvalue_reference<T&&> { using type = T; };


  /**
   * \brief Helper type for \ref remove_rvalue_reference.
   */
  template<typename T>
  using remove_rvalue_reference_t = typename remove_rvalue_reference<T>::type;

}

#endif
