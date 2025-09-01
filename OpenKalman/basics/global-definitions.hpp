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
#include <limits>
#include <functional>

namespace OpenKalman
{

  /**
   * \brief A constant indicating that a size or index is dynamic.
   * \details A dynamic size or index can be set, or change, during runtime and is not known at compile time.
   */
#ifdef __cpp_lib_span
  inline constexpr std::size_t dynamic_size = std::dynamic_extent;
#else
  inline constexpr std::size_t dynamic_size = std::numeric_limits<std::size_t>::max();
#endif


  /**
   * \brief A constant indicating that a difference in sizes or indices is dynamic.
   * \details A dynamic difference can be set, or change, during runtime and is not known at compile time.
   */
  inline constexpr std::ptrdiff_t dynamic_difference = std::numeric_limits<std::ptrdiff_t>::max();


  /**
   * \brief The applicability of a concept, trait, or restraint.
   * \details Determines whether something is necessarily applicable, or alternatively just permissible, at compile time.
   * Examples:
   * - <code>square_shaped<T, applicability::guaranteed></code> means that T is known at compile time to be square shaped.
   * - <code>square_shaped<T, applicability::permitted></code> means that T <em>could</em> be square shaped,
   * but whether it actually <em>is</em> cannot be determined at compile time.
   */
  enum struct applicability : int {
    guaranteed, ///< The concept, trait, or restraint represents a compile-time guarantee.
    permitted, ///< The concept, trait, or restraint is permitted, but whether it applies is not necessarily known at compile time.
  };


  constexpr applicability operator not (applicability x)
  {
    return x == applicability::guaranteed ? applicability::permitted : applicability::guaranteed;
  }


  constexpr applicability operator and (applicability x, applicability y)
  {
    return x == applicability::guaranteed and y == applicability::guaranteed ? applicability::guaranteed : applicability::permitted;
  }


  constexpr applicability operator or (applicability x, applicability y)
  {
    return x == applicability::guaranteed or y == applicability::guaranteed ? applicability::guaranteed : applicability::permitted;
  }


  namespace internal
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

}

#endif
