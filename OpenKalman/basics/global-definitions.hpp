/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021 Christopher Lee Ogden <ogden@gatech.edu>
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
#include <cstdint>
#include <limits>

namespace OpenKalman
{

  /**
   * \brief A constant indicating that the relevant dimension of a matrix has a size that is dynamic.
   * \details A dynamic dimension can be set, or change, during runtime and is not known at compile time.
   */
#ifndef __cpp_lib_span

  static constexpr std::size_t dynamic_size = std::numeric_limits<std::size_t>::max();
#endif


  /**
   * \brief The type of a triangular matrix.
   */
  enum struct TriangleType : int {
    lower, ///< A lower-left triangular matrix.
    upper, ///< An upper-right triangular matrix.
    diagonal, ///< A diagonal matrix (both a lower-left and an upper-right triangular matrix).
    none, ///< Neither upper, lower, or diagonal.
  };


  /**
   * \brief Whether a property is definitely known to apply at compile time (definitely) or not ruled out (maybe).
   */
  enum struct Likelihood : bool {
    maybe = false, ///< At compile time, the property is not ruled out.
    definitely = true, ///< At compile time, the property is known.
  };


  constexpr Likelihood operator!(Likelihood x)
  {
    return x == Likelihood::definitely ? Likelihood::maybe : Likelihood::definitely;
  }


  constexpr Likelihood operator&&(Likelihood x, Likelihood y)
  {
    return Likelihood {static_cast<bool>(x) && static_cast<bool>(y)};
  }


  constexpr Likelihood operator||(Likelihood x, Likelihood y)
  {
    return Likelihood {static_cast<bool>(x) || static_cast<bool>(y)};
  }


  /**
   * \brief The known/unknown status of a particular property at compile time.
   */
  enum struct CompileTimeStatus {
    any, ///< The property is either known or unknown at compile time.
    unknown, ///< The property is unknown at compile time but known at runtime.
    known, ///< The property is known at compile time.
  };



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


    // ------------------------- //
    //  constexpr_n_ary_function //
    // ------------------------- //

#ifdef __cpp_concepts
    template<typename Op, typename...Args>
    struct is_constexpr_n_ary_function : std::false_type {};

    template<typename Op, typename...Args>
    requires requires { typename std::bool_constant<0 == Op{}(std::decay_t<Args>::value...)>; }
    struct is_constexpr_n_ary_function<Op, Args...> : std::true_type {};

    template<typename Op, typename...Args>
    concept constexpr_n_ary_function = is_constexpr_n_ary_function<Op, Args...>::value;
#else
    template<typename Op, typename = void, typename...Args>
    struct is_constexpr_n_ary_function : std::false_type {};

    template<typename Op, typename...Args>
    struct is_constexpr_n_ary_function<Op, std::void_t<std::bool_constant<0 == Op{}(std::decay_t<Args>::value...)>>, Args...>
      : std::true_type {};

    template<typename Op, typename...Args>
    constexpr bool constexpr_n_ary_function = is_constexpr_n_ary_function<Op, void, Args...>::value;
#endif


  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_GLOBAL_DEFINITIONS_HPP
