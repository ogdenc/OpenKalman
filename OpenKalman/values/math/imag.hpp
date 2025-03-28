/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for value::imag.
 */

#ifndef OPENKALMAN_VALUE_IMAG_HPP
#define OPENKALMAN_VALUE_IMAG_HPP

#include "values/concepts/number.hpp"
#include "values/concepts/value.hpp"
#include "values/concepts/integral.hpp"
#include "values/traits/number_type_of_t.hpp"
#include "values/classes/operation.hpp"

namespace OpenKalman::value
{
  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct std_imag_defined : std::false_type {};

#ifdef __cpp_concepts
    template<typename T> requires requires(T t) { std::imag(t); }
    struct std_imag_defined<T> : std::true_type {};
#else
    template<typename T>
    struct std_imag_defined<T, std::void_t<decltype(std::imag(std::declval<T>()))>> : std::true_type {};
#endif


#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct imag_defined : std::false_type {};

#ifdef __cpp_concepts
    template<typename T> requires requires(T t) { imag(t); }
    struct imag_defined<T> : std::true_type {};
#else
    template<typename T>
    struct imag_defined<T, std::void_t<decltype(imag(std::declval<T>()))>> : std::true_type {};
#endif

  } // namespace detail


  /**
   * \brief A constexpr function to obtain the imaginary part of a (complex) number.
   * \details If arg is \ref value::complex "complex", arg must either match
   * <code>std::imag(value::to_number(arg))</code> or some defined function <code>imag(value::to_number(arg))</code>.
   * If arg is not \ref value::complex "complex" and no <code>imag</code> function is defined, the result will be
   * - <code>static_cast<double>(0)</code> if Arg is \ref value::integral "integral" or
   * - <code>static_cast<std::decay_t<Arg>>(0)</code> otherwise.
   */
#ifdef __cpp_concepts
  template<value Arg> requires (not value::complex<Arg>) or
    detail::std_imag_defined<number_type_of_t<Arg>>::value or detail::imag_defined<number_type_of_t<Arg>>::value
  constexpr value auto
#else
  template<typename Arg, std::enable_if_t<value<Arg> and (not value::complex<Arg> or
    detail::std_imag_defined<number_type_of_t<Arg>>::value or detail::imag_defined<number_type_of_t<Arg>>::value), int> = 0>
  constexpr auto
#endif
  imag(Arg arg)
  {
    if constexpr (not number<Arg>)
    {
      struct Op { constexpr auto operator()(number_type_of_t<Arg> a) const { return imag(std::move(a)); } };
      return operation {Op{}, std::move(arg)};
    }
    else if constexpr (detail::std_imag_defined<Arg&&>::value or detail::imag_defined<Arg&&>::value)
    {
      using std::imag;
      return imag(std::move(arg));
    }
    else
    {
      using Ret = std::conditional_t<integral<Arg>, double, std::decay_t<Arg>>;
      return static_cast<Ret>(0);
    }
  }


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_IMAG_HPP
