/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for scalar arithmetic.
 */

#ifndef OPENKALMAN_SCALAR_ARITHMETIC_HPP
#define OPENKALMAN_SCALAR_ARITHMETIC_HPP


namespace OpenKalman
{
  namespace internal
  {
    /**
     * \internal
     * \brief Whether T, a \ref scalar_constant, participates in user-defined constant arithmetic operations.
     * \details This is defined as <code>std::true_type</ref>for
     * - \ref internal::scalar_constant_operation,
     * - \ref internal::ScalarConstant,
     * - \ref constant_coefficient, and
     * - \ref constant_diagonal_coefficient
     */
    template<typename T>
    struct participates_in_constant_arithmetic;


    template<typename T>
    struct participates_in_constant_arithmetic : std::false_type {};


    template<typename Operation, typename...Ts>
    struct participates_in_constant_arithmetic<internal::scalar_constant_operation<Operation, Ts...>> : std::true_type {};


    template<Qualification b, typename C, auto...constant>
    struct participates_in_constant_arithmetic<internal::ScalarConstant<b, C, constant...>> : std::true_type {};
  } // namespace internal


  namespace detail
  {
    template<typename T>
#ifdef __cpp_concepts
    concept internal_constant =
#else
    constexpr bool internal_constant =
#endif
    internal::participates_in_constant_arithmetic<std::decay_t<T>>::value;

  } // namespace detail


#ifdef __cpp_concepts
  template<detail::internal_constant Arg>
#else
  template<typename Arg, std::enable_if_t<detail::internal_constant<Arg>, int> = 0>
#endif
  constexpr Arg&& operator+(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<detail::internal_constant Arg>
#else
  template<typename Arg, std::enable_if_t<detail::internal_constant<Arg>, int> = 0>
#endif
  constexpr auto operator-(const Arg& arg)
  {
    return internal::scalar_constant_operation {std::negate<>{}, arg};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
#endif
  constexpr auto operator+(const Arg1& arg1, const Arg2& arg2)
  {
    return internal::scalar_constant_operation {std::plus<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
#endif
  constexpr auto operator-(const Arg1& arg1, const Arg2& arg2)
  {
    return internal::scalar_constant_operation {std::minus<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
#endif
  constexpr auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    return internal::scalar_constant_operation {std::multiplies<>{}, arg1, arg2};
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires detail::internal_constant<Arg1> or detail::internal_constant<Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::internal_constant<Arg1> or detail::internal_constant<Arg2>, int> = 0>
#endif
  constexpr auto operator/(const Arg1& arg1, const Arg2& arg2)
  {
    return internal::scalar_constant_operation {std::divides<>{}, arg1, arg2};
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SCALAR_ARITHMETIC_HPP
