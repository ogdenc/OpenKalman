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
 * \brief Definitions for value arithmetic.
 */

#ifndef OPENKALMAN_VALUE_ARITHMETIC_HPP
#define OPENKALMAN_VALUE_ARITHMETIC_HPP

#include <functional>
#include "values/concepts/fixed.hpp"
#include "values/concepts/value.hpp"
#include "to_number.hpp"
#include "values/classes/operation.hpp"

namespace OpenKalman::value
{
  namespace detail
  {
    template<typename Arg1, typename Arg2>
    using binary_common_type = std::common_type<number_type_of_t<Arg1>, number_type_of_t<Arg2>>;

    template<typename Arg1, typename Arg2>
    using binary_common_type_t = std::common_type_t<number_type_of_t<Arg1>, number_type_of_t<Arg2>>;


#ifndef __cpp_concepts
    template<typename Arg1, typename Arg2, typename = void>
    struct value_common_with_impl : std::false_type {};

    template<typename Arg1, typename Arg2>
    struct value_common_with_impl<Arg1, Arg2, std::void_t<typename std::common_type<
      std::decay_t<decltype(value::to_number(std::declval<Arg1>()))>,
      std::decay_t<decltype(value::to_number(std::declval<Arg2>()))>>::type>>
      : std::true_type {};
#endif


    template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
    concept value_common_with = value<Arg1> and value<Arg2> and std::common_with<number_type_of_t<Arg1>, number_type_of_t<Arg2>>;
#else
    constexpr bool value_common_with = value_common_with_impl<Arg1, Arg2>::value;
#endif
  }


#ifdef __cpp_concepts
  template<value Arg>
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
#endif
  constexpr Arg&& operator+(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<value Arg> requires requires(Arg arg) { -to_number(std::move(arg)); }
#else
  template<typename Arg, std::enable_if_t<value<Arg>, int> = 0>
#endif
  constexpr auto operator-(Arg arg)
  {
    if constexpr (fixed<Arg>)
      return operation {std::negate{}, std::move(arg)};
    else
      return -to_number(std::move(arg));
  }


#ifdef __cpp_concepts
  template<value Arg1, detail::value_common_with<Arg1> Arg2> requires
    std::invocable<std::plus<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::value_common_with<Arg1, Arg2> and
    std::is_invocable<std::plus<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>::value, int> = 0>
#endif
  constexpr auto operator+(Arg1 arg1, Arg2 arg2)
  {
    using Common = detail::binary_common_type_t<Arg1, Arg2>;
    if constexpr (fixed<Arg1> and fixed<Arg2>) return operation {std::plus<Common>{}, std::move(arg1), std::move(arg2)};
    else return static_cast<Common>(to_number(std::move(arg1))) + static_cast<Common>(to_number(std::move(arg2)));
  }


#ifdef __cpp_concepts
  template<value Arg1, detail::value_common_with<Arg1> Arg2> requires
    std::invocable<std::minus<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::value_common_with<Arg1, Arg2> and
    std::is_invocable<std::minus<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>::value, int> = 0>
#endif
  constexpr auto operator-(Arg1 arg1, Arg2 arg2)
  {
    using Common = detail::binary_common_type_t<Arg1, Arg2>;
    if constexpr (fixed<Arg1> and fixed<Arg2>) return operation {std::minus<Common>{}, std::move(arg1), std::move(arg2)};
    else return static_cast<Common>(to_number(std::move(arg1))) - static_cast<Common>(to_number(std::move(arg2)));
  }


#ifdef __cpp_concepts
  template<value Arg1, detail::value_common_with<Arg1> Arg2> requires
    std::invocable<std::multiplies<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::value_common_with<Arg1, Arg2> and
    std::is_invocable<std::multiplies<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>::value, int> = 0>
#endif
  constexpr auto operator*(Arg1 arg1, Arg2 arg2)
  {
    using Common = detail::binary_common_type_t<Arg1, Arg2>;
    if constexpr (fixed<Arg1> and fixed<Arg2>) return operation {std::multiplies<Common>{}, std::move(arg1), std::move(arg2)};
    else return static_cast<Common>(to_number(std::move(arg1))) * static_cast<Common>(to_number(std::move(arg2)));
  }


#ifdef __cpp_concepts
  template<value Arg1, detail::value_common_with<Arg1> Arg2> requires
    std::invocable<std::divides<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<detail::value_common_with<Arg1, Arg2> and
    std::is_invocable<std::divides<detail::binary_common_type_t<Arg1, Arg2>>, Arg1&&, Arg2&&>::value, int> = 0>
#endif
  constexpr auto operator/(Arg1 arg1, Arg2 arg2)
  {
    using Common = detail::binary_common_type_t<Arg1, Arg2>;
    if constexpr (fixed<Arg1> and fixed<Arg2>) return operation {std::divides<Common>{}, std::move(arg1), std::move(arg2)};
    else return static_cast<Common>(to_number(std::move(arg1))) / static_cast<Common>(to_number(std::move(arg2)));
  }


#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<value A, detail::value_common_with<A> B> requires
    std::invocable<std::compare_three_way, number_type_of_t<A>, number_type_of_t<B>>
  constexpr auto operator<=>(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) <=> static_cast<Common>(to_number(b));
  }

  template<value A, detail::value_common_with<A> B> requires
    std::invocable<std::compare_three_way, number_type_of_t<A>, number_type_of_t<B>>
  constexpr bool operator==(const A& a, const B& b)
  {
    return std::is_eq(a <=> b);
  }
#else
  template<typename A, typename B, std::enable_if_t<detail::value_common_with<A, B> and
   std::is_invocable<std::equal_to<detail::binary_common_type_t<A, B>>, const A&, const B&>::value, int> = 0>
  constexpr bool operator==(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) == static_cast<Common>(to_number(b));
  }

  template<typename A, typename B, std::enable_if_t<detail::value_common_with<A, B> and
   std::is_invocable<std::not_equal_to<detail::binary_common_type_t<A, B>>, const A&, const B&>::value, int> = 0>
  constexpr bool operator!=(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) != static_cast<Common>(to_number(b));
  }

template<typename A, typename B, std::enable_if_t<detail::value_common_with<A, B> and
 std::is_invocable<std::less<detail::binary_common_type_t<A, B>>, const A&, const B&>::value, int> = 0>
  constexpr auto operator<(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) < static_cast<Common>(to_number(b));
  }

template<typename A, typename B, std::enable_if_t<detail::value_common_with<A, B> and
 std::is_invocable<std::greater<detail::binary_common_type_t<A, B>>, const A&, const B&>::value, int> = 0>
  constexpr auto operator>(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) > static_cast<Common>(to_number(b));
  }

template<typename A, typename B, std::enable_if_t<detail::value_common_with<A, B> and
 std::is_invocable<std::less_equal<detail::binary_common_type_t<A, B>>, const A&, const B&>::value, int> = 0>
  constexpr auto operator<=(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) <= static_cast<Common>(to_number(b));
  }

template<typename A, typename B, std::enable_if_t<detail::value_common_with<A, B> and
 std::is_invocable<std::greater_equal<detail::binary_common_type_t<A, B>>, const A&, const B&>::value, int> = 0>
  constexpr auto operator>=(const A& a, const B& b)
  {
    using Common = detail::binary_common_type_t<A, B>;
    return static_cast<Common>(to_number(a)) >= static_cast<Common>(to_number(b));
  }
#endif


} // namespace OpenKalman::value

#endif //OPENKALMAN_VALUE_ARITHMETIC_HPP
