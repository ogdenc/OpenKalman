/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition of \ref StaticScalar.
 */

#ifndef OPENKALMAN_VALUE_STATICSCALAR_HPP
#define OPENKALMAN_VALUE_STATICSCALAR_HPP

#ifdef __cpp_concepts
#include <concepts>
#endif
#include "basics/global-definitions.hpp"
#include "linear-algebra/values/concepts/scalar.hpp"
#include "linear-algebra/values/functions/to_number.hpp"

namespace OpenKalman::value
{
  /**
   * \internal
   * \brief A defined \ref value::static_scalar
   * \tparam C A scalar type
   * \tparam constant Optional compile-time arguments for constructing C
   */
  template<typename C, auto...constant>
  struct StaticScalar;


#if not defined(__cpp_concepts) or not defined(__cpp_impl_three_way_comparison)
  namespace detail
  {
    template<typename Derived, typename C, typename = void, auto...constant>
    struct ScalarConstantImpl;


    template<typename Derived, typename C, auto...constant>
    struct ScalarConstantImpl<Derived, C, std::enable_if_t<(value::static_scalar<C> or sizeof...(constant) > 0) and
      std::bool_constant<(C{constant...}, true)>::value>, constant...>
    {
      static constexpr auto value {value::to_number(C{constant...})};
      using value_type = std::decay_t<decltype(value)>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }

      constexpr ScalarConstantImpl() = default;

      template<typename T, std::enable_if_t<value::static_scalar<T> and T::value == value, int> = 0>
      explicit constexpr ScalarConstantImpl(const T&) {};

      template<typename T, std::enable_if_t<value::static_scalar<T> and T::value == value, int> = 0>
      constexpr Derived& operator=(const T&) { return static_cast<Derived&>(*this); }

      constexpr auto operator+() { return static_cast<Derived&>(*this); }

      constexpr auto operator-() { return static_scalar_operation {std::negate<value_type>{}, static_cast<Derived&>(*this)}; }
    };


    template<typename Derived, typename C>
    struct ScalarConstantImpl<Derived, C, std::enable_if_t<value::dynamic_scalar<C>>>
    {
      using value_type = std::decay_t<decltype(value::to_number(std::declval<C>()))>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }

      template<typename T, std::enable_if_t<value::scalar<T>, int> = 0>
      explicit constexpr ScalarConstantImpl(const T& t) : value {value::to_number(t)} {};

      template<typename T, std::enable_if_t<value::scalar<T>, int> = 0>
      constexpr Derived& operator=(const T& t) { value = t; return static_cast<Derived&>(*this); }

    private:
      value_type value;
    };
  } // namespace detail
#endif


#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<value::scalar C, auto...constant> requires std::bool_constant<(C{constant...}, true)>::value
  struct StaticScalar<C, constant...>
  {
    static constexpr auto value {value::to_number(C{constant...})};

    using value_type = std::decay_t<decltype(value)>;

    using type = StaticScalar;

    constexpr operator value_type() const { return value; }

    constexpr value_type operator()() const { return value; }

    constexpr StaticScalar() = default;

    template<value::static_scalar T> requires (T::value == value)
    explicit constexpr StaticScalar(const T&) {};

    template<value::static_scalar T> requires (T::value == value)
    constexpr StaticScalar& operator=(const T&) { return *this; }

    template<value::scalar A, value::scalar B> requires
    std::same_as<A, StaticScalar> or std::same_as<B, StaticScalar>
    friend constexpr auto operator<=>(const A& a, const B& b)
    {
      if constexpr (std::same_as<A, StaticScalar>) return value <=> value::to_number(b);
      else return value::to_number(a) <=> value;
    }

    template<value::scalar A, value::scalar B> requires
    std::same_as<A, StaticScalar> or std::same_as<B, StaticScalar>
    friend constexpr auto operator==(const A& a, const B& b)
    {
      return std::is_eq(a <=> b);
    }
  };


  template<value::dynamic_scalar C>
  struct StaticScalar<C>
  {
    using value_type = std::decay_t<decltype(value::to_number(std::declval<C>()))>;

    constexpr operator value_type() const { return value; }

    constexpr value_type operator()() const { return value; }

    using type = StaticScalar;

    template<value::scalar T>
    explicit constexpr StaticScalar(const T& t) : value {value::to_number(t)} {};

    template<value::scalar T>
    constexpr StaticScalar& operator=(const T& t) { value = t; return *this; }

    template<value::scalar A, value::scalar B> requires
      std::same_as<A, StaticScalar> or std::same_as<B, StaticScalar>
    friend constexpr auto operator<=>(const A& a, const B& b)
    {
      return value::to_number(a) <=> value::to_number(b);
    }

    template<value::scalar A, value::scalar B> requires
    std::same_as<A, StaticScalar> or std::same_as<B, StaticScalar>
    friend constexpr auto operator==(const A& a, const B& b)
    {
      return std::is_eq(a <=> b);
    }

  private:
    value_type value;
  };
#else
  template<typename C, auto...constant>
  struct StaticScalar : detail::ScalarConstantImpl<StaticScalar<C, constant...>, C, void, constant...>
  {
  private:
    static_assert(value::scalar<C>);
    using Base = detail::ScalarConstantImpl<StaticScalar, C, void, constant...>;
  public:
    using Base::Base;
    using Base::operator=;
    using type = StaticScalar;

    template<typename A, typename B, std::enable_if_t<value::scalar<A> and value::scalar<B> and
      (std::is_same_v<A, StaticScalar> or std::is_same_v<B, StaticScalar>), int> = 0>
    friend constexpr auto operator==(const A& a, const B& b)
    {
      return value::to_number(a) == value::to_number(b);
    }

    template<typename A, typename B, std::enable_if_t<value::scalar<A> and value::scalar<B> and
      (std::is_same_v<A, StaticScalar> or std::is_same_v<B, StaticScalar>), int> = 0>
    friend constexpr auto operator!=(const A& a, const B& b)
    {
      return value::to_number(a) != value::to_number(b);
    }

    template<typename A, typename B, std::enable_if_t<value::scalar<A> and value::scalar<B> and
      (std::is_same_v<A, StaticScalar> or std::is_same_v<B, StaticScalar>), int> = 0>
    friend constexpr auto operator<(const A& a, const B& b)
    {
      return value::to_number(a) < value::to_number(b);
    }

    template<typename A, typename B, std::enable_if_t<value::scalar<A> and value::scalar<B> and
      (std::is_same_v<A, StaticScalar> or std::is_same_v<B, StaticScalar>), int> = 0>
    friend constexpr auto operator>(const A& a, const B& b)
    {
      return value::to_number(a) > value::to_number(b);
    }

    template<typename A, typename B, std::enable_if_t<value::scalar<A> and value::scalar<B> and
      (std::is_same_v<A, StaticScalar> or std::is_same_v<B, StaticScalar>), int> = 0>
    friend constexpr auto operator<=(const A& a, const B& b)
    {
      return value::to_number(a) <= value::to_number(b);
    }

    template<typename A, typename B, std::enable_if_t<value::scalar<A> and value::scalar<B> and
      (std::is_same_v<A, StaticScalar> or std::is_same_v<B, StaticScalar>), int> = 0>
    friend constexpr auto operator>=(const A& a, const B& b)
    {
      return value::to_number(a) >= value::to_number(b);
    }

  };
#endif


  /**
   * \internal
   * \brief Deduction guide for \ref StaticScalar where T is already a \ref value::scalar.
   */
#ifdef __cpp_concepts
  template<value::scalar T>
#else
  template<typename T, std::enable_if_t<value::scalar<T>, int> = 0>
#endif
  explicit StaticScalar(const T&) -> StaticScalar<std::decay_t<T>>;


} // namespace OpenKalman::value


#endif //OPENKALMAN_VALUE_STATICSCALAR_HPP
