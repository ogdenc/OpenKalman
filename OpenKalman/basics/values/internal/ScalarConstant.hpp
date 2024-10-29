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
 * \brief Definition of \ref ScalarConstant.
 */

#ifndef OPENKALMAN_SCALARCONSTANT_HPP
#define OPENKALMAN_SCALARCONSTANT_HPP

#include "basics/values/scalars/scalar_constant.hpp"

namespace OpenKalman::values
{
  /**
   * \internal
   * \brief A defined scalar constant
   * \tparam C A constant type
   * \tparam constant Compile-time arguments for constructing C, if any.
   */
  template<typename C, auto...constant>
  struct ScalarConstant;


#if not defined(__cpp_concepts) or not defined(__cpp_impl_three_way_comparison)
  namespace detail
  {
    template<typename Derived, typename C, typename = void, auto...constant>
    struct ScalarConstantImpl;


    template<typename Derived, typename C, auto...constant>
    struct ScalarConstantImpl<Derived, C, std::enable_if_t<(scalar_constant<C, ConstantType::static_constant> or sizeof...(constant) > 0) and
      get_scalar_constant_value(C{constant...}) == get_scalar_constant_value(C{constant...})>, constant...>
    {
      static constexpr auto value {get_scalar_constant_value(C{constant...})};
      using value_type = std::decay_t<decltype(value)>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }

      constexpr ScalarConstantImpl() = default;

      template<typename T, std::enable_if_t<scalar_constant<T, ConstantType::static_constant> and T::value == value, int> = 0>
      explicit constexpr ScalarConstantImpl(const T&) {};

      template<typename T, std::enable_if_t<scalar_constant<T, ConstantType::static_constant> and T::value == value, int> = 0>
      constexpr Derived& operator=(const T&) { return static_cast<Derived&>(*this); }

      constexpr auto operator+() { return static_cast<Derived&>(*this); }

      constexpr auto operator-() { return scalar_constant_operation {std::negate<value_type>{}, static_cast<Derived&>(*this)}; }
    };


    template<typename Derived, typename C>
    struct ScalarConstantImpl<Derived, C, std::enable_if_t<scalar_constant<C, ConstantType::dynamic_constant>>>
    {
      using value_type = std::decay_t<decltype(get_scalar_constant_value(std::declval<C>()))>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }

      template<typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
      explicit constexpr ScalarConstantImpl(const T& t) : value {get_scalar_constant_value(t)} {};

      template<typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
      constexpr Derived& operator=(const T& t) { value = t; return static_cast<Derived&>(*this); }

    private:
      value_type value;
    };
  } // namespace detail
#endif


#if defined(__cpp_concepts) and defined(__cpp_impl_three_way_comparison)
  template<scalar_constant C, auto...constant> requires std::bool_constant<(C{constant...}, true)>::value
  struct ScalarConstant<C, constant...>
  {
    static constexpr auto value {get_scalar_constant_value(C{constant...})};

    using value_type = std::decay_t<decltype(value)>;

    using type = ScalarConstant;

    constexpr operator value_type() const { return value; }

    constexpr value_type operator()() const { return value; }

    constexpr ScalarConstant() = default;

    template<scalar_constant<ConstantType::static_constant> T> requires (T::value == value)
    explicit constexpr ScalarConstant(const T&) {};

    template<scalar_constant<ConstantType::static_constant> T> requires (T::value == value)
    constexpr ScalarConstant& operator=(const T&) { return *this; }

    template<scalar_constant A, scalar_constant B> requires
    std::same_as<A, ScalarConstant> or std::same_as<B, ScalarConstant>
    friend constexpr auto operator<=>(const A& a, const B& b)
    {
      if constexpr (std::same_as<A, ScalarConstant>) return value <=> get_scalar_constant_value(b);
      else return get_scalar_constant_value(a) <=> value;
    }

    template<scalar_constant A, scalar_constant B> requires
    std::same_as<A, ScalarConstant> or std::same_as<B, ScalarConstant>
    friend constexpr auto operator==(const A& a, const B& b)
    {
      return std::is_eq(a <=> b);
    }
  };


  template<scalar_constant<ConstantType::dynamic_constant> C>
  struct ScalarConstant<C>
  {
    using value_type = std::decay_t<decltype(get_scalar_constant_value(std::declval<C>()))>;

    constexpr operator value_type() const { return value; }

    constexpr value_type operator()() const { return value; }

    using type = ScalarConstant;

    template<scalar_constant T>
    explicit constexpr ScalarConstant(const T& t) : value {get_scalar_constant_value(t)} {};

    template<scalar_constant T>
    constexpr ScalarConstant& operator=(const T& t) { value = t; return *this; }

    template<scalar_constant A, scalar_constant B> requires
      std::same_as<A, ScalarConstant> or std::same_as<B, ScalarConstant>
    friend constexpr auto operator<=>(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) <=> get_scalar_constant_value(b);
    }

    template<scalar_constant A, scalar_constant B> requires
    std::same_as<A, ScalarConstant> or std::same_as<B, ScalarConstant>
    friend constexpr auto operator==(const A& a, const B& b)
    {
      return std::is_eq(a <=> b);
    }

  private:
    value_type value;
  };
#else
  template<typename C, auto...constant>
  struct ScalarConstant : detail::ScalarConstantImpl<ScalarConstant<C, constant...>, C, void, constant...>
  {
  private:
    static_assert(scalar_constant<C>);
    using Base = detail::ScalarConstantImpl<ScalarConstant, C, void, constant...>;
  public:
    using Base::Base;
    using Base::operator=;
    using type = ScalarConstant;

    template<typename A, typename B, std::enable_if_t<scalar_constant<A> and scalar_constant<B> and
      (std::is_same_v<A, ScalarConstant> or std::is_same_v<B, ScalarConstant>), int> = 0>
    friend constexpr auto operator==(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) == get_scalar_constant_value(b);
    }

    template<typename A, typename B, std::enable_if_t<scalar_constant<A> and scalar_constant<B> and
      (std::is_same_v<A, ScalarConstant> or std::is_same_v<B, ScalarConstant>), int> = 0>
    friend constexpr auto operator!=(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) != get_scalar_constant_value(b);
    }

    template<typename A, typename B, std::enable_if_t<scalar_constant<A> and scalar_constant<B> and
      (std::is_same_v<A, ScalarConstant> or std::is_same_v<B, ScalarConstant>), int> = 0>
    friend constexpr auto operator<(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) < get_scalar_constant_value(b);
    }

    template<typename A, typename B, std::enable_if_t<scalar_constant<A> and scalar_constant<B> and
      (std::is_same_v<A, ScalarConstant> or std::is_same_v<B, ScalarConstant>), int> = 0>
    friend constexpr auto operator>(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) > get_scalar_constant_value(b);
    }

    template<typename A, typename B, std::enable_if_t<scalar_constant<A> and scalar_constant<B> and
      (std::is_same_v<A, ScalarConstant> or std::is_same_v<B, ScalarConstant>), int> = 0>
    friend constexpr auto operator<=(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) <= get_scalar_constant_value(b);
    }

    template<typename A, typename B, std::enable_if_t<scalar_constant<A> and scalar_constant<B> and
      (std::is_same_v<A, ScalarConstant> or std::is_same_v<B, ScalarConstant>), int> = 0>
    friend constexpr auto operator>=(const A& a, const B& b)
    {
      return get_scalar_constant_value(a) >= get_scalar_constant_value(b);
    }

  };
#endif


  /**
   * \internal
   * \brief Deduction guide for \ref ScalarConstant where T is a scalar constant.
   */
#ifdef __cpp_concepts
  template<scalar_constant T>
#else
  template<typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
#endif
  explicit ScalarConstant(const T&) -> ScalarConstant<std::decay_t<T>>;


} // namespace OpenKalman::values

#endif //OPENKALMAN_SCALARCONSTANT_HPP
