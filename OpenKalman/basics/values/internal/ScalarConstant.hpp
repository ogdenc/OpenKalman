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
 * \brief Definition of \ref ScalarConstant.
 */

#ifndef OPENKALMAN_SCALARCONSTANT_HPP
#define OPENKALMAN_SCALARCONSTANT_HPP


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief A defined scalar constant
   * \tparam b The likelihood that the result is a constant
   * \tparam C A constant type
   * \tparam constant Compile-time arguments for constructing C, if any.
   */
  template<Qualification b, typename C, auto...constant>
  struct ScalarConstant;


#ifndef __cpp_concepts
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
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }

      constexpr ScalarConstantImpl() = default;

      template<typename T, std::enable_if_t<scalar_constant<T, ConstantType::static_constant> and T::value == value, int> = 0>
      explicit constexpr ScalarConstantImpl(const T&) {};

      template<typename T, std::enable_if_t<scalar_constant<T, ConstantType::static_constant> and T::value == value, int> = 0>
      constexpr Derived& operator=(const T&) { return static_cast<Derived&>(*this); }

      constexpr auto operator+() { return static_cast<Derived&>(*this); }

      constexpr auto operator-() { return scalar_constant_operation {std::negate<>{}, static_cast<Derived&>(*this)}; }
    };


    template<typename Derived, typename C>
    struct ScalarConstantImpl<Derived, C, std::enable_if_t<scalar_constant<C, ConstantType::dynamic_constant>>>
    {
      using value_type = std::decay_t<decltype(get_scalar_constant_value(std::declval<C>()))>;
      constexpr operator value_type() const noexcept { return value; }
      constexpr value_type operator()() const noexcept { return value; }

      template<typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
      explicit constexpr ScalarConstantImpl(const T& t) : value {get_scalar_constant_value(t)} {};

      template<typename T, std::enable_if_t<scalar_constant<T>, int> = 0>
      constexpr Derived& operator=(const T& t) { value = t; return static_cast<Derived&>(*this); }

    private:
      value_type value;
    };
  } // namespace detail
#endif


#ifdef __cpp_concepts
  template<Qualification b, scalar_constant C, auto...constant> requires std::bool_constant<(C{constant...}, true)>::value
  struct ScalarConstant<b, C, constant...>
  {
    static constexpr auto value {get_scalar_constant_value(C{constant...})};
    using value_type = std::decay_t<decltype(value)>;
    using type = ScalarConstant;
    static constexpr Qualification status = b;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }

    constexpr ScalarConstant() = default;

    template<scalar_constant<ConstantType::static_constant> T> requires (T::value == value)
    explicit constexpr ScalarConstant(const T&) {};

    template<scalar_constant<ConstantType::static_constant> T> requires (T::value == value)
    constexpr ScalarConstant& operator=(const T&) { return *this; }
  };


  template<Qualification b, scalar_constant<ConstantType::dynamic_constant> C>
  struct ScalarConstant<b, C>
  {
    using value_type = std::decay_t<decltype(get_scalar_constant_value(std::declval<C>()))>;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
    using type = ScalarConstant;
    static constexpr Qualification status = b;

    template<scalar_constant T>
    explicit constexpr ScalarConstant(const T& t) : value {get_scalar_constant_value(t)} {};

    template<scalar_constant T>
    constexpr ScalarConstant& operator=(const T& t) { value = t; return *this; }

  private:
    value_type value;
  };
#else
  template<Qualification b, typename C, auto...constant>
  struct ScalarConstant : detail::ScalarConstantImpl<ScalarConstant<b, C, constant...>, C, void, constant...>
  {
  private:
    static_assert(scalar_constant<C>);
    using Base = detail::ScalarConstantImpl<ScalarConstant, C, void, constant...>;
  public:
    using Base::Base;
    using Base::operator=;
    static constexpr Qualification status = b;
    using type = ScalarConstant;
  };
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_constant_status : std::false_type {};

    template<typename T>
    struct has_constant_status<T, std::enable_if_t<std::is_same_v<decltype(T::status), Qualification>>> : std::true_type {};
  }
#endif


  /**
   * \internal
   * \brief Deduction guide for \ref ScalarConstant where T has a <code>status</code> member.
   */
#ifdef __cpp_concepts
  template<scalar_constant T> requires requires { {T::status} -> std::same_as<Qualification>; }
#else
  template<typename T, std::enable_if_t<scalar_constant<T> and detail::has_constant_status<T>::value, int> = 0>
#endif
  explicit ScalarConstant(const T&) -> ScalarConstant<T::status, std::decay_t<T>>;


  /**
   * \internal
   * \brief Deduction guide for \ref ScalarConstant where T does not have a <code>status</code> member.
   */
#ifdef __cpp_concepts
  template<scalar_constant T> requires (not requires { {T::status} -> std::same_as<Qualification>; })
#else
  template<typename T, std::enable_if_t<scalar_constant<T> and not detail::has_constant_status<T>::value, int> = 0>
#endif
  explicit ScalarConstant(const T&) -> ScalarConstant<Qualification::unqualified, std::decay_t<T>>;


} // namespace OpenKalman::internal

#endif //OPENKALMAN_SCALARCONSTANT_HPP
