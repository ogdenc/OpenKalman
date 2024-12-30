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
 * \brief Definition of \ref value::operation.
 */

#ifndef OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP
#define OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP

#include "basics/global-definitions.hpp"
#include "linear-algebra/values/concepts/value.hpp"

namespace OpenKalman::value
{
  /**
   * \brief An operation involving some number of \ref value::value "values".
   * \details In this unspecialized case, the operation is not constant-evaluated.
   * \tparam Operation An operation taking <code>sizeof...(Ts)</code> parameters
   * \tparam Args A set of \ref value::value types
   */
#ifdef __cpp_concepts
  template<typename Operation, typename...Args>
  struct operation
  {
    explicit constexpr operation(const Operation& op, const Args&...args) : value {op(value::to_number(args)...)} {};
    using value_type = std::decay_t<decltype(std::declval<const Operation&>()(value::to_number(std::declval<const Args&>())...))>;
    using type = operation;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }

  private:

    value_type value;
  };


  /**
   * \brief A constant-evaluated operation involving some number of \ref value::fixed "fixed values".
   */
  template<typename Operation, value::fixed...Args> requires
    requires { typename std::bool_constant<(Operation{}(value::to_number(Args{})...), true)>; }
  struct operation<Operation, Args...>
  {
    constexpr operation() = default;
    explicit constexpr operation(const Operation&, const Args&...) {};
    static constexpr auto value = Operation{}(value::to_number(Args{})...);
    using value_type = std::decay_t<decltype(value)>;
    using type = operation;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }

  };
#else
  template<typename Operation, typename...Args>
  struct operation;


  namespace detail
  {
    // n-ary, not calculable at compile time
    template<typename Operation, typename = void, typename...Ts>
    struct constant_operation_impl
    {
      explicit constexpr constant_operation_impl(const Operation& op, const Ts&...ts)
        : value {op(value::to_number(ts)...)} {};

      using value_type = std::decay_t<decltype(std::declval<const Operation&>()(value::to_number(std::declval<const Ts&>())...))>;
      using type = operation<Operation, Ts...>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }

    private:

      value_type value;
    };


    // n-ary, all arguments calculable at compile time
    template<typename Operation, typename...Ts>
    struct constant_operation_impl<Operation, std::enable_if_t<
      (value::fixed<Ts> and ...) and std::bool_constant<(Operation{}(value::to_number(Ts{})...), true)>::value>, Ts...>
    {
      constexpr constant_operation_impl() = default;
      explicit constexpr constant_operation_impl(const Operation&, const Ts&...) {};
      static constexpr auto value = Operation{}(value::to_number(Ts{})...);
      using value_type = std::decay_t<decltype(value)>;
      using type = operation<Operation, Ts...>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }
      constexpr auto operator+() { return static_cast<type&>(*this); }
      constexpr auto operator-() { return operation {std::negate<value_type>{}, static_cast<type&>(*this)}; }
    };
  }


  template<typename Operation, typename...Args>
  struct operation : detail::constant_operation_impl<Operation, void, Args...>
  {
  private:
    using Base = detail::constant_operation_impl<Operation, void, Args...>;
  public:
    using Base::Base;
  };
#endif


  /**
   * \brief Deduction guide.
   */
  template<typename Operation, typename...Args>
  explicit operation(const Operation&, const Args&...) -> operation<Operation, Args...>;

} // namespace OpenKalman::value

#endif //OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP