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
 * \brief Definition of \ref static_scalar_operation.
 */

#ifndef OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP
#define OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP

#include "basics/global-definitions.hpp"
#include "linear-algebra/values/concepts/scalar.hpp"

namespace OpenKalman::value
{
#ifndef __cpp_concepts
  template<typename Operation, typename...Ts> struct static_scalar_operation;


  namespace detail
  {
    template<typename Operation, typename = void, typename...Ts>
    struct scalar_constant_operation_impl
    {
      explicit constexpr scalar_constant_operation_impl(const Operation&, const Ts&...) {};
    };


    // n-ary, all arguments calculable at compile time
    template<typename Operation, typename...Ts>
    struct scalar_constant_operation_impl<Operation, std::enable_if_t<
      (value::static_scalar<Ts> and ...) and OpenKalman::internal::constexpr_n_ary_function<Operation, Ts...>>, Ts...>
    {
      constexpr scalar_constant_operation_impl() = default;
      explicit constexpr scalar_constant_operation_impl(const Operation&, const Ts&...) {};
      static constexpr auto value = Operation{}(Ts::value...);
      using value_type = std::decay_t<decltype(value)>;
      using type = static_scalar_operation<Operation, Ts...>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }
      constexpr auto operator+() { return static_cast<type&>(*this); }
      constexpr auto operator-() { return static_scalar_operation {std::negate<value_type>{}, static_cast<type&>(*this)}; }
    };


    // n-ary, not all arguments calculable at compile time
    template<typename Operation, typename...Ts>
    struct scalar_constant_operation_impl<Operation, std::enable_if_t<(value::scalar<Ts> and ...) and
      std::is_invocable_v<const Operation&, decltype(value::to_number(std::declval<const Ts&>()))...> and
      (not (value::static_scalar<Ts> and ...) or
      not std::is_default_constructible_v<Operation> or not OpenKalman::internal::constexpr_n_ary_function<Operation, Ts...>)>, Ts...>
    {
      explicit constexpr scalar_constant_operation_impl(const Operation& op, const Ts&...ts)
        : value {op(value::to_number(ts)...)} {};

      using value_type = std::decay_t<decltype(std::declval<const Operation&>()(value::to_number(std::declval<const Ts&>())...))>;
      using type = static_scalar_operation<Operation, Ts...>;
      constexpr operator value_type() const { return value; }
      constexpr value_type operator()() const { return value; }

    private:

      value_type value;
    };
  }
#endif


  /**
   * \internal
   * \brief An operation involving some number of \ref value::scalar values.
   * \tparam Operation An operation taking <code>sizeof...(Ts)</code> parameters
   * \tparam Ts A set of \ref value::scalar types
   */
#ifdef __cpp_concepts
  template<typename Operation, typename...Ts>
  struct static_scalar_operation
  {
    explicit constexpr static_scalar_operation(const Operation&, const Ts&...) {};
  };


  /**
   * \internal
   * \brief An operation involving some number of static \ref value::scalar values.
   */
  template<typename Operation, value::static_scalar...Ts> requires
    internal::constexpr_n_ary_function<Operation, Ts...>
  struct static_scalar_operation<Operation, Ts...>
  {
    constexpr static_scalar_operation() = default;
    explicit constexpr static_scalar_operation(const Operation&, const Ts&...) {};
    static constexpr auto value = Operation{}(Ts::value...);
    using value_type = std::decay_t<decltype(value)>;
    using type = static_scalar_operation;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }

  };


  /**
   * \internal
   * \brief An operation involving some number of \ref value::scalar values, not all of which are static.
   */
  template<typename Operation, value::scalar...Ts> requires
    requires(const Operation& op, const Ts&...ts) { op(value::to_number(ts)...); } and
    (not (value::static_scalar<Ts> and ...) or not internal::constexpr_n_ary_function<Operation, Ts...>)
  struct static_scalar_operation<Operation, Ts...>
  {
    explicit constexpr static_scalar_operation(const Operation& op, const Ts&...ts) : value {op(value::to_number(ts)...)} {};
    using value_type = std::decay_t<decltype(std::declval<const Operation&>()(value::to_number(std::declval<const Ts&>())...))>;
    using type = static_scalar_operation;
    constexpr operator value_type() const { return value; }
    constexpr value_type operator()() const { return value; }

  private:

    value_type value;
  };
#else
  template<typename Operation, typename...Ts>
  struct static_scalar_operation : detail::scalar_constant_operation_impl<Operation, void, Ts...>
  {
  private:
    using Base = detail::scalar_constant_operation_impl<Operation, void, Ts...>;
  public:
    using Base::Base;
  };
#endif


  //-- Deduction guides --//

  /**
   * \internal
   * \brief Deduction guide.
   */
  template<typename Operation, typename...Ts>
  explicit static_scalar_operation(const Operation&, const Ts&...) -> static_scalar_operation<Operation, Ts...>;


  /**
   * \internal
   * \brief Helper template for \ref static_scalar_operation.
   */
#ifdef __cpp_concepts
  template<typename Operation, value::static_scalar...Ts>
#else
  template<typename Operation, typename...Ts>
#endif
  constexpr auto scalar_constant_operation_v = static_scalar_operation<Operation, Ts...>::value;


} // namespace OpenKalman::value

#endif //OPENKALMAN_SCALAR_CONSTANT_OPERATION_HPP
